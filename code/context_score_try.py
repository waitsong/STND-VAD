import torch
from torch import nn
import torch.nn.functional as F

from hd_dgtm_try import HDDGTMixer


class ContextScoreTryModel(nn.Module):
    def __init__(
        self,
        num_blocks,
        batch_size,
        n_features,
        context_mode='lgcc',
        dilations=(1, 2, 4, 8, 16),
        num_normal_prompts=4,
        num_abnormal_prompts=13,
        num_normal_centers=16,
        normal_text_features=None,
        abnormal_text_features=None,
        train_text_features=False,
        text_on_raw_clip=True,
        local_kernel=5,
        anchor_k=8,
        score_w_init=(1.0, 0.5, 1.0),
        score_combo='all',
        sdf_prior_weight=0.2,
        sdf_prior_mode='mean',
        sdf_topk=3,
        sdf_logit_norm='none',
        score_fusion_mode='linear',
        test_score_mode='same',
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.mlp_dim = n_features
        self.context_mode = context_mode
        self.text_on_raw_clip = text_on_raw_clip
        self.score_combo = score_combo
        self.sdf_prior_weight = sdf_prior_weight
        self.sdf_prior_mode = sdf_prior_mode
        self.sdf_topk = sdf_topk
        self.sdf_logit_norm = sdf_logit_norm
        self.score_fusion_mode = score_fusion_mode
        self.test_score_mode = test_score_mode
        self.local_kernel = local_kernel if local_kernel % 2 == 1 else local_kernel + 1
        self.anchor_k = anchor_k
        self.k_abn = 5
        self.k_nor = self.k_abn

        self.encoder = HDDGTMixer(n_features, dilations=dilations, dropout=0.1, use_shift=False)
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()

        if normal_text_features is None:
            normal_text_features = torch.randn(num_normal_prompts, n_features)
        if abnormal_text_features is None:
            abnormal_text_features = torch.randn(num_abnormal_prompts, n_features)
        normal_text_features = F.normalize(normal_text_features.float(), p=2, dim=-1, eps=1e-6)
        abnormal_text_features = F.normalize(abnormal_text_features.float(), p=2, dim=-1, eps=1e-6)
        if train_text_features:
            self.normal_prompts = nn.Parameter(normal_text_features)
            self.abnormal_prompts = nn.Parameter(abnormal_text_features)
        else:
            self.register_buffer('normal_prompts', normal_text_features)
            self.register_buffer('abnormal_prompts', abnormal_text_features)

        self.text_adapter = nn.Linear(n_features, n_features, bias=False)
        nn.init.eye_(self.text_adapter.weight)
        self.semantic_temperature = nn.Parameter(torch.tensor(10.0))

        self.score_w = nn.Parameter(torch.tensor(score_w_init, dtype=torch.float32))
        self.context_w = nn.Parameter(torch.ones(2))
        self.tcde_w = nn.Parameter(torch.ones(5))
        self.context_scale = nn.Parameter(torch.tensor(1.0))
        self.sdf_gate = nn.Parameter(torch.tensor(0.0))
        self.context_heads = nn.ModuleDict(
            {
                'lgcc': self._make_context_head(2),
                'tcde': self._make_context_head(5),
                'dcac': self._make_context_head(3),
                'cpr': self._make_context_head(3),
                'sdf': self._make_context_head(3),
            }
        )
        self.disc_dim = 128
        self.disc_proj = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.disc_dim),
        )
        self.disc_head = nn.Sequential(
            nn.Conv1d(self.disc_dim, self.disc_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.disc_dim, 1, kernel_size=1),
        )
        self.normal_centers = nn.Parameter(torch.randn(num_normal_centers, self.disc_dim))
        self.context_predictor = nn.Sequential(
            nn.LayerNorm(n_features * 2),
            nn.Linear(n_features * 2, n_features),
            nn.GELU(),
            nn.Linear(n_features, n_features),
        )
        self.sigmoid = nn.Sigmoid()

        self.last_logits = None
        self.last_visual_logits = None
        self.last_text_logits = None
        self.last_context_logits = None
        self.last_center_distance = None
        self.last_sdf_evidence = None
        self.last_disc_features = None
        self.last_features = None
        self.last_magnitude = None
        self.last_component_scores = {}

    def _active_components(self):
        combos = {
            'all': ('mlp', 'prompt', 'context'),
            'mlp_prompt': ('mlp', 'prompt'),
            'mlp_context': ('mlp', 'context'),
            'prompt_context': ('prompt', 'context'),
            'context': ('context',),
            'mlp': ('mlp',),
            'prompt': ('prompt',),
        }
        return combos[self.score_combo]

    def _make_context_head(self, in_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

    def _semantic_logits(self, semantic_features, bs, ncrops, t):
        feat = F.normalize(self.text_adapter(semantic_features), p=2, dim=-1, eps=1e-6)
        normal_prompts = F.normalize(self.normal_prompts, p=2, dim=-1, eps=1e-6)
        abnormal_prompts = F.normalize(self.abnormal_prompts, p=2, dim=-1, eps=1e-6)
        normal_sim = torch.matmul(feat, normal_prompts.transpose(0, 1)).max(dim=-1)[0]
        abnormal_sim = torch.matmul(feat, abnormal_prompts.transpose(0, 1)).max(dim=-1)[0]
        text_logits = self.semantic_temperature.abs() * (abnormal_sim - normal_sim)
        return text_logits.view(bs, ncrops, t).mean(dim=1)

    def _temporal_avg(self, features, kernel):
        pad = kernel // 2
        x = features.transpose(1, 2)
        x = F.pad(x, (pad, pad), mode='replicate')
        return F.avg_pool1d(x, kernel_size=kernel, stride=1).transpose(1, 2)

    def _cos_dist(self, a, b):
        a = F.normalize(a, p=2, dim=-1, eps=1e-6)
        b = F.normalize(b, p=2, dim=-1, eps=1e-6)
        return 1.0 - torch.sum(a * b, dim=-1)

    def _standardize(self, logits):
        return (logits - logits.mean(dim=1, keepdim=True)) / (
            logits.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )

    def _anchor(self, features, base_scores, largest):
        k = min(self.anchor_k, features.size(1))
        idx = torch.topk(base_scores.detach(), k, dim=1, largest=largest)[1]
        idx = idx.unsqueeze(-1).expand(-1, -1, features.size(-1))
        return torch.gather(features, 1, idx).mean(dim=1)

    def _lgcc_evidence(self, features, base_scores):
        local = self._temporal_avg(features, self.local_kernel)
        normal_anchor = self._anchor(features, base_scores, largest=False)
        global_ctx = normal_anchor.unsqueeze(1).expand_as(features)
        local_dist = self._cos_dist(features, local)
        global_dist = self._cos_dist(features, global_ctx)
        return torch.stack((local_dist, global_dist), dim=-1)

    def _tcde_evidence(self, features):
        distances = []
        for kernel in (3, 5, 9, 17, 33):
            max_kernel = features.size(1) if features.size(1) % 2 == 1 else features.size(1) - 1
            max_kernel = max(1, max_kernel)
            local = self._temporal_avg(features, min(kernel, max_kernel))
            distances.append(self._cos_dist(features, local))
        return torch.stack(distances, dim=-1)

    def _dcac_evidence(self, features, base_scores):
        normal_anchor = self._anchor(features, base_scores, largest=False)
        abnormal_anchor = self._anchor(features, base_scores, largest=True)
        f = F.normalize(features, p=2, dim=-1, eps=1e-6)
        qn = F.normalize(normal_anchor, p=2, dim=-1, eps=1e-6).unsqueeze(1)
        qa = F.normalize(abnormal_anchor, p=2, dim=-1, eps=1e-6).unsqueeze(1)
        sim_abnormal = torch.sum(f * qa, dim=-1)
        sim_normal = torch.sum(f * qn, dim=-1)
        return torch.stack((sim_abnormal, -sim_normal, sim_abnormal - sim_normal), dim=-1)

    def _cpr_evidence(self, features, base_scores):
        local = self._temporal_avg(features, self.local_kernel)
        normal_anchor = self._anchor(features, base_scores, largest=False)
        global_ctx = normal_anchor.unsqueeze(1).expand_as(features)
        pred = self.context_predictor(torch.cat((local, global_ctx), dim=-1))
        pred_dist = self._cos_dist(features, pred)
        local_dist = self._cos_dist(features, local)
        global_dist = self._cos_dist(features, global_ctx)
        return torch.stack((pred_dist, local_dist, global_dist), dim=-1)

    def _sdf_evidence(self, features):
        disc_features = self.disc_proj(features)
        self.last_disc_features = disc_features
        feat = F.normalize(disc_features, p=2, dim=-1, eps=1e-6)
        centers = F.normalize(self.normal_centers, p=2, dim=-1, eps=1e-6)
        sim = torch.matmul(feat, centers.transpose(0, 1))
        top_sim = torch.topk(sim, min(self.sdf_topk, sim.size(-1)), dim=-1)[0]
        max_sim = top_sim[:, :, 0]
        mean_top_sim = top_sim.mean(dim=-1)
        soft_assign = torch.softmax(sim * 10.0, dim=-1)
        entropy = -torch.sum(soft_assign * torch.log(soft_assign + 1e-6), dim=-1)
        entropy = entropy / torch.log(sim.new_tensor(float(sim.size(-1))))
        center_distance = 1.0 - max_sim
        self.last_center_distance = center_distance
        evidence = torch.stack((center_distance, 1.0 - mean_top_sim, entropy), dim=-1)
        self.last_sdf_evidence = evidence
        return evidence

    def _sdf_prior(self, evidence):
        center_distance = evidence[:, :, 0]
        top_distance = evidence[:, :, 1]
        entropy = evidence[:, :, 2]
        if self.sdf_prior_mode == 'mean':
            return self._standardize(evidence).mean(dim=-1)
        if self.sdf_prior_mode == 'max':
            return self._standardize(center_distance)
        if self.sdf_prior_mode == 'topmean':
            return self._standardize(top_distance)
        if self.sdf_prior_mode == 'entropy':
            return self._standardize(entropy)
        if self.sdf_prior_mode == 'max_entropy':
            return self._standardize(center_distance) + 0.5 * self._standardize(entropy)
        if self.sdf_prior_mode == 'top_entropy':
            return self._standardize(top_distance) + 0.5 * self._standardize(entropy)
        if self.sdf_prior_mode == 'max_top':
            return 0.5 * (self._standardize(center_distance) + self._standardize(top_distance))
        if self.sdf_prior_mode == 'gap':
            return self._standardize(center_distance + torch.relu(center_distance - top_distance))
        if self.sdf_prior_mode == 'energy':
            feat = F.normalize(self.last_disc_features, p=2, dim=-1, eps=1e-6)
            centers = F.normalize(self.normal_centers, p=2, dim=-1, eps=1e-6)
            sim = torch.matmul(feat, centers.transpose(0, 1))
            tau = 0.2
            energy = -tau * torch.logsumexp(sim / tau, dim=-1)
            return self._standardize(energy)
        if self.sdf_prior_mode == 'knn3':
            feat = F.normalize(self.last_disc_features, p=2, dim=-1, eps=1e-6)
            centers = F.normalize(self.normal_centers, p=2, dim=-1, eps=1e-6)
            sim = torch.matmul(feat, centers.transpose(0, 1))
            top = torch.topk(sim, min(3, sim.size(-1)), dim=-1)[0]
            return self._standardize(torch.sum(1.0 - top, dim=-1))
        raise ValueError(f'unknown sdf_prior_mode: {self.sdf_prior_mode}')

    def _sdf_logits(self, features):
        evidence = self._sdf_evidence(features)
        disc_logits = self.disc_head(self.last_disc_features.transpose(1, 2)).squeeze(1)
        center_prior = self._sdf_prior(evidence)
        if self.sdf_logit_norm == 'both':
            disc_logits = self._standardize(disc_logits)
            center_prior = self._standardize(center_prior)
        elif self.sdf_logit_norm == 'disc':
            disc_logits = self._standardize(disc_logits)
        elif self.sdf_logit_norm == 'prior':
            center_prior = self._standardize(center_prior)
        if self.sdf_logit_norm == 'gate':
            gate = torch.sigmoid(self.sdf_gate)
            return (1.0 - gate) * disc_logits + gate * center_prior
        return disc_logits + self.sdf_prior_weight * self.context_scale.abs() * center_prior

    def _context_logits(self, features, visual_logits, text_logits):
        base_scores = torch.sigmoid((visual_logits + 0.5 * text_logits).detach())
        if self.context_mode == 'lgcc':
            evidence = self._lgcc_evidence(features, base_scores)
        elif self.context_mode == 'tcde':
            evidence = self._tcde_evidence(features)
        elif self.context_mode == 'dcac':
            evidence = self._dcac_evidence(features, base_scores)
        elif self.context_mode == 'cpr':
            evidence = self._cpr_evidence(features, base_scores)
        elif self.context_mode == 'sdf':
            return self._sdf_logits(features)
        else:
            raise ValueError(f'unknown context_mode: {self.context_mode}')
        evidence = self._standardize(evidence)
        learned_logits = self.context_heads[self.context_mode](evidence.transpose(1, 2)).squeeze(1)
        prior_logits = evidence.mean(dim=-1)
        return learned_logits + 0.2 * self.context_scale.abs() * prior_logits

    def forward(self, inputs, test=False):
        bs, ncrops, t, f = inputs.size()
        flat_inputs = inputs.reshape(-1, t, f)
        flat_features = self.encoder(flat_inputs)

        visual = self.relu(self.drop_out(self.fc1(flat_features)))
        visual = self.relu(self.drop_out(self.fc2(visual)))
        visual_logits = self.fc3(visual).view(bs, ncrops, t).mean(dim=1)

        semantic_features = flat_inputs if self.text_on_raw_clip else flat_features
        text_logits = self._semantic_logits(semantic_features, bs, ncrops, t)

        features = flat_features.view(bs, ncrops, t, f).mean(dim=1)
        context_logits = self._context_logits(features, visual_logits, text_logits)
        active = self._active_components()
        component_logits = {
            'mlp': visual_logits,
            'prompt': text_logits,
            'context': context_logits,
        }
        if self.score_fusion_mode == 'zscore':
            component_logits = {name: self._standardize(value) for name, value in component_logits.items()}
        if self.score_fusion_mode in ('context_only', 'logit_max', 'prob_avg', 'prob_max', 'context_boost', 'context_agree'):
            mlp_prob = self.sigmoid(component_logits['mlp'])
            context_prob = self.sigmoid(component_logits['context'])
            if self.score_fusion_mode == 'context_only':
                scores = context_prob
                logits = component_logits['context']
            elif self.score_fusion_mode == 'logit_max':
                logits = torch.maximum(component_logits['mlp'], component_logits['context'])
                scores = self.sigmoid(logits)
            elif self.score_fusion_mode == 'context_boost':
                boost = torch.relu(mlp_prob - context_prob)
                scores = torch.clamp(context_prob + 0.25 * boost, 1e-6, 1.0 - 1e-6)
                logits = torch.logit(scores)
            elif self.score_fusion_mode == 'context_agree':
                agree = 0.5 + 0.5 * mlp_prob
                scores = torch.clamp(context_prob * agree, 1e-6, 1.0 - 1e-6)
                logits = torch.logit(scores)
            elif self.score_fusion_mode in ('prob_avg', 'prob_max'):
                probs = []
                weights = []
                if 'mlp' in active:
                    probs.append(self.sigmoid(component_logits['mlp']))
                    weights.append(self.score_w[0])
                if 'prompt' in active:
                    probs.append(self.sigmoid(component_logits['prompt']))
                    weights.append(self.score_w[1])
                if 'context' in active:
                    probs.append(self.sigmoid(component_logits['context']))
                    weights.append(self.score_w[2])
                prob_stack = torch.stack(probs, dim=0)
                if self.score_fusion_mode == 'prob_max':
                    scores = prob_stack.max(dim=0)[0]
                else:
                    weight = torch.softmax(torch.stack(weights), dim=0).view(-1, 1, 1)
                    scores = torch.sum(weight * prob_stack, dim=0)
                logits = torch.logit(scores.clamp(1e-6, 1.0 - 1e-6))
        else:
            logits = features.new_zeros(bs, t)
            if 'mlp' in active:
                logits = logits + self.score_w[0] * component_logits['mlp']
            if 'prompt' in active:
                logits = logits + self.score_w[1] * component_logits['prompt']
            if 'context' in active:
                logits = logits + self.score_w[2] * component_logits['context']
            scores = self.sigmoid(logits)

        self.last_logits = logits
        self.last_visual_logits = visual_logits
        self.last_text_logits = text_logits
        self.last_context_logits = context_logits
        self.last_features = features
        self.last_magnitude = torch.norm(features, dim=-1)
        all_component_scores = {
            'mlp': self.sigmoid(visual_logits),
            'prompt': self.sigmoid(text_logits),
            'context': self.sigmoid(context_logits),
        }
        self.last_component_scores = {name: all_component_scores[name] for name in active}
        if test and self.test_score_mode != 'same':
            if self.test_score_mode == 'context':
                scores = all_component_scores['context']
                logits = context_logits
            elif self.test_score_mode == 'mlp':
                scores = all_component_scores['mlp']
                logits = visual_logits
            elif self.test_score_mode == 'prob_max':
                scores = torch.maximum(all_component_scores['mlp'], all_component_scores['context'])
                logits = torch.logit(scores.clamp(1e-6, 1.0 - 1e-6))
            else:
                raise ValueError(f'unknown test_score_mode: {self.test_score_mode}')
            self.last_logits = logits

        if test:
            score_abnormal, score_normal = 0, 0
            score_select_abnormal, score_select_normal = 0, 0
            normal_similarity, abnormal_similarity = 0, 0
        else:
            normal_scores = scores[: self.batch_size]
            abnormal_scores = scores[self.batch_size:]
            idx_sample_abn = torch.topk(abnormal_scores, min(self.k_abn, t), dim=1)[1]
            score_select_abnormal = torch.gather(abnormal_scores, 1, idx_sample_abn)
            score_abnormal = torch.mean(score_select_abnormal, dim=1)

            idx_sample_nor = torch.topk(normal_scores, min(self.k_nor, t), dim=1)[1]
            score_select_normal = torch.gather(normal_scores, 1, idx_sample_nor)
            score_normal = torch.mean(score_select_normal, dim=1)

            abnormal_similarity = torch.gather(context_logits[self.batch_size:], 1, idx_sample_abn).mean(dim=1)
            normal_similarity = torch.gather(context_logits[: self.batch_size], 1, idx_sample_nor).mean(dim=1)

        return (
            score_abnormal,
            score_normal,
            scores,
            score_select_abnormal,
            score_select_normal,
            normal_similarity,
            abnormal_similarity,
        )
