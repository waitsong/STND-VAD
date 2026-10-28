import torch
from torch import nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    def forward(self, x):
        out = torch.zeros_like(x)
        fold = x.size(-1) // 4
        if fold == 0:
            return x
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out


class HDDGTMixer(nn.Module):
    def __init__(self, dim, dilations=(1, 2, 4, 8, 16), dropout=0.1, use_shift=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dilations = tuple(dilations)
        self.use_shift = use_shift
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=dim,
                )
                for dilation in self.dilations
            ]
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, len(self.dilations)),
        )
        self.shift = TemporalShift()
        self.shift_proj = nn.Linear(dim, dim) if use_shift else None
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.last_scale_features = None
        self.last_scale_gate = None

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        xt = x_norm.transpose(1, 2)
        branch_outs = [branch(xt).transpose(1, 2) for branch in self.branches]

        groups = torch.stack(branch_outs, dim=2)

        gate = torch.softmax(self.scale_gate(x_norm), dim=-1).unsqueeze(-1)
        self.last_scale_features = groups
        self.last_scale_gate = gate.squeeze(-1)
        mixed = torch.sum(groups * gate, dim=2)
        if self.use_shift:
            mixed = mixed + self.shift_proj(self.shift(x_norm))

        update = self.out_proj(mixed)
        x = self.out_norm(residual + residual * update)
        x = self.out_norm(x + self.ffn(x))
        return x


class HDDGTMTryModel(nn.Module):
    def __init__(self, num_blocks, batch_size, n_features, version='v1', dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.num_blocks = num_blocks
        self.mlp_dim = n_features
        self.batch_size = batch_size
        self.version = version
        self.k_abn = 5
        self.k_nor = self.k_abn
        self.mlp_blocks = HDDGTMixer(
            self.mlp_dim,
            dilations=dilations,
            dropout=0.1,
            use_shift=(version == 'v3'),
        )
        self.fc1 = nn.Linear(self.mlp_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        if version in ('v2', 'v3'):
            self.w = nn.Parameter(torch.tensor([1.0, 1.0, 0.2]))
        elif version == 'mscs':
            self.w = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
            self.mscs_bias = nn.Parameter(torch.tensor(-1.0))
        else:
            self.w = nn.Parameter(torch.ones(2))
        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.last_logits = None
        self.last_features = None
        self.last_mscs_conflict = None
        self.last_mscs_uncertainty = None
        self.last_mscs_persistence = None
        self.last_msifs_scales = None

    def _ifs_terms(self, features, bs, ncrops, t, test):
        feat = F.normalize(features, p=2, dim=-1, eps=1e-6)
        if test:
            feat_similarity = torch.matmul(feat, feat.transpose(1, 2))
            feat_sim = -torch.mean(feat_similarity, dim=2).unsqueeze(2)
            feat_var = torch.var(feat_similarity, dim=2, unbiased=False).unsqueeze(2)
            feat_var = torch.mean(feat_var.view(-1, ncrops, t), dim=1)
            feat_sim = torch.mean(feat_sim.view(-1, ncrops, t), dim=1)
            return feat_sim, feat_var

        normal_feat = feat[: self.batch_size * ncrops]
        abnormal_feat = feat[self.batch_size * ncrops:]

        afeat_similarity = torch.matmul(abnormal_feat, normal_feat.transpose(1, 2))
        afeat_sim = -torch.mean(afeat_similarity, dim=2).unsqueeze(2)
        afeat_var = torch.var(afeat_similarity, dim=2, unbiased=False).unsqueeze(2)
        afeat_var = torch.mean(afeat_var.view(-1, ncrops, t), dim=1)
        afeat_sim = torch.mean(afeat_sim.view(-1, ncrops, t), dim=1)

        nfeat_similarity = torch.matmul(normal_feat, normal_feat.transpose(1, 2))
        nfeat_sim = -torch.mean(nfeat_similarity, dim=2).unsqueeze(2)
        nfeat_var = torch.var(nfeat_similarity, dim=2, unbiased=False).unsqueeze(2)
        nfeat_var = torch.mean(nfeat_var.view(-1, ncrops, t), dim=1)
        nfeat_sim = torch.mean(nfeat_sim.view(-1, ncrops, t), dim=1)

        feat_sim = torch.cat((nfeat_sim, afeat_sim), dim=0)
        feat_var = torch.cat((nfeat_var, afeat_var), dim=0)
        return feat_sim, feat_var

    def _magnitude_term(self, features, bs, ncrops, t):
        magnitude = torch.norm(features, dim=-1).view(bs, ncrops, t).mean(dim=1)
        return (magnitude - magnitude.mean(dim=1, keepdim=True)) / (
            magnitude.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )

    def _mscs_logits(self, bs, ncrops, t):
        scale_features = self.mlp_blocks.last_scale_features
        scale_gate = self.mlp_blocks.last_scale_gate
        scale_features = F.normalize(scale_features, p=2, dim=-1, eps=1e-6)

        pair_conflicts = []
        for i in range(scale_features.size(2)):
            for j in range(i + 1, scale_features.size(2)):
                pair_sim = torch.sum(scale_features[:, :, i] * scale_features[:, :, j], dim=-1)
                pair_conflicts.append(1.0 - pair_sim)
        conflict = torch.stack(pair_conflicts, dim=-1).mean(dim=-1)

        norm_const = torch.log(scale_gate.new_tensor(float(scale_gate.size(-1))))
        uncertainty = -torch.sum(scale_gate * torch.log(scale_gate + 1e-6), dim=-1) / norm_const
        evidence = conflict + uncertainty
        persistence = F.avg_pool1d(evidence.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)

        conflict = conflict.view(bs, ncrops, t).mean(dim=1)
        uncertainty = uncertainty.view(bs, ncrops, t).mean(dim=1)
        persistence = persistence.view(bs, ncrops, t).mean(dim=1)
        weights = F.softplus(self.w)
        logits = self.mscs_bias + weights[0] * conflict + weights[1] * uncertainty + weights[2] * persistence

        self.last_mscs_conflict = conflict
        self.last_mscs_uncertainty = uncertainty
        self.last_mscs_persistence = persistence
        return logits

    def _single_scale_ifs_flat(self, scale_features, ncrops, test):
        feat = F.normalize(scale_features, p=2, dim=-1, eps=1e-6)
        if test:
            similarity = torch.matmul(feat, feat.transpose(1, 2))
            return -torch.mean(similarity, dim=2) + torch.var(similarity, dim=2, unbiased=False)

        normal_feat = feat[: self.batch_size * ncrops]
        abnormal_feat = feat[self.batch_size * ncrops:]

        abnormal_similarity = torch.matmul(abnormal_feat, normal_feat.transpose(1, 2))
        abnormal_ifs = -torch.mean(abnormal_similarity, dim=2) + torch.var(
            abnormal_similarity,
            dim=2,
            unbiased=False,
        )

        normal_similarity = torch.matmul(normal_feat, normal_feat.transpose(1, 2))
        normal_ifs = -torch.mean(normal_similarity, dim=2) + torch.var(
            normal_similarity,
            dim=2,
            unbiased=False,
        )
        return torch.cat((normal_ifs, abnormal_ifs), dim=0)

    def _multi_scale_ifs_terms(self, bs, ncrops, t, test):
        scale_features = self.mlp_blocks.last_scale_features
        scale_gate = self.mlp_blocks.last_scale_gate
        relation_by_scale = torch.stack(
            [
                self._single_scale_ifs_flat(scale_features[:, :, scale_idx], ncrops, test)
                for scale_idx in range(scale_features.size(2))
            ],
            dim=-1,
        )
        relation = torch.sum(relation_by_scale * scale_gate, dim=-1)
        self.last_msifs_scales = relation_by_scale.view(bs, ncrops, t, scale_features.size(2)).mean(dim=1)
        return relation.view(bs, ncrops, t).mean(dim=1)

    def forward(self, inputs, test=False):
        bs, ncrops, t, f = inputs.size()
        inputs = inputs.reshape(-1, t, f)
        features = self.mlp_blocks(inputs)

        if self.version == 'mscs':
            feat_sim = features.new_zeros(bs, t)
            logits = self._mscs_logits(bs, ncrops, t)
        elif self.version == 'msifs':
            feat_sim = self._multi_scale_ifs_terms(bs, ncrops, t, test)
            scores = self.relu(self.drop_out(self.fc1(features)))
            scores = self.relu(self.drop_out(self.fc2(scores)))
            cls_logits = self.fc3(scores).view(bs, ncrops, t).mean(dim=1)
            logits = self.w[0] * cls_logits + self.w[1] * feat_sim
        else:
            feat_sim, feat_var = self._ifs_terms(features, bs, ncrops, t, test)
            scores = self.relu(self.drop_out(self.fc1(features)))
            scores = self.relu(self.drop_out(self.fc2(scores)))
            cls_logits = self.fc3(scores).view(bs, ncrops, t).mean(dim=1)

            relation_logits = feat_sim + feat_var
            if self.version in ('v2', 'v3'):
                magnitude = self._magnitude_term(features, bs, ncrops, t)
                logits = self.w[0] * cls_logits + self.w[1] * relation_logits + self.w[2] * magnitude
            else:
                logits = self.w[0] * cls_logits + self.w[1] * relation_logits
        scores = self.sigmoid(logits)
        self.last_logits = logits
        self.last_features = torch.mean(features.view(bs, ncrops, t, f), dim=1)

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

            abnormal_similarity = torch.gather(feat_sim[self.batch_size:], 1, idx_sample_abn).mean(dim=1)
            normal_similarity = torch.gather(feat_sim[: self.batch_size], 1, idx_sample_nor).mean(dim=1)

        return (
            score_abnormal,
            score_normal,
            scores,
            score_select_abnormal,
            score_select_normal,
            normal_similarity,
            abnormal_similarity,
        )
