import torch
from torch import nn
import torch.nn.functional as F

from context_score_simple_10crop_try import ContextScoreSimple10CropTryModel


class ContextScoreShortLongSDF10CropTryModel(ContextScoreSimple10CropTryModel):
    """Short/long scoring with separate SDF evidence.

    Short branch:
        clip-wise MLP(z_t) + SDF_short(z_t)

    Long branch:
        MLP on temporally enhanced flat_features (512-d, adjacent-time conv + residual)
        SDF on long_q built from short_q with a separate temporal conv

    The two branches use different projection layers and normal prototypes.
    Long branch pairs neighboring time steps (t, t+1) before temporal convolution.
    """

    def __init__(
        self,
        *args,
        short_k=None,
        long_k=None,
        short_sdf_weight=0.2,
        long_sdf_weight=0.2,
        short_long_fusion='learned',
        long_head='temporal',
        long_temporal_kernel=3,
        use_dilated_encoder=True,
        use_long_branch=True,
        use_sdf_prototype=True,
        **kwargs,
    ):
        if long_k is not None:
            kwargs['num_normal_centers'] = long_k
        super().__init__(*args, **kwargs)
        self.long_k = self.normal_centers.size(0)
        self.short_k = self.long_k if short_k is None else short_k
        if self.short_k < 1 or self.long_k < 1:
            raise ValueError('short_k and long_k must be positive integers')
        self.short_sdf_weight = short_sdf_weight
        self.long_sdf_weight = long_sdf_weight
        self.short_long_fusion = short_long_fusion
        self.use_dilated_encoder = use_dilated_encoder
        self.use_long_branch = use_long_branch
        self.use_sdf_prototype = use_sdf_prototype
        self.long_head = long_head
        if long_temporal_kernel % 2 == 0:
            long_temporal_kernel += 1
        self.long_temporal_kernel = long_temporal_kernel

        self.short_disc_proj = nn.Sequential(
            nn.LayerNorm(self.mlp_dim),
            nn.Linear(self.mlp_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.disc_dim),
        )
        self.short_normal_centers = nn.Parameter(torch.randn(self.short_k, self.disc_dim))
        self.long_temporal_conv = nn.Sequential(
            nn.Conv1d(
                self.disc_dim * 2,
                self.disc_dim,
                kernel_size=self.long_temporal_kernel,
                padding=self.long_temporal_kernel // 2,
                stride=1,
            ),
            nn.GELU(),
        )
        self.long_feature_temporal_conv = nn.Sequential(
            nn.Conv1d(
                self.mlp_dim * 2,
                self.mlp_dim,
                kernel_size=self.long_temporal_kernel,
                padding=self.long_temporal_kernel // 2,
                stride=1,
            ),
            nn.GELU(),
        )
        self.long_fc1 = nn.Linear(self.mlp_dim, 512)
        self.long_fc2 = nn.Linear(512, 128)
        self.long_fc3 = nn.Linear(128, 1)

        self.last_short_sdf_evidence = None
        self.last_long_sdf_evidence = None
        self.last_short_sdf_prior = None
        self.last_long_sdf_prior = None
        self.last_short_center_distance = None
        self.last_long_center_distance = None
        self.last_short_disc_features = None
        self.last_long_disc_features = None

    def _sdf_evidence_from(self, projected_features, centers):
        feat = F.normalize(projected_features, p=2, dim=-1, eps=1e-6)
        centers = F.normalize(centers, p=2, dim=-1, eps=1e-6)
        sim = torch.matmul(feat, centers.transpose(0, 1))
        top_sim = torch.topk(sim, min(self.sdf_topk, sim.size(-1)), dim=-1)[0]
        max_sim = top_sim[:, :, 0]
        mean_top_sim = top_sim.mean(dim=-1)
        soft_assign = torch.softmax(sim * 10.0, dim=-1)
        entropy = -torch.sum(soft_assign * torch.log(soft_assign + 1e-6), dim=-1)
        entropy = entropy / torch.log(sim.new_tensor(float(sim.size(-1))))
        return torch.stack((1.0 - max_sim, 1.0 - mean_top_sim, entropy), dim=-1)

    def _sdf_prior_from_evidence(self, evidence):
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
        raise ValueError(f'unsupported sdf_prior_mode for short-long sdf: {self.sdf_prior_mode}')

    def _semantic_logits_crop(self, semantic_features, bs, ncrops, t):
        return semantic_features.new_zeros(bs, ncrops, t)

    def _build_temporal_residual(self, features, temporal_conv):
        """Concat adjacent time steps (t, t+1), apply temporal conv, residual-add."""
        t = features.size(1)
        if t < 2:
            paired = torch.cat([features, features], dim=-1)
            temporal = temporal_conv(paired.transpose(1, 2)).transpose(1, 2)
            return features + temporal

        paired = torch.cat([features[:, :-1, :], features[:, 1:, :]], dim=-1)
        temporal = temporal_conv(paired.transpose(1, 2)).transpose(1, 2)
        temporal_full = torch.cat([temporal, temporal[:, -1:]], dim=1)
        return features + temporal_full

    def _encode_features(self, flat_inputs):
        if self.use_dilated_encoder:
            return self.encoder(flat_inputs)
        return self.encoder_bypass(flat_inputs)

    def _sdf_term(self, prior):
        if not self.use_sdf_prototype:
            return prior.new_zeros(prior.shape)
        return self.short_sdf_weight * self.context_scale.abs() * prior

    def _long_sdf_term(self, prior):
        if not self.use_sdf_prototype:
            return prior.new_zeros(prior.shape)
        return self.long_sdf_weight * self.context_scale.abs() * prior

    def _fuse_short_long(self, short_logits_crop, long_logits_crop, short_scores_crop, long_scores_crop):
        if not self.use_long_branch:
            crop_scores = short_scores_crop
            scores = self._fuse_crops(crop_scores)
            return scores, self._prob_to_logit(scores)

        if self.short_long_fusion == 'mean':
            crop_scores = 0.5 * (short_scores_crop + long_scores_crop)
            scores = self._fuse_crops(crop_scores)
            logits = self._prob_to_logit(scores)
        elif self.short_long_fusion == 'short':
            crop_scores = short_scores_crop
            scores = self._fuse_crops(crop_scores)
            logits = self._prob_to_logit(scores)
        elif self.short_long_fusion == 'long':
            crop_scores = long_scores_crop
            scores = self._fuse_crops(crop_scores)
            logits = self._prob_to_logit(scores)
        elif self.short_long_fusion == 'max':
            crop_scores = torch.maximum(short_scores_crop, long_scores_crop)
            scores = self._fuse_crops(crop_scores)
            logits = self._prob_to_logit(scores)
        elif self.short_long_fusion == 'learned':
            logits_crop = self.score_w[0] * short_logits_crop + self.score_w[2] * long_logits_crop
            crop_scores = self.sigmoid(logits_crop)
            scores = self._fuse_crops(crop_scores)
            logits = self._prob_to_logit(scores)
        else:
            raise ValueError(f'unknown short_long_fusion: {self.short_long_fusion}')
        return scores, logits

    def forward(self, inputs, test=False):
        bs, ncrops, t, f = inputs.size()
        flat_inputs = inputs.reshape(-1, t, f)
        flat_features = self._encode_features(flat_inputs)

        # Short branch: point-wise MLP on clip features, optionally boosted by short SDF.
        visual = self.relu(self.drop_out(self.fc1(flat_features)))
        visual = self.relu(self.drop_out(self.fc2(visual)))
        mlp_logits_crop = self.fc3(visual).view(bs, ncrops, t)

        short_q = self.short_disc_proj(flat_features)
        short_evidence = self._sdf_evidence_from(short_q, self.short_normal_centers)
        short_prior = self._sdf_prior_from_evidence(short_evidence).view(bs, ncrops, t)
        short_logits_crop = mlp_logits_crop + self._sdf_term(short_prior)
        short_scores_crop = self.sigmoid(short_logits_crop)
        short_scores = self._fuse_crops(short_scores_crop)

        long_q = None
        long_evidence = None
        long_prior = None
        long_scores = None
        long_scores_crop = None
        long_logits_crop = None
        if self.use_long_branch:
            # Long branch: adjacent-time residual features plus optional independent SDF.
            long_q = self._build_temporal_residual(short_q, self.long_temporal_conv)
            if self.long_head == 'temporal':
                long_disc_logits = self.disc_head(long_q.transpose(1, 2)).squeeze(1).view(bs, ncrops, t)
            elif self.long_head == 'mlp':
                long_f = self._build_temporal_residual(flat_features, self.long_feature_temporal_conv)
                long_visual = self.relu(self.drop_out(self.long_fc1(long_f)))
                long_visual = self.relu(self.drop_out(self.long_fc2(long_visual)))
                long_disc_logits = self.long_fc3(long_visual).view(bs, ncrops, t)
            else:
                raise ValueError(f'unknown long_head: {self.long_head}')
            long_evidence = self._sdf_evidence_from(long_q, self.normal_centers)
            long_prior = self._sdf_prior_from_evidence(long_evidence).view(bs, ncrops, t)
            long_logits_crop = long_disc_logits + self._long_sdf_term(long_prior)
            long_scores_crop = self.sigmoid(long_logits_crop)
            long_scores = self._fuse_crops(long_scores_crop)

        scores, logits = self._fuse_short_long(
            short_logits_crop,
            long_logits_crop,
            short_scores_crop,
            long_scores_crop,
        )

        self.last_logits = logits
        self.last_visual_logits = self._prob_to_logit(short_scores)
        self.last_context_logits = self._prob_to_logit(long_scores) if self.use_long_branch else self.last_visual_logits
        self.last_text_logits = inputs.new_zeros(bs, t)
        self.last_features = flat_features.view(bs, ncrops, t, f).mean(dim=1)
        self.last_magnitude = torch.norm(self.last_features, dim=-1)
        self.last_short_disc_features = short_q.view(bs, ncrops, t, -1).mean(dim=1)
        self.last_long_disc_features = (
            long_q.view(bs, ncrops, t, -1).mean(dim=1) if self.use_long_branch else None
        )
        self.last_disc_features = (
            self.last_long_disc_features
            if self.last_long_disc_features is not None
            else self.last_short_disc_features
        )
        if self.use_sdf_prototype:
            self.last_short_center_distance = self._fuse_crops(
                short_evidence[:, :, 0].view(bs, ncrops, t)
            )
            self.last_long_center_distance = (
                self._fuse_crops(long_evidence[:, :, 0].view(bs, ncrops, t))
                if self.use_long_branch else None
            )
            evidence_source = long_evidence if self.use_long_branch else short_evidence
            self.last_center_distance = evidence_source[:, :, 0].view(bs, ncrops, t).mean(dim=1)
            self.last_sdf_evidence = evidence_source.view(bs, ncrops, t, -1).mean(dim=1)
        else:
            self.last_short_center_distance = None
            self.last_long_center_distance = None
            self.last_center_distance = None
            self.last_sdf_evidence = None
        self.last_short_sdf_evidence = short_evidence.view(bs, ncrops, t, -1).mean(dim=1)
        self.last_long_sdf_evidence = (
            long_evidence.view(bs, ncrops, t, -1).mean(dim=1) if self.use_long_branch else None
        )
        self.last_short_sdf_prior = self._fuse_crops(short_prior)
        self.last_long_sdf_prior = self._fuse_crops(long_prior) if self.use_long_branch else None
        self.last_component_scores = {'short': short_scores}
        if self.use_long_branch:
            self.last_component_scores['long'] = long_scores

        if test and self.test_score_mode != 'same':
            if self.test_score_mode in ('context', 'long'):
                if not self.use_long_branch:
                    raise ValueError('test_score_mode=long/context requires use_long_branch=True')
                scores = long_scores
                self.last_logits = self.last_context_logits
            elif self.test_score_mode in ('mlp', 'short'):
                scores = short_scores
                self.last_logits = self.last_visual_logits
            elif self.test_score_mode == 'prob_max':
                if not self.use_long_branch:
                    scores = short_scores
                else:
                    scores = torch.maximum(short_scores, long_scores)
                self.last_logits = self._prob_to_logit(scores)
            else:
                raise ValueError(f'unknown test_score_mode: {self.test_score_mode}')

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

            abnormal_similarity = torch.gather(self.last_context_logits[self.batch_size:], 1, idx_sample_abn).mean(dim=1)
            normal_similarity = torch.gather(self.last_context_logits[: self.batch_size], 1, idx_sample_nor).mean(dim=1)

        return (
            score_abnormal,
            score_normal,
            scores,
            score_select_abnormal,
            score_select_normal,
            normal_similarity,
            abnormal_similarity,
        )
