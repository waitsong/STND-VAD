import torch
import torch.nn.functional as F

from context_score_try import ContextScoreTryModel


class ContextScore10CropTryModel(ContextScoreTryModel):
    """10-crop variant: score each crop first, then fuse crop scores.

    The base model receives [B, N, T, D]. The old context branch averaged N crops
    into one feature sequence before SDF/context scoring. This variant keeps the
    crop dimension for MLP, prompt, and context/SDF logits, then fuses their
    probabilities into a video-level [B, T] score.
    """

    def __init__(self, *args, crop_fusion='mean', **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_fusion = crop_fusion

    def _standardize_sequence(self, values):
        return (values - values.mean(dim=-1, keepdim=True)) / (
            values.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
        )

    def _fuse_crops(self, values):
        if values.dim() < 3:
            return values
        if self.crop_fusion == 'mean':
            return values.mean(dim=1)
        if self.crop_fusion == 'max':
            return values.max(dim=1)[0]
        if self.crop_fusion == 'meanmax':
            return 0.5 * (values.mean(dim=1) + values.max(dim=1)[0])
        raise ValueError(f'unknown crop_fusion: {self.crop_fusion}')

    def _prob_to_logit(self, scores):
        return torch.logit(scores.clamp(1e-6, 1.0 - 1e-6))

    def _semantic_logits_crop(self, semantic_features, bs, ncrops, t):
        feat = F.normalize(self.text_adapter(semantic_features), p=2, dim=-1, eps=1e-6)
        normal_prompts = F.normalize(self.normal_prompts, p=2, dim=-1, eps=1e-6)
        abnormal_prompts = F.normalize(self.abnormal_prompts, p=2, dim=-1, eps=1e-6)
        normal_sim = torch.matmul(feat, normal_prompts.transpose(0, 1)).max(dim=-1)[0]
        abnormal_sim = torch.matmul(feat, abnormal_prompts.transpose(0, 1)).max(dim=-1)[0]
        text_logits = self.semantic_temperature.abs() * (abnormal_sim - normal_sim)
        return text_logits.view(bs, ncrops, t)

    def forward(self, inputs, test=False):
        bs, ncrops, t, f = inputs.size()
        flat_inputs = inputs.reshape(-1, t, f)
        flat_features = self.encoder(flat_inputs)

        visual = self.relu(self.drop_out(self.fc1(flat_features)))
        visual = self.relu(self.drop_out(self.fc2(visual)))
        visual_logits_crop = self.fc3(visual).view(bs, ncrops, t)

        semantic_features = flat_inputs if self.text_on_raw_clip else flat_features
        text_logits_crop = self._semantic_logits_crop(semantic_features, bs, ncrops, t)

        features_crop = flat_features.view(bs, ncrops, t, f)
        context_logits_flat = self._context_logits(
            features_crop.reshape(-1, t, f),
            visual_logits_crop.reshape(-1, t),
            text_logits_crop.reshape(-1, t),
        )
        context_logits_crop = context_logits_flat.view(bs, ncrops, t)

        if self.last_center_distance is not None and self.last_center_distance.size(0) == bs * ncrops:
            self.last_center_distance = self.last_center_distance.view(bs, ncrops, t).mean(dim=1)
        if self.last_sdf_evidence is not None and self.last_sdf_evidence.size(0) == bs * ncrops:
            self.last_sdf_evidence = self.last_sdf_evidence.view(bs, ncrops, t, -1).mean(dim=1)
        if self.last_disc_features is not None and self.last_disc_features.size(0) == bs * ncrops:
            self.last_disc_features = self.last_disc_features.view(bs, ncrops, t, -1).mean(dim=1)

        active = self._active_components()
        component_logits_crop = {
            'mlp': visual_logits_crop,
            'prompt': text_logits_crop,
            'context': context_logits_crop,
        }
        if self.score_fusion_mode == 'zscore':
            component_logits_crop = {
                name: self._standardize_sequence(value)
                for name, value in component_logits_crop.items()
            }

        if self.score_fusion_mode in (
            'context_only',
            'logit_max',
            'prob_avg',
            'prob_max',
            'context_boost',
            'context_agree',
        ):
            mlp_prob = self.sigmoid(component_logits_crop['mlp'])
            context_prob = self.sigmoid(component_logits_crop['context'])
            if self.score_fusion_mode == 'context_only':
                crop_scores = context_prob
            elif self.score_fusion_mode == 'logit_max':
                crop_scores = self.sigmoid(
                    torch.maximum(component_logits_crop['mlp'], component_logits_crop['context'])
                )
            elif self.score_fusion_mode == 'context_boost':
                crop_scores = torch.clamp(
                    context_prob + 0.25 * torch.relu(mlp_prob - context_prob),
                    1e-6,
                    1.0 - 1e-6,
                )
            elif self.score_fusion_mode == 'context_agree':
                crop_scores = torch.clamp(context_prob * (0.5 + 0.5 * mlp_prob), 1e-6, 1.0 - 1e-6)
            else:
                probs = []
                weights = []
                if 'mlp' in active:
                    probs.append(self.sigmoid(component_logits_crop['mlp']))
                    weights.append(self.score_w[0])
                if 'prompt' in active:
                    probs.append(self.sigmoid(component_logits_crop['prompt']))
                    weights.append(self.score_w[1])
                if 'context' in active:
                    probs.append(self.sigmoid(component_logits_crop['context']))
                    weights.append(self.score_w[2])
                prob_stack = torch.stack(probs, dim=0)
                if self.score_fusion_mode == 'prob_max':
                    crop_scores = prob_stack.max(dim=0)[0]
                else:
                    weight = torch.softmax(torch.stack(weights), dim=0).view(-1, 1, 1, 1)
                    crop_scores = torch.sum(weight * prob_stack, dim=0)
            scores = self._fuse_crops(crop_scores)
            logits = self._prob_to_logit(scores)
        else:
            logits_crop = features_crop.new_zeros(bs, ncrops, t)
            if 'mlp' in active:
                logits_crop = logits_crop + self.score_w[0] * component_logits_crop['mlp']
            if 'prompt' in active:
                logits_crop = logits_crop + self.score_w[1] * component_logits_crop['prompt']
            if 'context' in active:
                logits_crop = logits_crop + self.score_w[2] * component_logits_crop['context']
            scores = self._fuse_crops(self.sigmoid(logits_crop))
            logits = self._prob_to_logit(scores)

        all_component_scores_crop = {
            'mlp': self.sigmoid(visual_logits_crop),
            'prompt': self.sigmoid(text_logits_crop),
            'context': self.sigmoid(context_logits_crop),
        }
        all_component_scores = {
            name: self._fuse_crops(value)
            for name, value in all_component_scores_crop.items()
        }

        self.last_logits = logits
        self.last_visual_logits = self._prob_to_logit(all_component_scores['mlp'])
        self.last_text_logits = self._prob_to_logit(all_component_scores['prompt'])
        self.last_context_logits = self._prob_to_logit(all_component_scores['context'])
        features = features_crop.mean(dim=1)
        self.last_features = features
        self.last_magnitude = torch.norm(features, dim=-1)
        self.last_component_scores = {name: all_component_scores[name] for name in active}

        if test and self.test_score_mode != 'same':
            if self.test_score_mode == 'context':
                scores = all_component_scores['context']
                logits = self.last_context_logits
            elif self.test_score_mode == 'mlp':
                scores = all_component_scores['mlp']
                logits = self.last_visual_logits
            elif self.test_score_mode == 'prob_max':
                scores = torch.maximum(all_component_scores['mlp'], all_component_scores['context'])
                logits = self._prob_to_logit(scores)
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
