import torch
import torch.nn.functional as F

from train_semantic_mag_try import (
    MILLoss,
    l2_sparse,
    prompt_diversity_loss,
    semantic_alignment_loss,
)


def _branch_mil(scores, batch_size, nlabel, alabel, top_k=5):
    normal_scores = scores[:batch_size]
    abnormal_scores = scores[batch_size:]
    k_nor = min(top_k, normal_scores.size(1))
    k_abn = min(top_k, abnormal_scores.size(1))
    score_normal = torch.topk(normal_scores, k_nor, dim=1)[0].mean(dim=1)
    score_abnormal = torch.topk(abnormal_scores, k_abn, dim=1)[0].mean(dim=1)
    return MILLoss()(score_normal, score_abnormal, nlabel, alabel)


def _temporal_smooth_scores(scores, kernel=1, mode='none'):
    if mode == 'none' or kernel <= 1 or scores.size(1) < 2:
        return scores
    kernel = min(kernel, scores.size(1))
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    if kernel <= 1:
        return scores
    pad = kernel // 2
    x = F.pad(scores.unsqueeze(1), (pad, pad), mode='replicate')
    if mode == 'avg':
        return F.avg_pool1d(x, kernel_size=kernel, stride=1).squeeze(1)
    if mode == 'max':
        return F.max_pool1d(x, kernel_size=kernel, stride=1).squeeze(1)
    raise ValueError(f'unknown train_score_smooth_mode: {mode}')


def _video_topk_scores(scores, batch_size, top_k=5):
    normal_scores = scores[:batch_size]
    abnormal_scores = scores[batch_size:]
    k_nor = min(top_k, normal_scores.size(1))
    k_abn = min(top_k, abnormal_scores.size(1))
    score_normal = torch.topk(normal_scores, k_nor, dim=1)[0].mean(dim=1)
    score_abnormal = torch.topk(abnormal_scores, k_abn, dim=1)[0].mean(dim=1)
    return score_normal, score_abnormal


def _proposal_guided_video_scores(value_scores, proposal_scores, batch_size, top_k=5):
    normal_values = value_scores[:batch_size]
    abnormal_values = value_scores[batch_size:]
    normal_proposals = proposal_scores[:batch_size]
    abnormal_proposals = proposal_scores[batch_size:]
    k_nor = min(top_k, normal_values.size(1))
    k_abn = min(top_k, abnormal_values.size(1))
    normal_idx = torch.topk(normal_proposals.detach(), k_nor, dim=1)[1]
    abnormal_idx = torch.topk(abnormal_proposals.detach(), k_abn, dim=1)[1]
    score_normal = torch.gather(normal_values, 1, normal_idx).mean(dim=1)
    score_abnormal = torch.gather(abnormal_values, 1, abnormal_idx).mean(dim=1)
    return score_normal, score_abnormal


def context_gap_loss(context_logits, scores, batch_size, top_k=5, margin=0.2, lamda=0.1):
    normal_context = context_logits[:batch_size]
    abnormal_context = context_logits[batch_size:]
    normal_scores = scores[:batch_size]
    abnormal_scores = scores[batch_size:]
    k = min(top_k, normal_context.size(1), abnormal_context.size(1))
    normal_idx = torch.topk(normal_scores.detach(), k, dim=1)[1]
    abnormal_idx = torch.topk(abnormal_scores.detach(), k, dim=1)[1]
    normal_top = torch.gather(normal_context, 1, normal_idx).mean(dim=1)
    abnormal_top = torch.gather(abnormal_context, 1, abnormal_idx).mean(dim=1)
    return lamda * F.relu(margin - abnormal_top + normal_top).mean()


def normal_context_smooth_loss(features, batch_size, lamda=1e-4):
    if features is None or features.size(1) < 2:
        return torch.tensor(0.0, device=features.device if features is not None else 'cpu')
    normal_features = features[:batch_size]
    diff = normal_features[:, 1:] - normal_features[:, :-1]
    return lamda * torch.mean(diff ** 2)


def normal_context_score_smooth_loss(context_logits, batch_size, lamda=0.0):
    if lamda <= 0 or context_logits is None or context_logits.size(1) < 2:
        return context_logits.new_tensor(0.0) if context_logits is not None else torch.tensor(0.0)
    normal_scores = torch.sigmoid(context_logits[:batch_size])
    diff = normal_scores[:, 1:] - normal_scores[:, :-1]
    return lamda * torch.mean(diff ** 2)


def normal_center_balance_loss(model, batch_size, lamda=0.0, tau=0.2, branch='long'):
    if lamda <= 0:
        return next(model.parameters()).new_tensor(0.0)
    centers = getattr(model, 'short_normal_centers' if branch == 'short' else 'normal_centers', None)
    disc_features = getattr(model, f'last_{branch}_disc_features', None)
    if branch == 'long' and disc_features is None:
        disc_features = getattr(model, 'last_disc_features', None)
    if centers is None or disc_features is None:
        return next(model.parameters()).new_tensor(0.0)
    normal_feat = F.normalize(disc_features[:batch_size], p=2, dim=-1, eps=1e-6)
    centers = F.normalize(centers, p=2, dim=-1, eps=1e-6)
    sim = torch.matmul(normal_feat, centers.transpose(0, 1))
    assign = torch.softmax(sim / tau, dim=-1)
    usage = assign.mean(dim=(0, 1))
    uniform = usage.new_full(usage.shape, 1.0 / usage.numel())
    return lamda * torch.sum(usage * torch.log((usage + 1e-6) / uniform))


def normal_center_discriminative_loss(
    model,
    scores,
    batch_size,
    top_k=5,
    margin=0.5,
    lamda=0.1,
    div_lam=0.01,
    compact_lam=1.0,
    sep_lam=1.0,
    temporal_lam=1.0,
    branch='long',
):
    center_distance = getattr(model, f'last_{branch}_center_distance', None)
    centers = getattr(model, 'short_normal_centers' if branch == 'short' else 'normal_centers', None)
    disc_features = getattr(model, f'last_{branch}_disc_features', None)
    if branch == 'long':
        center_distance = center_distance if center_distance is not None else getattr(model, 'last_center_distance', None)
        disc_features = disc_features if disc_features is not None else getattr(model, 'last_disc_features', None)
    if center_distance is None or centers is None or disc_features is None:
        return scores.new_tensor(0.0)

    normal_dist = center_distance[:batch_size]
    abnormal_dist = center_distance[batch_size:]
    abnormal_scores = scores[batch_size:]

    compact = normal_dist.mean()
    k = min(top_k, abnormal_dist.size(1))
    abnormal_idx = torch.topk(abnormal_scores.detach(), k, dim=1)[1]
    abnormal_top_dist = torch.gather(abnormal_dist, 1, abnormal_idx).mean(dim=1)
    separate = F.relu(margin - abnormal_top_dist).mean()

    centers = F.normalize(centers, p=2, dim=-1, eps=1e-6)
    center_sim = torch.matmul(centers, centers.transpose(0, 1))
    eye = torch.eye(center_sim.size(0), device=center_sim.device, dtype=center_sim.dtype)
    diversity = ((center_sim - eye) ** 2).mean()

    normal_feat = F.normalize(disc_features[:batch_size], p=2, dim=-1, eps=1e-6)
    temporal_consistency = torch.mean((normal_feat[:, 1:] - normal_feat[:, :-1]) ** 2)
    return lamda * (
        compact_lam * compact
        + sep_lam * separate
        + temporal_lam * temporal_consistency
        + div_lam * diversity
    )


def train_context_score(
    nloader,
    aloader,
    model,
    batch_size,
    optimizer,
    device,
    sparse_lam=1e-2,
    txt_lam=0.1,
    div_lam=0.0,
    component_lam=0.1,
    context_gap_lam=0.1,
    context_margin=0.2,
    smooth_lam=1e-4,
    center_lam=0.1,
    center_margin=0.5,
    center_compact_lam=1.0,
    center_sep_lam=1.0,
    center_temporal_lam=1.0,
    center_div_lam=0.01,
    context_mil_lam=0.0,
    context_score_smooth_lam=0.0,
    center_balance_lam=0.0,
    center_balance_tau=0.2,
    train_score_smooth_kernel=1,
    train_score_smooth_mode='none',
    mil_value_mode='model',
    mil_proposal_mode='same',
    top_k=5,
):
    with torch.set_grad_enabled(True):
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)
        inputs = torch.cat((ninput, ainput), 0).to(device)

        score_abnormal, score_normal, scores, _, _, _, _ = model(inputs)
        nlabel = nlabel[:batch_size]
        alabel = alabel[:batch_size]

        context_scores = torch.sigmoid(model.last_context_logits)
        mlp_scores = torch.sigmoid(model.last_visual_logits)
        if mil_value_mode in ('context', 'long'):
            value_scores = context_scores
        elif mil_value_mode in ('mlp', 'short'):
            value_scores = mlp_scores
        elif mil_value_mode == 'model':
            value_scores = scores
        else:
            raise ValueError(f'unknown mil_value_mode: {mil_value_mode}')

        if mil_proposal_mode == 'same':
            proposal_scores = value_scores
        elif mil_proposal_mode in ('context', 'long'):
            proposal_scores = context_scores
        elif mil_proposal_mode in ('mlp', 'short'):
            proposal_scores = mlp_scores
        elif mil_proposal_mode == 'prob_max':
            proposal_scores = torch.maximum(mlp_scores, context_scores)
        elif mil_proposal_mode == 'model':
            proposal_scores = scores
        else:
            raise ValueError(f'unknown mil_proposal_mode: {mil_proposal_mode}')

        mil_scores = _temporal_smooth_scores(
            value_scores,
            kernel=train_score_smooth_kernel,
            mode=train_score_smooth_mode,
        )
        proposal_scores = _temporal_smooth_scores(
            proposal_scores,
            kernel=train_score_smooth_kernel,
            mode=train_score_smooth_mode,
        )
        if mil_value_mode != 'model' or mil_proposal_mode != 'same' or (
            train_score_smooth_mode != 'none' and train_score_smooth_kernel > 1
        ):
            if mil_proposal_mode == 'same':
                score_normal, score_abnormal = _video_topk_scores(mil_scores, batch_size, top_k=top_k)
            else:
                score_normal, score_abnormal = _proposal_guided_video_scores(
                    mil_scores,
                    proposal_scores,
                    batch_size,
                    top_k=top_k,
                )

        loss_mil = MILLoss()(score_normal, score_abnormal, nlabel, alabel)
        loss_sparse = l2_sparse(mil_scores, sparse_lam)
        if 'prompt' in getattr(model, '_active_components')():
            loss_txt = semantic_alignment_loss(model.last_text_logits, batch_size, top_k=top_k, lamda=txt_lam)
        else:
            loss_txt = scores.new_tensor(0.0)
        has_long_branch = getattr(model, 'use_long_branch', True)
        has_sdf_prototype = getattr(model, 'use_sdf_prototype', True)
        loss_context_mil = scores.new_tensor(0.0)
        if context_mil_lam > 0 and has_long_branch:
            context_scores = torch.sigmoid(model.last_context_logits)
            loss_context_mil = context_mil_lam * _branch_mil(context_scores, batch_size, nlabel, alabel, top_k=top_k)
        if has_long_branch:
            loss_gap = context_gap_loss(
                model.last_context_logits,
                mil_scores,
                batch_size,
                top_k=top_k,
                margin=context_margin,
                lamda=context_gap_lam,
            )
        else:
            loss_gap = scores.new_tensor(0.0)
        loss_smooth = normal_context_smooth_loss(model.last_features, batch_size, smooth_lam)
        if has_long_branch:
            loss_context_score_smooth = normal_context_score_smooth_loss(
                model.last_context_logits,
                batch_size,
                context_score_smooth_lam,
            )
        else:
            loss_context_score_smooth = scores.new_tensor(0.0)
        if has_sdf_prototype:
            center_losses = []
            balance_losses = []
            branches = ('short', 'long') if has_long_branch else ('short',)
            for branch in branches:
                branch_scores = model.last_component_scores.get(branch)
                if branch_scores is None:
                    continue
                center_losses.append(normal_center_discriminative_loss(
                    model,
                    branch_scores,
                    batch_size,
                    top_k=top_k,
                    margin=center_margin,
                    lamda=center_lam,
                    compact_lam=center_compact_lam,
                    sep_lam=center_sep_lam,
                    temporal_lam=center_temporal_lam,
                    div_lam=center_div_lam,
                    branch=branch,
                ))
                balance_losses.append(normal_center_balance_loss(
                    model,
                    batch_size,
                    center_balance_lam,
                    center_balance_tau,
                    branch=branch,
                ))
            loss_center = torch.stack(center_losses).mean() if center_losses else scores.new_tensor(0.0)
            loss_center_balance = torch.stack(balance_losses).mean() if balance_losses else scores.new_tensor(0.0)
        else:
            loss_center = scores.new_tensor(0.0)
            loss_center_balance = scores.new_tensor(0.0)
        loss_div = prompt_diversity_loss(model, div_lam)

        loss_components = scores.new_tensor(0.0)
        for component_scores in model.last_component_scores.values():
            loss_components = loss_components + _branch_mil(component_scores, batch_size, nlabel, alabel, top_k=top_k)
        loss_components = component_lam * loss_components / max(1, len(model.last_component_scores))

        cost = (
            loss_mil
            + loss_sparse
            + loss_txt
            + loss_context_mil
            + loss_gap
            + loss_smooth
            + loss_context_score_smooth
            + loss_center
            + loss_center_balance
            + loss_div
            + loss_components
        )
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        return cost
