import torch
import torch.nn.functional as F


class MILLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel):
        label = torch.cat((nlabel, alabel), 0).to(score_normal.device)
        score = torch.cat((score_normal, score_abnormal), 0).squeeze()
        return self.criterion(score, label)


def l2_sparse(scores, lamda=1e-2):
    return lamda * torch.norm(scores.reshape(-1))


def semantic_alignment_loss(text_logits, batch_size, top_k=5, lamda=0.1):
    normal_logits = text_logits[:batch_size]
    abnormal_logits = text_logits[batch_size:]
    normal_targets = torch.zeros_like(normal_logits)
    normal_loss = F.binary_cross_entropy_with_logits(normal_logits, normal_targets)

    k = min(top_k, abnormal_logits.size(1))
    abnormal_top = torch.topk(abnormal_logits, k, dim=1)[0]
    abnormal_targets = torch.ones_like(abnormal_top)
    abnormal_loss = F.binary_cross_entropy_with_logits(abnormal_top, abnormal_targets)
    return lamda * (normal_loss + abnormal_loss)


def magnitude_contrastive_loss(magnitude, scores, batch_size, top_k=5, margin=1.0, lamda=0.05):
    normal_mag = magnitude[:batch_size]
    abnormal_mag = magnitude[batch_size:]
    normal_scores = scores[:batch_size]
    abnormal_scores = scores[batch_size:]
    k = min(top_k, normal_mag.size(1), abnormal_mag.size(1))

    normal_idx = torch.topk(normal_scores.detach(), k, dim=1)[1]
    abnormal_idx = torch.topk(abnormal_scores.detach(), k, dim=1)[1]
    normal_top_mag = torch.gather(normal_mag, 1, normal_idx).mean(dim=1)
    abnormal_top_mag = torch.gather(abnormal_mag, 1, abnormal_idx).mean(dim=1)
    return lamda * F.relu(margin - abnormal_top_mag + normal_top_mag).mean()


def prompt_diversity_loss(model, lamda=1e-3):
    losses = []
    for prompts in (model.normal_prompts, model.abnormal_prompts):
        if prompts.size(0) <= 1:
            continue
        p = F.normalize(prompts, p=2, dim=-1, eps=1e-6)
        sim = torch.matmul(p, p.transpose(0, 1))
        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        losses.append(((sim - eye) ** 2).mean())
    if not losses:
        return model.score_w.sum() * 0
    return lamda * sum(losses)


def train_semantic_mag(
    nloader,
    aloader,
    model,
    batch_size,
    optimizer,
    device,
    sparse_lam=1e-2,
    txt_lam=0.1,
    mag_lam=0.05,
    div_lam=1e-3,
    mag_margin=1.0,
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

        loss_mil = MILLoss()(score_normal, score_abnormal, nlabel, alabel)
        loss_txt = semantic_alignment_loss(model.last_text_logits, batch_size, top_k=top_k, lamda=txt_lam)
        loss_mag = magnitude_contrastive_loss(
            model.last_magnitude,
            scores,
            batch_size,
            top_k=top_k,
            margin=mag_margin,
            lamda=mag_lam,
        )
        loss_sparse = l2_sparse(scores, sparse_lam)
        loss_div = prompt_diversity_loss(model, div_lam)
        cost = loss_mil + loss_txt + loss_mag + loss_sparse + loss_div

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        return cost
