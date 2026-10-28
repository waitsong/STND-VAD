import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _load_gt(args):
    if args.dataset == 'shanghai':
        return np.load('list/gt-sh.npy')
    if args.dataset == 'ucf':
        if getattr(args, 'use_ucf_10crop_gt', False):
            return np.load(getattr(args, 'ucf_10crop_gt', 'list_vadclip_10crop/gt-ucf-clip-ifs-vad-10crop.npy'))
        return np.load(f'list/gt-ucf-{args.feature_extractor}.npy')
    return np.load(f'list/gt-xd-{args.feature_extractor}.npy')


def _match_lengths(pred, gt):
    pred = np.repeat(np.array(pred), 16)
    if len(pred) != len(gt):
        eval_len = min(len(pred), len(gt))
        print(f'Warning: pred/gt length mismatch ({len(pred)} vs {len(gt)}), using first {eval_len} samples.')
        pred = pred[:eval_len]
        gt = gt[:eval_len]
    return pred, gt


def _metric(args, gt, pred):
    if args.dataset in ('shanghai', 'ucf'):
        fpr, tpr, _ = roc_curve(list(gt), pred)
        return auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(list(gt), pred)
    return auc(recall, precision)


def _metric_name(args):
    if args.dataset in ('shanghai', 'ucf'):
        return 'AUC'
    return 'AP'


def _smooth_scores(scores, args):
    kernel = getattr(args, 'test_smooth_kernel', 1)
    mode = getattr(args, 'test_smooth_mode', 'none')
    if kernel <= 1 or mode == 'none' or scores.size(-1) < 2:
        return scores
    kernel = min(kernel, scores.size(-1))
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    if kernel <= 1:
        return scores
    pad = kernel // 2
    x = scores.unsqueeze(1)
    if mode == 'avg':
        x = F.pad(x, (pad, pad), mode='replicate')
        return F.avg_pool1d(x, kernel_size=kernel, stride=1).squeeze(1)
    if mode == 'max':
        x = F.pad(x, (pad, pad), mode='replicate')
        return F.max_pool1d(x, kernel_size=kernel, stride=1).squeeze(1)
    if mode == 'avgmax':
        x = F.pad(x, (pad, pad), mode='replicate')
        avg = F.avg_pool1d(x, kernel_size=kernel, stride=1)
        avg = F.pad(avg, (pad, pad), mode='replicate')
        return F.max_pool1d(avg, kernel_size=kernel, stride=1).squeeze(1)
    if mode == 'maxavg':
        x = F.pad(x, (pad, pad), mode='replicate')
        pooled = F.max_pool1d(x, kernel_size=kernel, stride=1)
        pooled = F.pad(pooled, (pad, pad), mode='replicate')
        return F.avg_pool1d(pooled, kernel_size=kernel, stride=1).squeeze(1)
    raise ValueError(f'unknown test_smooth_mode: {mode}')


def _class_evidence_means(evidence, gt):
    names = ('nearest_distance', 'neighborhood_distance', 'assignment_uncertainty')
    result = {}
    gt = np.asarray(gt).reshape(-1)
    for index, name in enumerate(names):
        values, labels = _match_lengths(evidence[:, index], gt)
        normal = values[labels == 0]
        abnormal = values[labels == 1]
        result[name] = {
            'normal_mean': float(normal.mean()) if normal.size else None,
            'abnormal_mean': float(abnormal.mean()) if abnormal.size else None,
            'normal_count': int(normal.size),
            'abnormal_count': int(abnormal.size),
        }
    return result


def test_with_component_auc(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        component_preds = {}
        sdf_evidence = {'short': [], 'long': []}

        for inputs in dataloader:
            inputs = inputs.to(device)
            _, _, scores, _, _, _, _ = model(inputs, test=True)
            scores = _smooth_scores(scores, args)
            pred = torch.cat((pred, scores.reshape(-1)))
            for name, values in model.last_component_scores.items():
                if name not in component_preds:
                    component_preds[name] = torch.zeros(0, device=device)
                values = _smooth_scores(values, args)
                component_preds[name] = torch.cat((component_preds[name], values.reshape(-1)))
            if getattr(args, 'report_sdf_stats', False):
                for branch, attribute in (
                    ('short', 'last_short_sdf_evidence'),
                    ('long', 'last_long_sdf_evidence'),
                ):
                    evidence = getattr(model, attribute, None)
                    if evidence is not None:
                        sdf_evidence[branch].append(evidence.reshape(-1, 3).cpu().numpy())

        gt = _load_gt(args)
        metric_name = _metric_name(args)
        pred_np, gt = _match_lengths(pred.cpu().detach().numpy(), gt)
        main_auc = _metric(args, gt, pred_np)
        print(f'{metric_name} : ' + str(main_auc))

        component_auc = {}
        for name, values in component_preds.items():
            component_np, component_gt = _match_lengths(values.cpu().detach().numpy(), gt)
            component_auc[name] = _metric(args, component_gt, component_np)
            print(f'{metric_name}_{name} : {component_auc[name]}')
        model.last_sdf_stats = {}
        if getattr(args, 'report_sdf_stats', False):
            for branch, batches in sdf_evidence.items():
                if not batches:
                    continue
                stats = _class_evidence_means(np.concatenate(batches, axis=0), gt)
                model.last_sdf_stats[branch] = stats
                for name, values in stats.items():
                    print(
                        f'SDF_{branch}_{name}: normal_mean={values["normal_mean"]} '
                        f'abnormal_mean={values["abnormal_mean"]}'
                    )
        return main_auc, component_auc
