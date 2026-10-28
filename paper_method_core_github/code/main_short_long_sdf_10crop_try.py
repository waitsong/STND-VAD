from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from dataset import Dataset
from train_context_score_try import train_context_score
from test_component_auc_10crop_try import test_with_component_auc
import option
from tqdm import tqdm
from config import *
from context_score_short_long_sdf_10crop_try import ContextScoreShortLongSDF10CropTryModel
import os
import random
import json
import numpy as np


def _add_arg(*args, **kwargs):
    if not any(arg in option.parser._option_string_actions for arg in args):
        option.parser.add_argument(*args, **kwargs)


def _parse_dilations(value):
    return tuple(int(item) for item in value.split(',') if item.strip())


def _parse_float_tuple(value):
    return tuple(float(item) for item in value.split(',') if item.strip())


def _set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _best_basename(args):
    base = f'best-{args.model_name}-context_{args.context_mode}-{args.dataset}-{args.feature_extractor}'
    run_suffix = getattr(args, 'run_suffix', '')
    if run_suffix:
        base = f'{base}-{run_suffix}'
    return base


def _save_best_artifacts(args, model, step, auc, component_auc):
    base = os.path.join('result', _best_basename(args))
    torch.save(model.state_dict(), f'{base}.pkl')
    with open(f'{base}-config.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'best_epoch': step,
                'best_auc': auc,
                'auc_short': component_auc.get('short'),
                'auc_long': component_auc.get('long'),
                'deterministic': args.deterministic,
                'seed': args.seed,
                'args': vars(args),
                'sdf_stats': getattr(model, 'last_sdf_stats', {}),
            },
            f,
            indent=2,
            default=str,
        )
    print(f'saved best checkpoint: {base}.pkl (auc={auc}, epoch={step})')
    print(f'saved best config: {base}-config.json')


def _metric_name(args):
    if args.dataset in ('shanghai', 'ucf'):
        return 'AUC'
    return 'AP'


if __name__ == '__main__':
    _add_arg('--seed', type=int, default=42)
    _add_arg('--context-mode', choices=['lgcc', 'tcde', 'dcac', 'cpr', 'sdf'], default='lgcc')
    _add_arg('--hd-dilations', default='1,2,4,8,16')
    _add_arg('--run-suffix', default='')
    _add_arg('--scale-gate-mode', choices=['learned', 'uniform', 'static'], default='static')
    _add_arg('--local-kernel', type=int, default=5)
    _add_arg('--anchor-k', type=int, default=8)
    _add_arg('--num-normal-centers', type=int, default=16)
    _add_arg('--num-normal-prompts', type=int, default=4)
    _add_arg('--num-abnormal-prompts', type=int, default=13)
    _add_arg('--sparse-lam', type=float, default=1e-2)
    _add_arg('--txt-lam', type=float, default=0.1)
    _add_arg('--div-lam', type=float, default=0.0)
    _add_arg('--component-lam', type=float, default=0.5)
    _add_arg('--context-gap-lam', type=float, default=0.1)
    _add_arg('--context-margin', type=float, default=0.2)
    _add_arg('--smooth-lam', type=float, default=1e-4)
    _add_arg('--center-lam', type=float, default=0.1)
    _add_arg('--center-margin', type=float, default=0.5)
    _add_arg('--score-w-init', default='1.0,0.5,1.0')
    _add_arg('--sdf-prior-weight', type=float, default=0.2)
    _add_arg(
        '--sdf-prior-mode',
        choices=['mean', 'max', 'topmean', 'entropy', 'max_entropy', 'top_entropy', 'max_top', 'gap', 'energy', 'knn3'],
        default='mean',
    )
    _add_arg('--sdf-topk', type=int, default=3)
    _add_arg('--sdf-logit-norm', choices=['none', 'disc', 'prior', 'both', 'gate'], default='none')
    _add_arg(
        '--score-fusion-mode',
        choices=['linear', 'zscore', 'prob_avg', 'prob_max', 'context_only', 'logit_max', 'context_boost', 'context_agree'],
        default='linear',
    )
    _add_arg('--test-score-mode', choices=['same', 'short', 'long', 'mlp', 'context', 'prob_max'], default='same')
    _add_arg('--center-compact-lam', type=float, default=1.0)
    _add_arg('--center-sep-lam', type=float, default=1.0)
    _add_arg('--center-temporal-lam', type=float, default=1.0)
    _add_arg('--center-div-lam', type=float, default=0.01)
    _add_arg('--context-mil-lam', type=float, default=0.0)
    _add_arg('--context-score-smooth-lam', type=float, default=0.0)
    _add_arg('--center-balance-lam', type=float, default=0.0)
    _add_arg('--center-balance-tau', type=float, default=0.2)
    _add_arg('--report-sdf-stats', action='store_true', default=False)
    _add_arg('--short-k', type=int, default=None, help='number of normal prototypes in the short branch')
    _add_arg('--long-k', type=int, default=None, help='number of normal prototypes in the long branch')
    _add_arg('--test-smooth-kernel', type=int, default=1)
    _add_arg('--test-smooth-mode', choices=['none', 'avg', 'max', 'avgmax', 'maxavg'], default='none')
    _add_arg('--train-score-smooth-kernel', type=int, default=1)
    _add_arg('--train-score-smooth-mode', choices=['none', 'avg', 'max'], default='none')
    _add_arg('--crop-fusion', choices=['mean', 'max', 'meanmax'], default='mean')
    _add_arg('--ucf-10crop-train-list', default='list_vadclip_10crop/ucf-clip-train-10crop-merged.list')
    _add_arg('--ucf-10crop-test-list', default='list_vadclip_10crop/ucf-clip-test-10crop-merged.list')
    _add_arg('--use-ucf-10crop-gt', action='store_true', default=True)
    _add_arg('--ucf-10crop-gt', default='list_vadclip_10crop/gt-ucf-clip-ifs-vad-10crop.npy')
    _add_arg('--mil-value-mode', choices=['model', 'short', 'long', 'mlp', 'context'], default='model')
    _add_arg('--mil-proposal-mode', choices=['same', 'model', 'short', 'long', 'mlp', 'context', 'prob_max'], default='same')
    _add_arg('--score-combo', choices=['mlp_context', 'context', 'mlp'], default='mlp_context')
    _add_arg('--disable-ffn', action='store_true', default=False)
    _add_arg('--short-sdf-weight', type=float, default=0.2)
    _add_arg('--long-sdf-weight', type=float, default=0.2)
    _add_arg('--short-long-fusion', choices=['learned', 'mean', 'max', 'short', 'long'], default='learned')
    _add_arg('--long-head', choices=['temporal', 'mlp'], default='temporal')
    _add_arg('--long-temporal-kernel', type=int, default=3)
    _add_arg('--use-dilated-encoder', dest='use_dilated_encoder', action='store_true', default=True)
    _add_arg('--no-use-dilated-encoder', dest='use_dilated_encoder', action='store_false')
    _add_arg('--use-long-branch', dest='use_long_branch', action='store_true', default=True)
    _add_arg('--no-use-long-branch', dest='use_long_branch', action='store_false')
    _add_arg('--use-sdf-prototype', dest='use_sdf_prototype', action='store_true', default=True)
    _add_arg('--no-use-sdf-prototype', dest='use_sdf_prototype', action='store_false')
    _add_arg('--semantic-topk', type=int, default=5)
    _add_arg('--selection-metric', choices=['main', 'short', 'long'], default='main')
    _add_arg('--early-stop-drop', type=float, default=0.0)
    _add_arg('--early-stop-min-epoch', type=int, default=20)
    _add_arg('--fixed-lr', type=float, default=0.001)
    _add_arg('--deterministic', action='store_true', default=True, help='use deterministic cudnn/cublas')
    _add_arg('--no-deterministic', dest='deterministic', action='store_false')

    pre_step = 0
    pre_auc = -1
    args = option.parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpus}'
    _set_seed(args.seed, deterministic=args.deterministic)
    if args.deterministic:
        print(f'deterministic training enabled (seed={args.seed})')

    config = Config(args)
    config.lr = [args.fixed_lr] * 5000
    metric_name = _metric_name(args)

    if args.dataset == 'shanghai':
        args.feature_size = 2048 if args.feature_extractor == 'i3d' else 512
        save_auc = 0.97
    elif args.dataset == 'ucf':
        if args.feature_extractor == 'i3d':
            args.feature_size = 2048
            save_auc = 0.84
        else:
            args.feature_size = 512
            save_auc = 0.85
    else:
        save_auc = 0.85
        args.feature_size = 1024 if args.feature_extractor == 'i3d' else 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_list = args.ucf_10crop_train_list if args.dataset == 'ucf' and args.feature_extractor == 'clip' else None
    test_list = args.ucf_10crop_test_list if args.dataset == 'ucf' and args.feature_extractor == 'clip' else None
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True, list_file=train_list), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True, generator=generator)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False, list_file=train_list), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True, generator=generator)
    test_loader = DataLoader(Dataset(args, test_mode=True, list_file=test_list), batch_size=1, shuffle=False, num_workers=0, pin_memory=False, generator=generator)

    model = ContextScoreShortLongSDF10CropTryModel(
        1,
        args.batch_size,
        n_features=args.feature_size,
        context_mode=args.context_mode,
        dilations=_parse_dilations(args.hd_dilations),
        num_normal_centers=args.num_normal_centers,
        short_k=args.short_k,
        long_k=args.long_k,
        local_kernel=args.local_kernel,
        anchor_k=args.anchor_k,
        score_w_init=_parse_float_tuple(args.score_w_init),
        score_combo=args.score_combo,
        scale_gate_mode=args.scale_gate_mode,
        sdf_prior_weight=args.sdf_prior_weight,
        sdf_prior_mode=args.sdf_prior_mode,
        sdf_topk=args.sdf_topk,
        sdf_logit_norm=args.sdf_logit_norm,
        score_fusion_mode=args.score_fusion_mode,
        test_score_mode=args.test_score_mode,
        crop_fusion=args.crop_fusion,
        use_ffn=not args.disable_ffn,
        short_sdf_weight=args.short_sdf_weight,
        long_sdf_weight=args.long_sdf_weight,
        short_long_fusion=args.short_long_fusion,
        long_head=args.long_head,
        long_temporal_kernel=args.long_temporal_kernel,
        use_dilated_encoder=args.use_dilated_encoder,
        use_long_branch=args.use_long_branch,
        use_sdf_prototype=args.use_sdf_prototype,
    ).to(device)

    os.makedirs('./ckpt', exist_ok=True)
    os.makedirs('./result', exist_ok=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr[0], weight_decay=0.0005)

    test_info = {'epoch': [], 'test_AUC': [], 'selection_AUC': []}
    best_AUC = max(-1, pre_auc)
    loadern_iter = iter(train_nloader)
    loadera_iter = iter(train_aloader)

    for step in tqdm(range(1, args.max_epoch + 1), initial=pre_step, total=args.max_epoch, dynamic_ncols=True):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.lr[step - 1]

        train_context_score(
            loadern_iter,
            loadera_iter,
            model,
            args.batch_size,
            optimizer,
            device,
            sparse_lam=args.sparse_lam,
            txt_lam=0.0,
            div_lam=0.0,
            component_lam=args.component_lam,
            context_gap_lam=args.context_gap_lam,
            context_margin=args.context_margin,
            smooth_lam=0.0,
            center_lam=args.center_lam,
            center_margin=args.center_margin,
            center_compact_lam=args.center_compact_lam,
            center_sep_lam=args.center_sep_lam,
            center_temporal_lam=0.0,
            center_div_lam=0.0,
            context_mil_lam=args.context_mil_lam,
            context_score_smooth_lam=args.context_score_smooth_lam,
            center_balance_lam=args.center_balance_lam,
            center_balance_tau=args.center_balance_tau,
            train_score_smooth_kernel=args.train_score_smooth_kernel,
            train_score_smooth_mode=args.train_score_smooth_mode,
            mil_value_mode=args.mil_value_mode,
            mil_proposal_mode=args.mil_proposal_mode,
            top_k=args.semantic_topk,
        )

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)
        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        auc, component_auc = test_with_component_auc(test_loader, model, args, device)
        selected_auc = auc
        if args.selection_metric != 'main':
            if args.selection_metric not in component_auc:
                raise ValueError(
                    f'selection_metric={args.selection_metric} is unavailable; '
                    f'available components: {sorted(component_auc.keys())}'
                )
            selected_auc = component_auc[args.selection_metric]
        test_info['epoch'].append(step + pre_step)
        test_info['test_AUC'].append(auc)
        test_info['selection_AUC'].append(selected_auc)
        if selected_auc > best_AUC:
            best_AUC = selected_auc
            _save_best_artifacts(args, model, step, selected_auc, component_auc)
            if best_AUC >= save_auc:
                name = f'{args.model_name}-context_{args.context_mode}-{args.dataset}-{step}-{args.feature_extractor}.pkl'
                torch.save(model.state_dict(), os.path.join('result', name))
        selection_suffix = '' if args.selection_metric == 'main' else f'_{args.selection_metric}'
        print(f'best{selection_suffix}_{metric_name}:' + str(best_AUC))

        if args.early_stop_drop > 0 and step >= args.early_stop_min_epoch and best_AUC - selected_auc >= args.early_stop_drop:
            print(f'early_stop: current_auc={selected_auc}, best_auc={best_AUC}, drop={best_AUC - selected_auc}')
            break
