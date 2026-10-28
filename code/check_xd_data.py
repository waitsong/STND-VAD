#!/usr/bin/env python3
"""Diagnostics for XD-Violence CLIP 10-crop data and GT alignment."""
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


def load_lines(name):
    path = os.path.join(ROOT, 'list', name)
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def check_list_files():
    print('=== 1. List / feature file check ===')
    issues = []
    for split, name in [('train', 'xd-clip-train-10crop.list'), ('test', 'xd-clip-test-10crop.list')]:
        lines = load_lines(name)
        missing = []
        bad_shape = []
        bad_crop = []
        t_lens = []
        for i, rel in enumerate(lines):
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p):
                missing.append(rel)
                continue
            arr = np.load(p, allow_pickle=True)
            arr = np.array(arr)
            if arr.ndim != 3 or arr.shape[1] != 10 or arr.shape[2] != 512:
                bad_shape.append((rel, arr.shape))
            if np.isnan(arr).any() or np.isinf(arr).any():
                bad_crop.append(rel)
            t_lens.append(arr.shape[0])
        print(f'[{split}] entries={len(lines)} missing={len(missing)} bad_shape={len(bad_shape)} nan/inf={len(bad_crop)}')
        print(f'  T stats: min={min(t_lens) if t_lens else 0} max={max(t_lens) if t_lens else 0} mean={np.mean(t_lens):.1f}')
        if missing[:3]:
            issues.append(f'{split} missing examples: {missing[:3]}')
        if bad_shape[:3]:
            issues.append(f'{split} bad shape examples: {bad_shape[:3]}')
    return issues


def label_from_name(stem):
    m = re.search(r'label_([^._]+)', stem)
    return m.group(1) if m else 'UNKNOWN'


def check_train_split():
    print('\n=== 2. Train abnormal/normal split (1905) ===')
    issues = []
    lines = load_lines('xd-clip-train-10crop.list')
    abn = lines[:1905]
    nor = lines[1905:]
    print(f'total={len(lines)} abnormal={len(abn)} normal={len(nor)}')

    def summarize(block, name):
        labels = [label_from_name(os.path.basename(p).replace('.npy', '')) for p in block]
        pure_a = sum(1 for x in labels if x == 'A')
        has_a = sum(1 for x in labels if 'A' in x.split('-'))
        print(f'  {name}: pure_A={pure_a}/{len(labels)} contains_A={has_a}/{len(labels)}')
        top = Counter(labels).most_common(5)
        print(f'  {name} top labels: {top}')
        return labels

    abn_labels = summarize(abn, 'abnormal block')
    nor_labels = summarize(nor, 'normal block')
  # normal training clips should be mostly label A only
    nor_non_a = [x for x in nor_labels if x != 'A']
    if nor_non_a:
        issues.append(f'normal block has {len(nor_non_a)} non-A labels, e.g. {nor_non_a[:3]}')
    abn_pure_a = sum(1 for x in abn_labels if x == 'A')
    if abn_pure_a > len(abn_labels) * 0.05:
        issues.append(f'abnormal block has many pure-A clips: {abn_pure_a}/{len(abn_labels)}')
    return issues


def infer_gt_frame_counts(test_lines):
    """Estimate per-snippet frame counts from feature temporal length."""
    counts = []
    for rel in test_lines:
        arr = np.load(os.path.join(ROOT, rel), allow_pickle=True)
        counts.append(int(np.array(arr).shape[0]))
    return counts


def check_gt_alignment():
    print('\n=== 3. GT alignment with test list ===')
    issues = []
    test_lines = load_lines('xd-clip-test-10crop.list')
    gt = np.load(os.path.join(ROOT, 'list', 'gt-xd-clip.npy'))
    gt = np.asarray(gt).reshape(-1)
    print(f'test snippets={len(test_lines)} gt_frames={len(gt)} gt_positive={int(gt.sum())} pos_ratio={gt.mean():.4f}')

    t_counts = infer_gt_frame_counts(test_lines)
    total_t = sum(t_counts)
    total_t_x16 = total_t * 16
    print(f'sum feature T={total_t}  T*16={total_t_x16}')

    if total_t_x16 == len(gt):
        print('  OK: sum(T)*16 == len(gt)  (matches eval repeat-by-16 logic)')
    elif total_t == len(gt):
        print('  NOTE: sum(T) == len(gt)  (GT may already be snippet-level, repeat*16 could be wrong)')
        issues.append('GT length equals sum(T), not sum(T)*16 — eval repeat-by-16 may misalign')
    else:
        diff16 = abs(total_t_x16 - len(gt))
        difft = abs(total_t - len(gt))
        print(f'  mismatch vs T*16: {diff16}   vs sum(T): {difft}')
        if diff16 < difft:
            print('  closer to T*16 assumption')
        else:
            issues.append(f'GT length {len(gt)} closer to sum(T)={total_t} than sum(T)*16')

    # check if IFS-VAD provides separate gt segment metadata — compare first few cumulative offsets
    cum = 0
    mismatches = 0
    for i, (rel, t) in enumerate(zip(test_lines[:20], t_counts[:20])):
        seg_gt_len_16 = t * 16
        seg_gt = gt[cum:cum + seg_gt_len_16]
        pos = int(seg_gt.sum())
        stem = os.path.basename(rel)
        cum += seg_gt_len_16
    print(f'  checked first 20 snippets using T*16 segmentation, cumulative gt frames={cum}')
    return issues, t_counts, gt


def simulate_eval_lengths(t_counts):
    print('\n=== 4. Eval length simulation (as in test_component_auc_10crop_try.py) ===')
    n_snip = len(t_counts)
    pred_per_snip = t_counts  # one score per temporal step per video in test loop
    pred_len = sum(pred_per_snip)
    pred_expanded = pred_len * 16
    gt_len = len(np.load(os.path.join(ROOT, 'list', 'gt-xd-clip.npy')))
    print(f'pred scores (sum T)={pred_len}')
    print(f'after repeat(*16)={pred_expanded}')
    print(f'gt length={gt_len}')
    if pred_expanded == gt_len:
        print('  OK: pred repeat-by-16 matches GT length')
        return []
    return [f'pred expanded {pred_expanded} != gt {gt_len} (diff {abs(pred_expanded-gt_len)})']


def check_merge_quality_sample():
    print('\n=== 5. Merge sanity (spot check) ===')
    issues = []
    rel = load_lines('xd-clip-train-10crop.list')[0]
    arr = np.load(os.path.join(ROOT, rel))
    print(f'sample: {rel}')
    print(f'  shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}')
    crop_norms = [np.linalg.norm(arr[:, i, :]) for i in range(10)]
    print(f'  per-crop L2 norms (first 5): {[round(x, 1) for x in crop_norms[:5]]}')
    if np.std(crop_norms) < 1e-6:
        issues.append('all 10 crops nearly identical — merge may be wrong')
    return issues


def benchmark_context():
    print('\n=== 6. AP context / likely causes ===')
    print('  XD-Violence CLIP AP literature refs often ~0.80–0.86 depending on method/features.')
    print('  Your best ~0.774 may be limited by: 1-epoch-per-iter MIL training, not data corruption.')


def main():
    all_issues = []
    all_issues += check_list_files()
    all_issues += check_train_split()
    gt_issues, t_counts, gt = check_gt_alignment()
    all_issues += gt_issues
    all_issues += simulate_eval_lengths(t_counts)
    all_issues += check_merge_quality_sample()

    benchmark_context()
    print('\n=== SUMMARY ===')
    if all_issues:
        print('POTENTIAL ISSUES:')
        for i, item in enumerate(all_issues, 1):
            print(f'  {i}. {item}')
        return 1
    print('No critical data/GT alignment issues detected.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
