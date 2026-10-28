#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict

import numpy as np


def merge_split(src_dir, dst_dir, expected_crops=10):
    os.makedirs(dst_dir, exist_ok=True)
    groups = defaultdict(list)
    for fname in os.listdir(src_dir):
        if not fname.endswith('.npy') or '__' not in fname:
            continue
        stem, crop_id = fname.rsplit('__', 1)
        crop_id = crop_id[:-4]
        if crop_id.isdigit():
            groups[stem].append((int(crop_id), fname))

    stems = sorted(groups.keys())
    merged = skipped = failed = 0
    for stem in stems:
        out_path = os.path.join(dst_dir, stem + '.npy')
        if os.path.exists(out_path):
            skipped += 1
            continue

        items = sorted(groups[stem], key=lambda x: x[0])
        if len(items) != expected_crops:
            print(f'[warn] skip {stem}: expected {expected_crops} crops, got {len(items)}')
            failed += 1
            continue

        crops = []
        for crop_id, fname in items:
            feat = np.array(np.load(os.path.join(src_dir, fname), allow_pickle=True), dtype=np.float32)
            if feat.ndim != 2:
                print(f'[warn] skip {stem}: {fname} has shape {feat.shape}')
                failed += 1
                crops = None
                break
            crops.append(feat)

        if crops is None:
            continue

        lengths = {c.shape[0] for c in crops}
        if len(lengths) != 1:
            print(f'[warn] skip {stem}: mismatched temporal lengths {sorted(lengths)}')
            failed += 1
            continue

        merged_feat = np.stack(crops, axis=1)
        np.save(out_path, merged_feat)
        merged += 1
        if merged % 200 == 0:
            print(f'[progress] merged {merged}, skipped {skipped}, failed {failed}')

    print(f'[done] src={src_dir} dst={dst_dir} merged={merged} skipped={skipped} failed={failed} total={len(stems)}')


def main():
    parser = argparse.ArgumentParser(description='Merge XD per-crop CLIP features into 10-crop tensors.')
    parser.add_argument('--src-train', default='../../XD-Violence/CLIP_features/XDTrainClipFeatures')
    parser.add_argument('--src-test', default='../../XD-Violence/CLIP_features/XDTestClipFeatures')
    parser.add_argument('--dst-train', default='../features/xd_clip_10crop/train')
    parser.add_argument('--dst-test', default='../features/xd_clip_10crop/test')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_train = os.path.normpath(os.path.join(script_dir, args.src_train))
    src_test = os.path.normpath(os.path.join(script_dir, args.src_test))
    dst_train = os.path.normpath(os.path.join(script_dir, args.dst_train))
    dst_test = os.path.normpath(os.path.join(script_dir, args.dst_test))

    merge_split(src_train, dst_train)
    merge_split(src_test, dst_test)


if __name__ == '__main__':
    main()
