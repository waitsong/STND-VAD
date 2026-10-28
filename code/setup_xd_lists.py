#!/usr/bin/env python3
import os
import urllib.request


ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
LIST_DIR = os.path.join(ROOT, 'list')
BASE_URL = 'https://raw.githubusercontent.com/Ria5331/IFS-VAD/main/list'

FILES = {
    'xd-clip-train-10crop.list': (
        'xd-clip-train-10crop.list',
        'list/XD_Train_ten_crop_clip/',
        'features/xd_clip_10crop/train/',
    ),
    'xd-clip-test-10crop.list': (
        'xd-clip-test-10crop.list',
        'list/XD_Test_ten_crop_clip/',
        'features/xd_clip_10crop/test/',
    ),
}


def main():
    os.makedirs(LIST_DIR, exist_ok=True)
    for out_name, (remote_name, old_prefix, new_prefix) in FILES.items():
        url = f'{BASE_URL}/{remote_name}'
        with urllib.request.urlopen(url) as resp:
            content = resp.read().decode('utf-8')
        content = content.replace(old_prefix, new_prefix)
        out_path = os.path.join(LIST_DIR, out_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'wrote {out_path} ({content.count(chr(10))} lines)')

    gt_url = f'{BASE_URL}/gt-xd-clip.npy'
    gt_path = os.path.join(LIST_DIR, 'gt-xd-clip.npy')
    urllib.request.urlretrieve(gt_url, gt_path)
    print(f'wrote {gt_path}')


if __name__ == '__main__':
    main()
