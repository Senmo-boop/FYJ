# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import mmcv
import torch
from tqdm import tqdm
from mmedit.apis import generation_inference, init_model
from mmedit.utils import modify_args
import sys
sys.path.append('/data/code2024/Q4/2025-01-17-02/')
import cv2

def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Generation demo')
    parser.add_argument('--config', default="../my_config/RetinexNet_tr.py", help='test config file path')
    parser.add_argument('--checkpoint', default="../work_dirs/RetinexNet_tr/iter_5000.pth", help='checkpoint file')
    parser.add_argument('--img_path', help='path to input image file')
    parser.add_argument('--save_path', default="../datasets/pred/RetinexNet_tr", help='path to save generation result')
    parser.add_argument(
        '--unpaired_path', default="../datasets/test/low", help='path to unpaired image file')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=1, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
    model = init_model(args.config, args.checkpoint, device=device)
    file_list = os.listdir(args.unpaired_path)
    for file_name in tqdm(file_list):
        output = generation_inference(model, args.img_path, os.path.join(args.unpaired_path, file_name), patch_inference=False, patch_size=256)
        # output = cv2.resize(output, (614, 614))
        mmcv.imwrite(output, os.path.join(args.save_path, file_name))


if __name__ == '__main__':
    main()
