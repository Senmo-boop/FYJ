import os
import cv2
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 设置路径
source_dir = '/data/code2025/Q2/2025-06-02-01/HR'
output_dir = '/data/code2025/Q2/2025-06-02-01/datasets'
scales = [2, 3, 4]
test_size = 100

def make_dirs():
    for mode in ['train', 'test']:
        for s in scales:
            os.makedirs(os.path.join(output_dir, mode, f'LRx{s}'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, mode, f'HRx{s}'), exist_ok=True)

def process_single_image(args):
    img_name, test_images = args
    img_path = os.path.join(source_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ 无法读取图像: {img_path}")
        return

    h, w = img.shape[:2]
    mode = 'test' if img_name in test_images else 'train'

    for s in scales:
        # 调整HR尺寸到能被scale整除
        new_h, new_w = h - h % s, w - w % s
        hr = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 生成LR
        lr = cv2.resize(hr, (new_w // s, new_h // s), interpolation=cv2.INTER_CUBIC)

        lr_path = os.path.join(output_dir, mode, f'LRx{s}', img_name)
        hr_path = os.path.join(output_dir, mode, f'HRx{s}', img_name)

        os.makedirs(os.path.dirname(lr_path), exist_ok=True)
        os.makedirs(os.path.dirname(hr_path), exist_ok=True)

        cv2.imwrite(lr_path, lr)
        cv2.imwrite(hr_path, hr)

def process_images_parallel():
    all_images = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    random.shuffle(all_images)
    test_images = set(all_images[:test_size])

    args_list = [(img_name, test_images) for img_name in all_images]

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_single_image, args_list), total=len(all_images)))

if __name__ == '__main__':
    make_dirs()
    process_images_parallel()
    print("✅ 多进程数据集生成完毕！")
