import os
import cv2
from tqdm.contrib.concurrent import process_map
from functools import partial

def process_image(img_name, src_dir, lr2x_dir, hr2x_dir, lr4x_dir, hr4x_dir):
    img_path = os.path.join(src_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    h, w = img.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    h4, w4 = h - h % 4, w - w % 4

    img_2x = cv2.resize(img[:h2, :w2], (w2 // 2, h2 // 2), interpolation=cv2.INTER_CUBIC)
    img_4x = cv2.resize(img[:h4, :w4], (w4 // 4, h4 // 4), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(os.path.join(hr2x_dir, img_name), img[:h2, :w2])
    cv2.imwrite(os.path.join(lr2x_dir, img_name), img_2x)
    cv2.imwrite(os.path.join(hr4x_dir, img_name), img[:h4, :w4])
    cv2.imwrite(os.path.join(lr4x_dir, img_name), img_4x)


if __name__ == '__main__':
    # 路径设置（使用原始字符串避免转义问题）
    src_dir = r'F:\HR'         # ← 改成你的实际路径
    output_base = r'F:\train dataests'   # ← 改成你的实际路径

    lr2x_dir = os.path.join(output_base, 'LR2x')
    hr2x_dir = os.path.join(output_base, 'HR2x')
    lr4x_dir = os.path.join(output_base, 'LR4x')
    hr4x_dir = os.path.join(output_base, 'HR4x')

    # 创建输出文件夹
    for d in [lr2x_dir, hr2x_dir, lr4x_dir, hr4x_dir]:
        os.makedirs(d, exist_ok=True)

    # 获取图像列表
    img_list = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 多进程处理（必须在 __main__ 中调用）
    process_map(
        partial(process_image, src_dir=src_dir, lr2x_dir=lr2x_dir,
                hr2x_dir=hr2x_dir, lr4x_dir=lr4x_dir, hr4x_dir=hr4x_dir),
        img_list,
        max_workers=os.cpu_count(),
        chunksize=10
    )