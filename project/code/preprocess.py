import os
import random
import argparse
import json
import shutil
import pydicom
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map
from functools import partial

# ==================== 第一部分：随机分配子文件夹 ====================
def split_folders(source_dir, train_count=100, val_count=10, test_count=10):
    """
    从 source_dir 中随机抽取指定数量的子文件夹名称，不重复。
    返回 (train_folders, val_folders, test_folders) 三个列表。
    """
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"源目录不存在：{source_dir}")

    all_subdirs = [d for d in os.listdir(source_dir)
                   if os.path.isdir(os.path.join(source_dir, d))]
    total_needed = train_count + val_count + test_count
    if len(all_subdirs) < total_needed:
        raise ValueError(f"源目录下只有 {len(all_subdirs)} 个子文件夹，不足 {total_needed} 个")

    selected = random.sample(all_subdirs, total_needed)
    train_folders = selected[:train_count]
    val_folders = selected[train_count:train_count + val_count]
    test_folders = selected[train_count + val_count:]
    return train_folders, val_folders, test_folders

# ==================== 第二部分：DICOM → PNG ====================
def process_dcm(args):
    """单个DICOM文件的处理函数（供多进程调用）"""
    dcm_path, src_root, dst_root = args
    try:
        dcm = pydicom.dcmread(dcm_path)
        pixel_array = dcm.pixel_array

        # 归一化到[0,255]
        image = pixel_array.astype(float)
        # 防止除零
        max_val = image.max()
        if max_val > 0:
            image = (np.maximum(image, 0) / max_val) * 255.0
        else:
            image = np.zeros_like(image, dtype=np.uint8)
        image = image.astype("uint8")

        # 生成唯一文件名（使用相对路径前缀避免重名）
        relative_path = os.path.relpath(os.path.dirname(dcm_path), src_root)
        unique_prefix = relative_path.replace(os.sep, "_")
        png_filename = f"{unique_prefix}_{os.path.splitext(os.path.basename(dcm_path))[0]}.png"
        png_path = os.path.join(dst_root, png_filename)

        cv2.imwrite(png_path, image)
    except Exception as e:
        print(f"[ERROR] {dcm_path}: {e}")

def dcm_to_png_selected(folder_list, src_root, dst_hr_root, num_workers=None):
    """
    将 folder_list 中所有子文件夹内的 .dcm 文件转换为 PNG，保存到 dst_hr_root。
    """
    os.makedirs(dst_hr_root, exist_ok=True)

    # 收集所有需要处理的DICOM文件路径
    dcm_files = []
    for folder in folder_list:
        folder_path = os.path.join(src_root, folder)
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".dcm"):
                    dcm_files.append((os.path.join(root, file), src_root, dst_hr_root))

    if not dcm_files:
        print(f"警告：所选文件夹中未找到任何 .dcm 文件。")
        return

    workers = num_workers or cpu_count()
    print(f"开始转换DICOM，使用 {workers} 个工作进程，共 {len(dcm_files)} 个文件")
    with Pool(workers) as pool:
        pool.map(process_dcm, dcm_files)
    print("DICOM转换完成。")

# ==================== 第三部分：生成超分辨率数据集 ====================
def process_image(img_name, src_dir, lr2x_dir, hr2x_dir, lr4x_dir, hr4x_dir):
    """单个图像的处理函数：生成2倍和4倍的LR/HR对"""
    img_path = os.path.join(src_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"无法读取图像：{img_path}")
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

def generate_lr_hr_pairs(src_dir, output_base, num_workers=None):
    """读取src_dir中的图像，生成LR/HR对并保存到output_base下的四个子文件夹"""
    lr2x_dir = os.path.join(output_base, 'LR2x')
    hr2x_dir = os.path.join(output_base, 'HR2x')
    lr4x_dir = os.path.join(output_base, 'LR4x')
    hr4x_dir = os.path.join(output_base, 'HR4x')

    for d in [lr2x_dir, hr2x_dir, lr4x_dir, hr4x_dir]:
        os.makedirs(d, exist_ok=True)

    # 获取所有图像文件（支持png/jpg/jpeg）
    img_list = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_list:
        print(f"警告：在 {src_dir} 中未找到任何图像文件。")
        return

    workers = num_workers or cpu_count()
    print(f"开始生成LR/HR对，使用 {workers} 个工作进程，共 {len(img_list)} 个文件")

    process_map(
        partial(process_image, src_dir=src_dir,
                lr2x_dir=lr2x_dir, hr2x_dir=hr2x_dir,
                lr4x_dir=lr4x_dir, hr4x_dir=hr4x_dir),
        img_list,
        max_workers=workers,
        chunksize=10
    )
    print("LR/HR对生成完成。")

# ==================== 第四部分：处理单个集合（训练/验证/测试）====================
def process_collection(folder_list, src_root, dst_collection_root, num_workers=None):
    """
    对一个集合（如训练集）执行完整流程：
    1. 将 folder_list 中所有子文件夹内的 DICOM 转换为 PNG，保存在 dst_collection_root/HR
    2. 基于 HR 生成 LR/HR 对，保存在 dst_collection_root 下的 LR2x, HR2x, LR4x, HR4x
    """
    hr_dir = os.path.join(dst_collection_root, "HR")
    # 第一步：DICOM转PNG
    dcm_to_png_selected(folder_list, src_root, hr_dir, num_workers)
    # 第二步：生成超分数据集
    generate_lr_hr_pairs(hr_dir, dst_collection_root, num_workers)

# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description="从DICOM文件夹中随机分配子文件夹并生成超分辨率数据集")
    parser.add_argument("--source", required=True, help="包含众多子文件夹的原始DICOM大文件夹路径")
    parser.add_argument("--output", required=True, help="输出根目录路径")
    parser.add_argument("--train", type=int, default=100, help="训练集抽取的文件夹数量 (默认: 100)")
    parser.add_argument("--val", type=int, default=10, help="验证集抽取的文件夹数量 (默认: 10)")
    parser.add_argument("--test", type=int, default=10, help="测试集抽取的文件夹数量 (默认: 10)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，用于可重现的分割")
    parser.add_argument("--overwrite", action="store_true", help="如果输出目录已存在，是否覆盖（清空后重新生成）")
    args = parser.parse_args()

    source_root = args.source
    output_root = args.output
    train_count = args.train
    val_count = args.val
    test_count = args.test
    seed = args.seed
    overwrite = args.overwrite

    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        print(f"随机种子已设置为: {seed}")

    # 检查输出目录是否存在且非空
    if os.path.exists(output_root) and os.listdir(output_root):
        if overwrite:
            print(f"输出目录 {output_root} 已存在，正在清空（--overwrite 已指定）...")
            shutil.rmtree(output_root)
            os.makedirs(output_root, exist_ok=True)
        else:
            raise RuntimeError(
                f"输出目录 {output_root} 已存在且非空。请使用 --overwrite 覆盖，或指定一个新的输出目录以避免数据泄漏。"
            )
    else:
        os.makedirs(output_root, exist_ok=True)

    # 1. 随机分配子文件夹
    print("正在随机分配子文件夹...")
    train_folders, val_folders, test_folders = split_folders(
        source_root, train_count, val_count, test_count
    )
    print(f"训练集：{len(train_folders)} 个文件夹")
    print(f"验证集：{len(val_folders)} 个文件夹")
    print(f"测试集：{len(test_folders)} 个文件夹")

    # 2. 保存分割清单（JSON格式）
    split_info = {
        "seed": seed,
        "train": train_folders,
        "val": val_folders,
        "test": test_folders,
        "source": source_root,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count
    }
    split_json_path = os.path.join(output_root, "split.json")
    with open(split_json_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4, ensure_ascii=False)
    print(f"分割清单已保存至: {split_json_path}")

    # 3. 分别处理三个集合
    collections = [
        (train_folders, os.path.join(output_root, "train datasets")),
        (val_folders,   os.path.join(output_root, "val datasets")),
        (test_folders,  os.path.join(output_root, "test datasets"))
    ]

    for folders, dst_root in collections:
        if not folders:
            continue
        print("\n" + "="*60)
        print(f"正在处理集合：{os.path.basename(dst_root)}")
        print("="*60)
        process_collection(folders, source_root, dst_root, num_workers=None)

    print("\n所有处理完成！")
    print(f"结果保存在：{output_root}")
    for name in ["train datasets", "val datasets", "test datasets"]:
        path = os.path.join(output_root, name)
        if os.path.exists(path):
            print(f"  - {name}：{path}")

if __name__ == "__main__":
    main()
