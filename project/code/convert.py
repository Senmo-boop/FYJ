import os
import pydicom
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count


def process_dcm(args):
    dcm_path, src_root, dst_root = args
    try:
        # 读取 DICOM 文件
        dcm = pydicom.dcmread(dcm_path)
        pixel_array = dcm.pixel_array

        # 归一化像素值
        image = pixel_array.astype(float)
        image = (np.maximum(image, 0) / image.max()) * 255.0
        image = image.astype("uint8")

        # 生成唯一文件名（防止重名）
        relative_path = os.path.relpath(os.path.dirname(dcm_path), src_root)
        unique_prefix = relative_path.replace(os.sep, "_")
        png_filename = f"{unique_prefix}_{os.path.splitext(os.path.basename(dcm_path))[0]}.png"
        png_path = os.path.join(dst_root, png_filename)

        # 保存 PNG
        cv2.imwrite(png_path, image)
        print(f"[OK] {png_filename}")

    except Exception as e:
        print(f"[ERROR] {dcm_path}: {e}")


def dcm_to_png_parallel(src_root, dst_root, num_workers=None):
    os.makedirs(dst_root, exist_ok=True)

    # 收集所有 DICOM 文件路径
    dcm_files = []
    for root, _, files in os.walk(src_root):
        for file in files:
            if file.endswith(".dcm"):
                dcm_files.append((os.path.join(root, file), src_root, dst_root))

    # 使用多进程池进行处理
    workers = num_workers or cpu_count()
    print(f"Starting with {workers} workers, total files: {len(dcm_files)}")
    with Pool(workers) as pool:
        pool.map(process_dcm, dcm_files)


if __name__ == '__main__':
    src_folder = r"G:/code/code/datasets/fastMRI_brain_DICOM"
    dst_folder = r"F:/HR"
    dcm_to_png_parallel(src_folder, dst_folder)
