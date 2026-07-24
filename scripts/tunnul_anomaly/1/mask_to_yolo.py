import os
import cv2
import numpy as np
from tqdm import tqdm

def mask_to_yolo_by_order(mask_dir, img_dir, output_dir, neg_list_path, mapping_path,
                          crack_thresh=(50, 150), leakage_thresh=(180, 255),
                          min_area=50, visualize=False):
    """
    按顺序匹配img和mask，仅当检测到病害时才生成标注文件，
    同时记录 txt 文件名与原始图片文件名的映射关系。
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    if len(mask_files) != len(img_files):
        print(f"警告：mask 文件数 {len(mask_files)}，img 文件数 {len(img_files)}，数量不一致！")
        min_len = min(len(mask_files), len(img_files))
        mask_files = mask_files[:min_len]
        img_files = img_files[:min_len]

    neg_list = []          # 无病害图片文件名（原始图像）
    mapping = []           # 每个有病害的txt与原始图片文件名的对应关系

    for mask_file, img_file in tqdm(zip(mask_files, img_files), total=len(mask_files), desc="Processing"):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取 mask: {mask_path}，跳过")
            continue

        h, w = mask.shape

        crack_mask = cv2.inRange(mask, crack_thresh[0], crack_thresh[1])
        leakage_mask = cv2.inRange(mask, leakage_thresh[0], leakage_thresh[1])

        crack_contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        leakage_contours, _ = cv2.findContours(leakage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crack_contours = [cnt for cnt in crack_contours if cv2.contourArea(cnt) >= min_area]
        leakage_contours = [cnt for cnt in leakage_contours if cv2.contourArea(cnt) >= min_area]

        if not crack_contours and not leakage_contours:
            neg_list.append(img_file)   # 无病害，记录图片名
        else:
            # 生成 txt 文件名（以 mask 文件名命名）
            txt_name = mask_file.replace(mask_file.split('.')[-1], 'txt')
            txt_path = os.path.join(output_dir, txt_name)

            with open(txt_path, 'w') as f:
                for cnt in crack_contours:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    x_center = (x + bw/2) / w
                    y_center = (y + bh/2) / h
                    bw_norm = bw / w
                    bh_norm = bh / h
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")
                for cnt in leakage_contours:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    x_center = (x + bw/2) / w
                    y_center = (y + bh/2) / h
                    bw_norm = bw / w
                    bh_norm = bh / h
                    f.write(f"1 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

            # 记录映射关系
            mapping.append(f"{txt_name},{img_file}")

        # 可视化（可选）
        if visualize:
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            for cnt in crack_contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            for cnt in leakage_contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
            cv2.imshow('Result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # 保存负样本列表
    with open(neg_list_path, 'w') as f:
        for name in neg_list:
            f.write(name + '\n')
    print(f"负样本列表已保存至 {neg_list_path}，共 {len(neg_list)} 张无病害图片")

    # 保存映射文件
    with open(mapping_path, 'w') as f:
        f.write("txt_filename,img_filename\n")  # 可选的标题行
        for line in mapping:
            f.write(line + '\n')
    print(f"映射文件已保存至 {mapping_path}，共 {len(mapping)} 条记录")

    print(f"有病害的标注文件已保存至 {output_dir}")

if __name__ == "__main__":
    # ========== 配置区域 ==========
    mask_dir = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\3_mask"
    img_dir = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\3_img"
    output_dir = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\labels"
    neg_list_path = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\negatives.txt"
    mapping_path = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\TACK_Tunnel_Data\mapping.csv"
    crack_thresh = (50, 150)
    leakage_thresh = (180, 255)
    min_area = 50
    visualize = False   # 批量处理时关闭可视化
    # =============================

    mask_to_yolo_by_order(mask_dir, img_dir, output_dir, neg_list_path, mapping_path,
                          crack_thresh, leakage_thresh, min_area, visualize)