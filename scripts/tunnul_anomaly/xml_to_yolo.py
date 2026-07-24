import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

def convert_voc_to_yolo(xml_dir, output_dir, classes):
    """
    将VOC XML标注转换为YOLO格式txt文件

    Args:
        xml_dir: 存放XML文件的文件夹路径
        output_dir: 输出txt文件的文件夹路径（会自动创建）
        classes: 类别名称列表，例如 ['crack', 'leakage']
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有xml文件
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    for xml_file in tqdm(xml_files, desc="转换中"):
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图片尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 输出txt文件名（与xml同名）
        txt_name = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(output_dir, txt_name)

        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                # 类别名称
                name = obj.find('name').text
                if name not in classes:
                    print(f"警告：未知类别 '{name}' 在文件 {xml_file} 中，已跳过")
                    continue
                class_id = classes.index(name)

                # 边界框坐标（VOC格式：xmin, ymin, xmax, ymax）
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                # 转换为YOLO格式：中心点归一化坐标和归一化宽高
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height

                # 写入文件（保留6位小数）
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"转换完成！共处理 {len(xml_files)} 个文件，输出到 {output_dir}")

if __name__ == "__main__":
    # ========== 配置区域 ==========
    # 你的XML文件夹路径
    xml_dir = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\Generalization Test-Object Detection\Annotations"
    # 输出YOLO标签文件夹路径
    output_dir = r"E:\MyProject\Datasets\tunnel_anomaly\dataset\Generalization Test-Object Detection\labels"
    # 你的类别名称列表（按顺序，与模型训练保持一致）
    classes = ['Crack', 'Leakage','Spalling']   # 根据实际修改
    # =============================

    convert_voc_to_yolo(xml_dir, output_dir, classes)