import os
import xml.etree.ElementTree as ET


# 读取 xml 类型标注文件
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1) if name == "person" else labels.append(0)

    return boxes, labels


if __name__ == "__main__":
    from read_cfg import load_yaml_as_namespace
    from func import get_absolute_path

    file_path = "utils/config.yaml"                             # 替换成你的YAML文件路径
    args = load_yaml_as_namespace(get_absolute_path(file_path))

    image_folder = args.image_folder
    xml_folder = args.xml_folder
    # 遍历文件夹
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_folder, filename)
            image_filename = filename.replace('.xml', '.jpg')
            image_path = os.path.join(image_folder, image_filename)
            
            # 解析 XML 文件
            annotations = parse_xml(xml_path)
            print(f"Image: {image_filename}, Annotations: {annotations}")
