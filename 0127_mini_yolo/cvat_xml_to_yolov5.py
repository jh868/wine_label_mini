# cvat xml to yolo

import os
import glob
import cv2
from xml.etree.ElementTree import parse


# xml 1 ~ 5
def find_xml_file(xml_folder_path):
    all_root = []
    for (path, dir, files) in os.walk(xml_folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            # ext -> .xml
            if ext == ".xml":
                root = os.path.join(path, filename)
                # ./xml_data/test.xmla
                all_root.append(root)
            else:
                print("no xml files.....")
                break

    return all_root


xml_folder_dir = "./wine labels_voc_dataset/labels"
xml_paths = find_xml_file(xml_folder_dir)
# test = glob.glob(os.path.join(xml_folder_path, "*.xml"))

label_dict = {
    'wine': 0,
    'AlcoholPercentage': 1,
    'Appellation AOC DOC AVARegion': 2,
    'Appellation QualityLevel': 3,
    'CountryCountry': 4,
    'Distinct Logo': 5,
    'Established YearYear': 6,
    'Maker-Name': 7,
    'Organic': 8,
    'Sustainable': 9,
    'Sweetness-Brut-SecSweetness-Brut-Sec': 10,
    'TypeWine Type': 11,
    'VintageYear': 12}

for xml_path in xml_paths:
    tree = parse(xml_path)
    root = tree.getroot()
    img_metas = root.findall("filename")
    for img_meta in img_metas:
        # xml image name
        # image_name = img_meta.attrib["name"]
        image_name = img_meta.text
        # print(image_name)
        # image_name >> aditganteng_mp4-165_jpg.rf.976ae8b8ed6d79aab3f9566dba1f4645.jpg

        img_sizes = root.findall("size")
        for img_size in img_sizes:
            img_width = int(img_size.find('width').text)
            img_height = int(img_size.find('height').text)

            box_metas = root.findall('object')
            for box_meta in box_metas:
                box_label = box_meta.find('name').text
                box = [int(box_meta.find('bndbox').find('xmin').text),
                       int(box_meta.find('bndbox').find('xmax').text),
                       int(box_meta.find('bndbox').find('ymin').text),
                       int(box_meta.find('bndbox').find('ymax').text)
                       ]

                yolo_x = round(((box[0] + box[1]) / 2) / img_width, 6)
                yolo_y = round(((box[2] + box[3]) / 2) / img_height, 6)
                yolo_w = round((box[1] - box[0]) / img_width, 6)
                yolo_h = round((box[3] - box[2]) / img_height, 6)

                print("yolo xywh", yolo_x, yolo_y, yolo_w, yolo_h)
                image_name_temp = image_name.replace(".jpg", ".txt")

                # txt file save folder
                os.makedirs("./wine labels_voc_dataset/yolo", exist_ok=True)

                # label
                label = label_dict[box_label]

                # txt save
                with open(f"./wine labels_voc_dataset/yolo/{image_name_temp}", 'a') as f:
                    f.write(f"{label} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n")
