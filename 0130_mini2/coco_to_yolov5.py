import os
import glob
import cv2
import json

path = './wine labels_coco/'

all_json_path = glob.glob(os.path.join(path, 'valid', '*.json'))
# print(all_json_path)

for json_path in all_json_path:
    with open(json_path, 'r') as f:
        data = json.load(f)
    label = data['categories']
    img_info = data['images']

    for anno in data['annotations']:
        image_id = anno['image_id']
        category_id = anno['category_id']

        for info in img_info:
            if info['id'] == image_id:
                filename, img_width, img_height = info['file_name'], info['width'], info['height']
                # print(filename, img_width, img_height)

                yolo_x = round((anno['bbox'][0] + anno['bbox'][2] / 2) / img_width, 6)
                yolo_y = round((anno['bbox'][1] + anno['bbox'][3] / 2) / img_height, 6)
                yolo_w = round((anno['bbox'][2]) / img_width, 6)
                yolo_h = round((anno['bbox'][3]) / img_height, 6)

                os.makedirs('./wine labels_coco/valid/labels/', exist_ok=True)
                filename_temp = filename.replace('.jpg', '.txt')
                with open(f"./wine labels_coco/valid/labels/{filename_temp}", 'a') as f:
                    f.write(f"{anno['category_id']} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n")