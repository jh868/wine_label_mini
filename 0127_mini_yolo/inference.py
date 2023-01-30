import torch
import os
import glob
import cv2
import xml.etree.ElementTree as ET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('ultralytics/yolov5', 'custom', path="./runs/train/exp/weights/best.pt")  # yolov5s
model.conf = 0.5  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.to(device)

image_dir = "./dataset/test/images/"
image_path = glob.glob(os.path.join(image_dir, "*.jpg"))
label_dict = {0: "wine",
              1: "AlcoholPercentage",
              2: "Appellation AOC DOC AVARegion",
              3: "Appellation QualityLevel",
              4: "CountryCountry",
              5: "Distinct Logo",
              6: "Established YearYear",
              7: "Maker-Name",
              8: "Organic",
              9: "Sustainable",
              10: "Sweetness-Brut-SecSweetness-Brut-Sec",
              11: "TypeWine Type",
              12: "VintageYear"}

tree = ET.ElementTree()
root = ET.Element("annotations")

seen_count = 0
for img_path in image_path:
    img = cv2.imread(img_path)
    result = model(img, size=640)
    bbox = result.xyxy[0]
    image_name = os.path.basename(img_path)

    h, w, c = img.shape
    xml_frame = ET.SubElement(root, "image", id="%d" % seen_count, name=image_name,
                              width="%d" % w, height="%d" % h)

    for box in bbox:
        x1 = box[0].item()
        y1 = box[1].item()
        x2 = box[2].item()
        y2 = box[3].item()
        xtl = str(round(x1, 3))
        ytl = str(round(y1, 3))
        xbr = str(round(x2, 3))
        ybr = str(round(y2, 3))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        class_number = box[5].item()
        class_number_int = int(class_number)
        labels = label_dict[class_number_int]

        sc = box[4].item()
        ET.SubElement(xml_frame, "box", label=labels, occluded="0", source="manual",
                      xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr, z_order="0")
    seen_count += 1

    cv2.imshow("test", img)
    cv2.waitKey(1)
tree._setroot(root)
tree.write("ann.xml", encoding="utf-8")
