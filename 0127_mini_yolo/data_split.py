import os
import glob
from sklearn.model_selection import train_test_split
from shutil import copyfile

file_path = './wine labels_voc_dataset/'

image_names = glob.glob(os.path.join(file_path, 'images', '*.jpg'))
label_names = glob.glob(os.path.join(file_path, 'yolo', '*.txt'))

train_images, tmp_images, train_labels, tmp_labels = train_test_split(
    image_names, label_names,test_size=0.2, random_state=42)

val_images, test_images, val_labels, test_labels = train_test_split(
    tmp_images, tmp_labels, test_size=0.5, random_state=42)

os.makedirs('./dataset/train/images/', exist_ok=True)
os.makedirs('./dataset/valid/images/', exist_ok=True)
os.makedirs('./dataset/test/images/', exist_ok=True)
os.makedirs('./dataset/train/labels/', exist_ok=True)
os.makedirs('./dataset/valid/labels/', exist_ok=True)
os.makedirs('./dataset/test/labels/', exist_ok=True)

for train_image in train_images:
    copyfile(train_image, './dataset/train/images/' + train_image.split('\\')[-1])
for val_image in val_images:
    copyfile(val_image, './dataset/valid/images/' + val_image.split('\\')[-1])
for test_image in test_images:
    copyfile(test_image, './dataset/test/images/' + test_image.split('\\')[-1])
for train_label in train_labels:
    copyfile(train_label, './dataset/train/labels/' + train_label.split('\\')[-1])
for val_label in val_labels:
    copyfile(val_label, './dataset/valid/labels/' + val_label.split('\\')[-1])
for test_label in test_labels:
    copyfile(test_label, './dataset/test/labels/' + test_label.split('\\')[-1])

