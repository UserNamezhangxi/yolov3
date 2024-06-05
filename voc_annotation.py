import os
import xml.etree.ElementTree as ET

from utils.utils import get_classes

classes, classes_len = get_classes('dataset/voc_classes.txt')


def convert_annotation(image_id, list_file):
    dataset_path = 'dataset/VOC2007/Annotations'
    lines_data = []
    annotation_path = os.path.join(dataset_path, image_id + ".xml")

    with open(annotation_path) as f:
        tree = ET.parse(f)
        root = tree.getroot()

        for obj in root.iter("object"):
            difficult = 0
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
            cls = obj.find("name").text

            if cls not in classes or int(difficult) == 1: # difficult 标签为1 可以不要，为了减小计算量
                continue

            cls_id = str(classes.index(cls))
            lines_data.append(annotation_path)
            boxes = obj.find("bndbox")
            x_min = boxes.find("xmin").text
            y_min = boxes.find("ymin").text
            x_max = boxes.find("xmax").text
            y_max = boxes.find("ymax").text
            list_file.write(' ' + x_min + "," + y_min + "," + x_max + "," + y_max + "," + cls_id)

if __name__ == "__main__":
    image_ids = open('dataset/VOC2007/ImageSets/Main/train.txt').read().strip().split()
    list_file = open('2007_train.txt', 'w')
    for img_id in image_ids:
        list_file.write(os.path.abspath(os.path.join('dataset/VOC2007/JPEGImages', img_id + '.jpg')))
        convert_annotation(img_id, list_file)
        list_file.write('\n')
    list_file.close()

    image_ids = open('dataset/VOC2007/ImageSets/Main/test.txt').read().strip().split()
    list_file = open('2007_test.txt', 'w')
    for img_id in image_ids:
        list_file.write(os.path.abspath(os.path.join('dataset/VOC2007/JPEGImages', img_id + '.jpg')))
        convert_annotation(img_id, list_file)
        list_file.write('\n')
    list_file.close()
