import os

from PIL import Image
from tqdm import tqdm
from Yolo import Yolo
import xml.etree.ElementTree as ET
from utils.utils import get_classes
from utils.utils_map import get_map, get_coco_map

# 创建对应目录
"""
1 ground-truth 保存每一个图片的groundTruth
2 detection-results 保存网络预测后的每一个图片的结果
3 images-optional 可视化
"""
map_out_path = 'map_out'
if not os.path.exists(os.path.join(map_out_path, "./ground-truth")):
    os.makedirs(os.path.join(map_out_path, "./ground-truth"))
if not os.path.exists(os.path.join(map_out_path, "./detection-results")):
    os.makedirs(os.path.join(map_out_path, "./detection-results"))
if not os.path.exists(os.path.join(map_out_path, "./images-optional")):
    os.makedirs(os.path.join(map_out_path, "./images-optional"))

class_names, _ = get_classes('./dataset/voc_classes.txt')

# 1、获取真实框，并写入文件
image_ids = open('./dataset/VOC2007/ImageSets/Main/test.txt').read().strip().split()

score_threhold = 0.5
confidence = 0.05
nms_iou = 0.5
map_vis = False
MINOVERLAP = 0.5
mode = 1

if mode == 0:
    print("Load model.")
    yolo = Yolo(confidence = confidence, nms_iou = nms_iou)
    print("Load model done.")

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path = os.path.join("./dataset/VOC2007/JPEGImages/"+image_id+".jpg")
        image = Image.open(image_path)
        if map_vis:
            image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
        yolo.get_map_txt(image_id, image, map_out_path)
    print("Get predict result done.")

    print("Get ground truth result.")
    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join("./dataset/VOC2007/Annotations/"+image_id+".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult') is not None:
                    difficult = obj.find('difficult').text
                    if int(difficult) == 1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")

    # 2、获取网络预测框，并写入文件
    get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)

if mode == 1:
    # 3 通过 pycocotools 计算mAP 需要 pip3 install pycocotools -i  https://pypi.tuna.tsinghua.edu.cn/simple
    print("Get map.")
    get_coco_map(class_names, map_out_path)
    print("Get map done.")