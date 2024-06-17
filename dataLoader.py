import cv2
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.utils import cvtColor, preprocess_input

class YoloDataset(Dataset):
    def __init__(self, train_path,input_shape, isTrain):
        super(YoloDataset, self).__init__()
        self.train_path = train_path
        self.input_shape = input_shape
        self.isTrain = isTrain

        self.lines = open(train_path, 'r').readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        img, boxes = self.get_item_data(index) # 获取图片和 ground truth
        image = np.transpose(preprocess_input(np.array(img, dtype=np.float32)), (2, 0, 1))
        boxes = np.array(boxes, dtype=np.float32)

        if len(boxes) != 0:
            # ground truth 宽高进行归一化
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]

            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]  # 计算中心点xy
            boxes[:, 0:2] = boxes[:, 0:2] + boxes[:, 2:4] / 2  # 计算 ground truth 的w,h

        return image, boxes

    def get_item_data(self, index, jitter=.3 ,hue=.1, sat=0.7, val=0.4, random=True):
        item = self.lines[index].split()
        image = Image.open(item[0])
        image = cvtColor(image)

        iw, ih = image.size
        w, h = self.input_shape  # 416 * 416

        # 计算缩放后的bnd box
        box = np.array([np.array(list(map(int, box.split(',')))) for box in item[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # 将原始图像进行缩放符合416*416 的输入特征
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # bndbox 进行缩放后其他区域的填充操作 对应的 ground truth 框也需要进行dx dy 的偏移
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw/iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh/ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            return image_data, box

        # 对图像进行随机缩放并且进行长款的扭曲
        new_ar = iw/ih * (self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter))
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw,nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 图像左右翻转
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw/iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh/ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]] # ?? box[:, [0, 2]] = w - box[:, [0, 2]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h

            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]

            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box


    def rand(self, a=0,b=1):
        return np.random.rand() * (b - a) + a

def yolo_dataset_collate(batch):
    # 方法1
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes

    # 方法2
    # images, targets = tuple(zip(*batch))
    # images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
    # return images, bboxes

if __name__ == "__main__":
    dataset = YoloDataset('2007_train.txt', (416, 416), True)
    image, box = dataset[0]
    image_data = image.reshape(-1, image.shape[0], image.shape[1], image.shape[2])
    writer = SummaryWriter(log_dir='logs')
    writer.add_images("img", image_data, 1, dataformats="NCHW")
    writer.close()

    print(image.shape)
    print(image)
    print(box)
