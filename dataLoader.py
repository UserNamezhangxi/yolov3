from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def preprocess_input(img):
    img = img / 255.0
    return img

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


class YoloDataset(Dataset):
    def __init__(self, train_path, isTrain):
        super(YoloDataset, self).__init__()
        self.train_path = train_path
        self.input_shape = (416, 416)
        self.isTrain = isTrain

        self.lines = open(train_path, 'r').readlines()

    def __len__(self):
        len(self.lines)

    def __getitem__(self, index):
        img, boxes = self.get_item_data(index)
        image = np.transpose(preprocess_input(np.array(img, dtype=np.float32)), (2, 0, 1))
        return image, boxes

    def get_item_data(self, index):
        item = self.lines[index].split()
        image = Image.open(item[0])
        image = cvtColor(image)

        iw, ih = image.size
        w, h = self.input_shape  # 416 * 416

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        # print("缩放比".format(scale))
        # print("原始图的寬高={},{}".format(iw,ih))
        # print("缩放后寬高={},{}".format(nw,nh))

        # 将原始图像进行缩放符合416*416 的输入特征
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (0, 0))
        image_data = np.array(new_image, np.float32)

        # 计算缩放后的bnd box
        box = np.array([np.array(list(map(int, box.split(',')))) for box in item[1:]], dtype=np.float32)
        box[:, :4] *= scale

        # 图形归一化
        # image_data = image_data / 255.0
        # bndbox 归一化
        box[:, [0, 2]] = box[:, [0, 2]] / nw
        box[:, [1, 3]] = box[:, [1, 3]] / nh
        # print("缩放后的box={}{},{}{}".format(box[0,0], box[0,1],box[0,2], box[0,3]))

        # x_min = box[0,0] * nw / scale
        # y_min = box[0,1] * nh / scale
        # x_max = box[0,2] * nw / scale
        # y_max = box[0,3] * nh / scale
        # print("计算原图的box {} {},{} {}".format(x_min,y_min,x_max,y_max))

        return image_data, box


if __name__ == "__main__":
    dataset = YoloDataset('2007_train.txt', True)
    image, box = dataset[0]
    image_data = image.reshape(-1, image.shape[0], image.shape[1], image.shape[2])
    writer = SummaryWriter(log_dir='logs')
    writer.add_images("img", image_data, 1, dataformats="NCHW")
    writer.close()

    print(image.shape)
    print(image)
    print(box)
