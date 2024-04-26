import numpy as np
import torch

from nets.yolo_train import weight_init
from nets.yolonet import YoloNet

pretrained = False  # 是否进行预训练

# 检查是否有CUDA支持的GPU可用，如果有，则使用第一个GPU；否则，使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_weight_pth = 'model/yolo_weights.pth'
model = YoloNet(pretrained)

if not pretrained:
    # 加载yolo的预训练模型参数
    weight_init(model)

# ------------------------------------------------------#
#   根据预训练权重的Key和模型的Key进行加载
# ------------------------------------------------------#
model_dict = model.state_dict()

# 将训练好的yolo网络权重加载进来， 和自己的yolo网络进行比对，加载已有的权重参数到自己的网络中来
# 这就是迁移学习！！！
if yolo_weight_pth != '':
    pretrained_dict = torch.load(yolo_weight_pth, map_location=device)

    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
            print("load {}  shape {}".format(k, np.shape(model_dict[k])))
        else:
            no_load_key.append(k)
            print("no load {} ".format(k))

    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    print("\nSuccessful Load Key:", str(load_key)[:1500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:1500], "……\nFail To Load Key num:", len(no_load_key))
    print("\n温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。")