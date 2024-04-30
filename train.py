import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn

from dataLoader import YoloDataset, yolo_dataset_collate
from nets.yolo_train import weight_init, set_optimizer_lr, get_lr_scheduler, YOLOLoss
from nets.yolonet import YoloNet
from utils.utils import get_classes, get_anchors
from utils.utils_fit import fit_one_epoch

pretrained = False  # 是否进行预训练

"""
    1、设置学习率的优化公式
    2、关注冻结和解冻训练
"""




# 检查是否有CUDA支持的GPU可用，如果有，则使用第一个GPU；否则，使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_weight_pth = 'model/yolo_weights.pth'
model = YoloNet(pretrained)
model.to(device)

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


train_dataset = YoloDataset("2007_train.txt", isTrain=True)
test_dataset = YoloDataset("2007_test.txt", isTrain=False)



train_sampler = None
val_sampler = None
shuffle = True
batch_size = 4
nw = 0


# 构建dataloader
data_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=nw, pin_memory=True,
                 drop_last=True, collate_fn=yolo_dataset_collate)

data_loader_val = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=nw, pin_memory=True,
                 drop_last=True, collate_fn=yolo_dataset_collate)

# ------------------------------------------------------------------#
#   冻结阶段训练参数
#   此时模型的主干被冻结了，特征提取网络不发生改变
#   占用的显存较小，仅对网络进行微调
#   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
#                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
#                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
#                       （断点续练时使用）
#   Freeze_Epoch        模型冻结训练的Freeze_Epoch
#                       (当Freeze_Train=False时失效)
#   Freeze_batch_size   模型冻结训练的batch_size
#                       (当Freeze_Train=False时失效)
# ------------------------------------------------------------------#
Init_Epoch = 0
Freeze_Epoch = 50
Freeze_batch_size = 16

# ------------------------------------------------------------------#
#   解冻阶段训练参数
#   此时模型的主干不被冻结了，特征提取网络会发生改变
#   占用的显存较大，网络所有的参数都会发生改变
#   UnFreeze_Epoch          模型总共训练的epoch
#                           SGD需要更长的时间收敛，因此设置较大的UnFreeze_Epoch
#                           Adam可以使用相对较小的UnFreeze_Epoch
#   Unfreeze_batch_size     模型在解冻后的batch_size
# ------------------------------------------------------------------#
UnFreeze_Epoch = 300
Unfreeze_batch_size = 8


optimizer_type = 'adam'
Init_lr = 1e-2
Min_lr = Init_lr * 0.01
nbs = 64
momentum = 0.937
weight_decay = 5e-4


input_shape = [416, 416]
anchors = torch.asarray([[10.,  13.], [16.,  30.], [33.,  23.], [30.,  61.], [62.,  45.], [59., 119.], [116.,  90.], [156., 198.], [373., 326.]])


lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)



#---------------------------------------#
#   根据optimizer_type选择优化器
#---------------------------------------#
pg0, pg1, pg2 = [], [], []
for k, v in model.named_modules():
    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)
    if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        pg0.append(v.weight)
    elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)
optimizer = {
    'adam'  : optim.Adam(params=pg0, lr=Init_lr_fit, betas = (momentum, 0.999)),
    'sgd'   : optim.SGD(params=pg0, lr=Init_lr_fit, momentum = momentum, nesterov=True)
}[optimizer_type]
optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
optimizer.add_param_group({"params": pg2}) # 設置了3組optimizer ,每一组对应优化的参数对象不相同

lr_decay_type = "cos"
#   获得学习率下降的公式
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

# model_train = torch.nn.DataParallel(model)
# cudnn.benchmark = True
# model_train = model_train.cuda()

# ---------------------------#
#   读取数据集对应的txt
# ---------------------------#
with open("2007_train.txt") as f:
    train_lines = f.readlines()
with open("2007_test.txt") as f:
    val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)

classes, classes_len = get_classes('dataset/voc_classes.txt')
anchors, anchors_len = get_anchors('dataset/yolo_anchors.txt')
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


loss_history = None # LossHistory(log_dir, model, input_shape=input_shape)
yolo_loss = YOLOLoss(anchors, classes_len, input_shape, device, anchors_mask)
eval_callback = None
epoch_step = num_train // batch_size
epoch_step_val = num_val // batch_size

# ------------------------------------------------------------------#
#   save_period     多少个epoch保存一次权值
# ------------------------------------------------------------------#
save_period = 10

# ------------------------------------------------------------------#
#   save_dir        权值与日志文件保存的文件夹
# ------------------------------------------------------------------#
save_dir = 'logs'
local_rank = 0

Cuda = True

# 开始模型训练
for epoch in range(Init_Epoch, UnFreeze_Epoch):
    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

    print("model is in ", next(model.parameters()).device)
    fit_one_epoch(model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, data_loader, data_loader_val, UnFreeze_Epoch, Cuda, False, None, save_period, save_dir, device, local_rank)

# if local_rank == 0:
#     loss_history.writer.close()
