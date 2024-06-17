import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataLoader import YoloDataset, yolo_dataset_collate
from nets.yolo_train import weights_init, set_optimizer_lr, get_lr_scheduler, YOLOLoss
from nets.yolonet import YoloNet
from utils.utils import get_classes, get_anchors
from utils.utils_fit import fit_one_epoch

# ----------------------------------------------------------------------------------------------------------------------------#
#   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
#                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
#                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
#                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
# ----------------------------------------------------------------------------------------------------------------------------#
pretrained = False  # 是否进行预训练

"""
    1、设置学习率的优化公式
    2、关注冻结和解冻训练
"""

train_sampler = None
val_sampler = None
shuffle = True
batch_size = 16
nw = 0

# 检查是否有CUDA支持的GPU可用，如果有，则使用第一个GPU；否则，使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_rank = 0

yolo_weight_pth = 'model/yolo_weights.pth'
classes, classes_len = get_classes('dataset/voc_classes.txt')
anchors, anchors_len = get_anchors('dataset/yolo_anchors.txt')
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
input_shape = [416, 416]
model = YoloNet(anchors_mask, classes_len, pretrained=pretrained)
model.to(device)

if not pretrained:
    # 加载yolo的预训练模型参数
    weights_init(model)

# ------------------------------------------------------#
#   保存当前模型的权重
# ------------------------------------------------------#
model_dict = model.state_dict()

# 将训练好的yolo网络权重加载进来， 和自己的yolo网络进行比对，加载已有的权重参数到自己的网络中来
# 这就是迁移学习！！！
if yolo_weight_pth != '':
    # 从外部文件加载的预训练模型的权重
    pretrained_dict = torch.load(yolo_weight_pth, map_location=device)

    load_key, no_load_key, temp_dict = [], [], {}

    # 循环遍历预训练权重字典：代码通过一个循环遍历 `pretrained_dict` 中的键-值对，其中键 `k` 是预训练权重的名称，值 `v` 是相应的权重张量。
    for k, v in pretrained_dict.items():
        # 比较权重形状：对于每个键 `k`，代码检查它是否存在于当前模型的权重字典 `model_dict` 中，
        # 并且检查权重的形状是否与当前模型中的权重形状相匹配。如果键 `k` 存在于 `model_dict` 中且形状匹配，
        # 那么将这个权重添加到临时字典 `temp_dict` 中，并将键 `k` 添加到 `load_key` 列表中，表示这个权重需要加载。
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            # 更新当前模型的权重字典：将临时字典 `temp_dict` 中的权重添加到当前模型的权重字典 `model_dict` 中，
            # 以确保只有匹配的权重被加载，不匹配的权重不会破坏当前模型的结构
            temp_dict[k] = v
            load_key.append(k)
            print("load {}  shape {}".format(k, np.shape(model_dict[k])))
        else:
            no_load_key.append(k)
            print("no load {} ".format(k))
    # 加载更新后的权重：最后，使用 `model.load_state_dict(model_dict)` 将更新后的权重加载到当前模型中，
    # 从而将预训练的权重应用到当前模型中
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    print("\nSuccessful Load Key:", str(load_key)[:1500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:1500], "……\nFail To Load Key num:", len(no_load_key))
    print("\n温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。")


train_dataset = YoloDataset("2007_train.txt", input_shape, isTrain=True)
test_dataset = YoloDataset("2007_test.txt", input_shape, isTrain=False)


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

optimizer_type = 'sgd'
Init_lr = 1e-3 if optimizer_type == 'adam' else 1e-2
Min_lr = Init_lr * 0.01
momentum = 0.937
weight_decay = 0 if optimizer_type == 'adam' else 5e-4

input_shape = [416, 416]

# ---------------------------#
#   读取数据集对应的txt
# ---------------------------#
with open("2007_train.txt") as f:
    train_lines = f.readlines()
with open("2007_test.txt") as f:
    val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)


loss_history = None # LossHistory(log_dir, model, input_shape=input_shape)
yolo_loss = YOLOLoss(anchors, classes_len, input_shape, device, anchors_mask)
yolo_loss.to(device)
eval_callback = None

# ------------------------------------------------------------------#
#   save_period     多少个epoch保存一次权值
# ------------------------------------------------------------------#
save_period = 5

# ------------------------------------------------------------------#
#   save_dir        权值与日志文件保存的文件夹
# ------------------------------------------------------------------#
save_dir = './logs/'
local_rank = 0

# ------------------------------------------------------------------#
#   Freeze_Train    是否进行冻结训练
#                   默认先冻结主干训练后解冻训练。
# ------------------------------------------------------------------#
Freeze_Train = True

UnFreeze_flag = False

#------------------------------------#
#   冻结一定部分训练
#------------------------------------#
if Freeze_Train:
    for param in model.backbone.parameters():
        param.requires_grad = False


#-------------------------------------------------------------------#
#   设置batch_size
#   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
#-------------------------------------------------------------------#
batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size


#-------------------------------------------------------------------#
#   判断当前batch_size，自适应调整学习率
#-------------------------------------------------------------------#
nbs = 64
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

#---------------------------------------#
#   判断每一个epoch的长度
#---------------------------------------#
epoch_step = num_train // batch_size
epoch_step_val = num_val // batch_size

# 添加tensorboard
writer = SummaryWriter(log_dir='./tensorboard_logs')

# 开始模型训练
for epoch in range(Init_Epoch, UnFreeze_Epoch):

    # 如果模型有冻结学习部分，则解冻，并设置新的解冻后的参数
    if epoch > Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
        batch_size = Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # 主干特征网络也参与训练
        for param in model.backbone.parameters():
            param.requires_grad = True

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        UnFreeze_flag = True

    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

    print("model is in ", next(model.parameters()).device)
    fit_one_epoch(model, yolo_loss, loss_history, eval_callback,
                  optimizer, epoch, epoch_step,
                  epoch_step_val, data_loader, data_loader_val,
                  UnFreeze_Epoch, save_period, save_dir, device, writer, local_rank)
torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


# if local_rank == 0:
#     loss_history.writer.close()
