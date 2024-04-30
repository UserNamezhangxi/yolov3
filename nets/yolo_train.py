import torch.nn.init
from torch import nn
import math
from functools import partial


class YOLOLoss(nn.Module):
    def __init__(self, anchors, number_classes, input_shape, device, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.input_shape = input_shape
        self.number_classes = number_classes
        self.device = device
        self.anchors_mask = anchors_mask

    def forward(self, l, input, targets=None):
        # l：当前第几个特征层
        # input 输入进来的特征层 shape 为：
        # bs, 3*(5+num_classes), 13 ,13
        # bs, 3*(5+num_classes), 26 ,26
        # bs, 3*(5+num_classes), 52 ,52
        # targets 代表的是真实框
        bs = input.shape[0]
        in_h = input.shape[2]
        in_w = input.shape[3]

        # 计算步长
        # 每一个特征图上的点对应到原图上多少个像素点
        # 如果是13 * 13 的特征图的话，一个特征点就对应原来图上的32 个像素点
        # 如果是26 * 26 的特征图的话，一个特征点就对应原来图上的16 个像素点
        # 如果是52 * 52 的特征图的话，一个特征点就对应原来图上的8 个像素点
        # stride_h = stride_w = 32, 16, 8
        stride_h = self.input_shape[0]
        stride_w = self.input_shape[1]

        # 此时获得的scaled——anchor 大小是相对于特征图的
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # 对于输入的特征图对他们进行review
        # bs, 3*(5 + number_classes), 52 ,52
        print("input shape", input.shape)


        # 将输入转为最终的输出shape
        # bs, 3*(5 + number_classes), 52 ,52  --->  bs , 3 , 13, 13 ,5+20
        prediction = input.view(bs, len(self.anchors_mask[l]), 5 + self.number_classes, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # 先验框中心调整至参数
        x = torch.sigmoid(prediction[..., 0])  # 取最后一个维度的 第 0 轴的 数据
        y = torch.sigmoid(prediction[..., 1])  # 取最后一个维度的 第 1 轴的 数据

        # 先验框的宽高调整参数
        h = prediction[..., 2]  # 取最后一个维度的 第 2 轴的 数据
        w = prediction[..., 3]  # 取最后一个维度的 第 3 轴的 数据
        conf = prediction[..., 4]   # 取最后一个维度的 第 4 轴的 数据

        # 获取分类结果的置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])
        self.get_target(l, targets, scaled_anchors, in_h, in_w)


        pass

    def get_target(self, l, targets, scaled_anchors, in_h, in_w):
        # 真实数据的一个batch 共有多少张图
        bs = len(targets)

        # 用于选取哪些先验框不包含物体
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)

        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)

        # batch_size, 3, 13, 13, 5 + num_classes
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, 25, requires_grad=False)

        for b in range(bs):
            if targets[b] == 0: # 背景忽略
                continue

            batch_target = torch.zeros_like(targets[b])

            batch_target[:, [0, 2]] = batch_target[b][:, [0, 2]] * in_h
            batch_target[:, [1, 3]] = batch_target[b][:, [1, 3]] * in_w
            batch_target[:, 4] = batch_target[b][:, 4]
            batch_target = batch_target.to(self.device)







def weight_init(net, init_gain=0.02):
    def init_func(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0, init_gain)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, std=init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
