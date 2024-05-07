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

        # 获取网络应有的预测结果
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)


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

            # 将 tensor target * 输入特征图的高宽，获得真实框在特征图上的中心点坐标值和宽高值
            batch_target[:, [0, 2]] = batch_target[b][:, [0, 2]] * in_h
            batch_target[:, [1, 3]] = batch_target[b][:, [1, 3]] * in_w
            batch_target[:, 4] = batch_target[b][:, 4]
            batch_target = batch_target.to(self.device)

            # 将真实框和anchors 的框框左上角重合计算iou，看是属于哪一个范围内的框框（例如计算出来的可能是13*13里的大物体或者26*26中等物体52*52 小物体）
            # torch.zeros(batch_target.size(0), 2)  一个图中有多少个真实框
            # batch_target[:, 2:4] 拿到真实框的右下角坐标
            #  num_true_box, 4  将中心点固定在0,0 组成 （0,0,高,宽）样子的坐标
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)),  batch_target[:, 2:4]), dim=1))

            # 将中心点固定在0,0 组成 （0，0，高，宽）样子的坐标
            anchor_boxes = torch.cat((torch.zeros(len(scaled_anchors), 2), scaled_anchors))

            # 这样以来就能固定中心点，计算真实框 和 每一个anchors 的 iou 从而得到当前真实框是属于哪一个类型的anchors,比如大目标的框、中等目标的框、小目标的框
            # 这里就是在计算 当前真实框是属于哪个目标范围框框内的的最大的iou
            # 比如图片内那只狗，它计算出来可能就用最大的竖向的框去框它，
            # 再或者street.jpg 中有一个自行车，那么就用中等框横向的框这个自行车
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_boxes), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue

                # 判断当前先验框是当前特征的那一个先验框
                k = self.anchors_mask[l].index(best_n)

                # 获得真实框属于哪个网格点
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()

                # 取出真实框的种类
                c = batch_target[t, 4].long()

                noobj_mask[b, k, i, j] = 0

                # ----------------------------------------#
                #   tx、ty代表中心调整参数的真实值
                # ----------------------------------------#
                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1

                # ----------------------------------------#
                #  TODO 这里是什么含义呢
                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                # ----------------------------------------#
                box_loss_scale[b, k, i, j] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale


    def calculate_iou(self, _box_a, _box_b):
        #-----------------------------------------------------------#
        #   计算真实框的左上角和右下角
        #-----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        #-----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        #-----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        #-----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        #-----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        #-----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        #-----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        #-----------------------------------------------------------#
        #   计算交的面积
        #-----------------------------------------------------------#
        max_xy  = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy  = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter   = torch.clamp((max_xy - min_xy), min=0)
        inter   = inter[:, :, 0] * inter[:, :, 1]
        #-----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        #-----------------------------------------------------------#
        #   求IOU
        #-----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]


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
