import numpy as np
import torch.nn.init
from torch import nn
import math
from functools import partial


class YOLOLoss(nn.Module):
    def __init__(self, anchors, number_classes, input_shape, device, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.number_classes = number_classes
        self.bbox_attrs = 5 + number_classes
        self.input_shape = input_shape
        self.device = device
        self.anchors_mask = anchors_mask
        self.ignore_threshold = 0.5
        # self.loss_fn = nn.BCELoss()
        # self.loss_fn.to(device)

        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (number_classes / 80)
        self.balance = [0.4, 1.0, 4]

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
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        # 此时获得的scaled——anchor 大小是相对于特征图的
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # 对于输入的特征图对他们进行review
        # bs, 3*(5 + number_classes), 52 ,52
        # print("input shape", input.shape)

        # 将输入转为最终的输出shape
        # bs, 3*(5 + number_classes), 52 ,52  --->  bs , 3 , 13, 13 ,5+20
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # 先验框中心调整至参数
        x = torch.sigmoid(prediction[..., 0])  # 取最后一个维度的 第 0 轴的 数据
        y = torch.sigmoid(prediction[..., 1])  # 取最后一个维度的 第 1 轴的 数据

        # 先验框的宽高调整参数
        h = prediction[..., 2]  # 取最后一个维度的 第 2 轴的 数据
        w = prediction[..., 3]  # 取最后一个维度的 第 3 轴的 数据
        conf = torch.sigmoid(prediction[..., 4])  # 取最后一个维度的 第 4 轴的 数据 置信度

        # 获取分类结果的置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 获取网络应有的预测结果
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)

        # ---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        # ----------------------------------------------------------------#
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        # if cuda:
        y_true = y_true.type_as(x)
        noobj_mask = noobj_mask.type_as(x)
        box_loss_scale = box_loss_scale.type_as(x)

        # --------------------------------------------------------------------------#
        #   box_loss_scale是真实框宽高的乘积，宽高均在0-1之间，因此乘积也在0-1之间。
        #   2-宽高的乘积代表真实框越大，比重越小，小框的比重更大。
        # --------------------------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale

        loss = 0
        # 真实框存在目标的地方 标记为True
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)
        if n != 0:
            # if self.giou:
            # ---------------------------------------------------------------#
            #   计算预测结果和真实结果的giou
            # ----------------------------------------------------------------#
            giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc = torch.mean((1 - giou)[obj_mask])  # 只计算有目标的位置和预测值之间的损失

            # loss_cls = self.loss_fn(pred_cls[obj_mask], y_true[..., 5:][obj_mask])
            loss_cls = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

        # loss_conf = self.loss_fn(conf, obj_mask)
        loss_conf = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[l] * self.obj_ratio

        return loss

    def get_target(self, l, targets, scaled_anchors, in_h, in_w):
        # 真实数据的一个batch 共有多少张图
        bs = len(targets)

        # 用于选取哪些先验框不包含物体
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)

        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)

        # batch_size, 3, 13, 13, 5 + num_classes
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)

        for b in range(bs):
            if len(targets[b]) == 0:  # 没有真实框，忽略
                continue

            batch_target = torch.zeros_like(targets[b])

            # 将 tensor target * 输入特征图的高宽，获得真实框在特征图上的中心点坐标值和宽高值
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            # 将真实框和anchors 的框框左上角重合计算iou，看是属于哪一个范围内的框框（例如计算出来的可能是13*13里的大物体或者26*26中等物体52*52 小物体）
            # torch.zeros(batch_target.size(0), 2)  一个图中有多少个真实框
            # batch_target[:, 2:4] 拿到真实框的右下角坐标
            #  num_true_box, 4  将中心点固定在0,0 组成 （0,0,高,宽）样子的坐标
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), dim=1))

            # 将中心点固定在0,0 组成 （0，0，高，宽）样子的坐标
            anchor_boxes = torch.FloatTensor(torch.cat((torch.zeros((len(scaled_anchors), 2)), torch.FloatTensor(scaled_anchors)),  dim=1))
            # 这样以来就能固定中心点，计算真实框 和 每一个anchors 的 iou 从而得到当前真实框是属于哪一个类型的anchors,比如大目标的框、中等目标的框、小目标的框
            # 这里就是在计算 当前真实框是属于哪个目标范围框框内的的最大的iou
            # 比如图片内那只狗，它计算出来可能就用最大的竖向的框去框它，
            # 再或者street.jpg 中有一个自行车，那么就用中等框横向的框这个自行车
            # shape = [num_true_box , 9] 代表每一个真实框和先验框的重合情况
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

                noobj_mask[b, k, j, i] = 0

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
                #   大目标loss权重小，小目标loss权重大     这里 用目标值的 w,h 相互× ，然后再除去 这个网络的网格宽高，就得到一个当前网格宽高相对于特征层分辨率（13*13 ，26*26 ，52 *52）的比值，
                #   又因为 大目标选择的是 13* 13 的 小目标选择是 52*52 的，所以这样算下来会得到一个目标相对于当前层的宽高 的比值，所以大目标的比值大， 小目标的比值小， 用 1 - 这个值就得到
                #   小目标的损失大，大目标的损失小
                # ----------------------------------------#
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        # print("in_h={},in_w={}".format(in_h, in_w))
        # print(l)
        # print(self.anchors_mask[l])
        bs = len(targets)

        # start(float) - 区间的起始点
        # end (float) - 区间的终点
        # steps(int）- 在start 和 end间生成的样本数
        # 这里就是在构造网格,x 方向从 0....12  y 方向从 0...12
        grid_x = (torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1)
                  .repeat(int(bs * len(self.anchors_mask[l])), 1, 1)
                  .view(x.shape)
                  .type_as(x))
        grid_y = (torch.linspace(0, in_h - 1, in_h)
                  .repeat(in_w, 1)
                  .t()
                  .repeat(int(bs * len(self.anchors_mask[l])), 1, 1)
                  .view(y.shape)
                  .type_as(x))

        # 生成对应层l 的 先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(dim=1, index=torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(dim=1, index=torch.LongTensor([1])).type_as(x)
        # print("anchor_w", anchor_w)

        # 给网格上的每一个点构造 这个点对应的 先验框的 在特征图上的w,h
        # 针对13*13 的网格 的 先验框 w, h:
        # ((3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875))
        # 13*13 = 169 个 (3.625,4.875,11.65625)
        # 有 batch_size 个图 ，所以 最后是 batch_size * 169 个 网格点，每个点都有与之对应的先验框w,h
        # 下面这段代码就是实现了以上的描述,构造了每一个网格点的w,h
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        pre_box_x = torch.unsqueeze(x + grid_x, -1)
        pre_box_y = torch.unsqueeze(y + grid_y, -1)
        pre_box_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pre_box_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)

        pre_boxes = torch.cat([pre_box_x, pre_box_y, pre_box_w, pre_box_h], -1)

        for b in range(bs):
            # 将预测框按照batch_size 维度一个一个取出来，然后进行view,
            pre_box_for_ignore = pre_boxes[b].view(-1, 4)

            # 将真实框转换为相对于特征层的大小
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
                batch_target = batch_target[:, :4].type_as(x)

                # 计算真实框 和 每一个预测框的 iou
                # (3, 3*13*13)
                anch_ious = self.calculate_iou(batch_target, pre_box_for_ignore)

                # 每个先验框对应真实框的最大重合度
                anchor_iou_max, _ = torch.max(anch_ious, dim=0)
                anchor_iou_max = anchor_iou_max.view(pre_boxes[b].size()[:3])

                # 当前batch_size 的 对应位置的iou 是不是大于 阈值,
                # 如果大于阈值则认为是有目标的，设置为0
                noobj_mask[b][anchor_iou_max > self.ignore_threshold] = 0

        return noobj_mask, pre_boxes

    def calculate_iou(self, _box_a, _box_b):
        # -----------------------------------------------------------#
        #   计算真实框的左上角和右下角
        # -----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        # -----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        # -----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        # -----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        # -----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        # -----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        # -----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        # -----------------------------------------------------------#
        #   计算交的面积
        # -----------------------------------------------------------#
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        # -----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        # -----------------------------------------------------------#
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        # -----------------------------------------------------------#
        #   求IOU
        # -----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def box_giou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # ----------------------------------------------------#
        #   求出预测框左上角右下角
        # ----------------------------------------------------#
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # ----------------------------------------------------#
        #   求出真实框左上角右下角
        # ----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # ----------------------------------------------------#
        #   求真实框和预测框所有的iou
        # ----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        # ----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        # ----------------------------------------------------#
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        # ----------------------------------------------------#
        #   计算对角线距离
        # ----------------------------------------------------#
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area

        return giou

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output


def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
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

    print("epoch {} , set_optimizer_lr {} ".format(epoch, lr))