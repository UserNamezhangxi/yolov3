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
        # l����ǰ�ڼ���������
        # input ��������������� shape Ϊ��
        # bs, 3*(5+num_classes), 13 ,13
        # bs, 3*(5+num_classes), 26 ,26
        # bs, 3*(5+num_classes), 52 ,52
        # targets ���������ʵ��
        bs = input.shape[0]
        in_h = input.shape[2]
        in_w = input.shape[3]

        # ���㲽��
        # ÿһ������ͼ�ϵĵ��Ӧ��ԭͼ�϶��ٸ����ص�
        # �����13 * 13 ������ͼ�Ļ���һ��������Ͷ�Ӧԭ��ͼ�ϵ�32 �����ص�
        # �����26 * 26 ������ͼ�Ļ���һ��������Ͷ�Ӧԭ��ͼ�ϵ�16 �����ص�
        # �����52 * 52 ������ͼ�Ļ���һ��������Ͷ�Ӧԭ��ͼ�ϵ�8 �����ص�
        # stride_h = stride_w = 32, 16, 8
        stride_h = self.input_shape[0]
        stride_w = self.input_shape[1]

        # ��ʱ��õ�scaled����anchor ��С�����������ͼ��
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # �������������ͼ�����ǽ���review
        # bs, 3*(5 + number_classes), 52 ,52
        print("input shape", input.shape)


        # ������תΪ���յ����shape
        # bs, 3*(5 + number_classes), 52 ,52  --->  bs , 3 , 13, 13 ,5+20
        prediction = input.view(bs, len(self.anchors_mask[l]), 5 + self.number_classes, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # ��������ĵ���������
        x = torch.sigmoid(prediction[..., 0])  # ȡ���һ��ά�ȵ� �� 0 ��� ����
        y = torch.sigmoid(prediction[..., 1])  # ȡ���һ��ά�ȵ� �� 1 ��� ����

        # �����Ŀ�ߵ�������
        h = prediction[..., 2]  # ȡ���һ��ά�ȵ� �� 2 ��� ����
        w = prediction[..., 3]  # ȡ���һ��ά�ȵ� �� 3 ��� ����
        conf = prediction[..., 4]   # ȡ���һ��ά�ȵ� �� 4 ��� ����

        # ��ȡ�����������Ŷ�
        pred_cls = torch.sigmoid(prediction[..., 5:])
        self.get_target(l, targets, scaled_anchors, in_h, in_w)


        pass

    def get_target(self, l, targets, scaled_anchors, in_h, in_w):
        # ��ʵ���ݵ�һ��batch ���ж�����ͼ
        bs = len(targets)

        # ����ѡȡ��Щ����򲻰�������
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)

        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)

        # batch_size, 3, 13, 13, 5 + num_classes
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, 25, requires_grad=False)

        for b in range(bs):
            if targets[b] == 0: # ��������
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
