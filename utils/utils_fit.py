import os

import torch
from tqdm import tqdm
from utils.utils import get_lr



def fit_one_epoch(model, yolo_loss, loss_history, eval_callback,
                  optimizer, epoch, epoch_step,
                  epoch_step_val, data_loader, data_loader_val,
                  Epoch, save_period, save_dir, device, writer, local_rank=0):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model.train()

    for iteration, batch in enumerate(data_loader):
        print("iteration", iteration)
        if iteration >= epoch_step:
            break

        images, targets = batch

        with torch.no_grad():
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]

        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()

        # ----------------------#
        #   前向传播
        # ----------------------#
        outputs = model(images)

        loss_value_all = 0
        # ----------------------#
        #   计算损失
        # ----------------------#
        for l in range(len(outputs)):
            loss_item = yolo_loss(l, outputs[l], targets)
            loss_value_all += loss_item
        loss_value = loss_value_all

        #----------------------#
        #   反向传播
        #----------------------#
        loss_value.backward()

        # ----------------------#
        #   更新学习率
        # ----------------------#
        optimizer.step()

        # 一个batch 内的损失求和
        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1), 'lr': get_lr(optimizer)})
            pbar.update(1)

    writer.add_scalar("train_loss", loss, epoch)


    # 一个batch 数据走完前向传播和反向传播更新了权重，然后对验证集进行验证
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model.eval()

    for iteration, batch in enumerate(data_loader_val):
        print("val iteration", iteration)
        if iteration >= epoch_step_val:
            # print("测试集的迭代次数大于设定的测试集步长！")
            break

        images, targets = batch

        with torch.no_grad():
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]

            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model(images)

            loss_value_all = 0

            #---------------------------#
            # 计算loss
            # ---------------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item

            loss_value = loss_value_all

        val_loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    writer.add_scalar("test_loss", val_loss, epoch)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        # loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        # eval_callback.on_epoch_end(epoch + 1, model)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

    # -----------------------------------------------#
    #   保存权值
    # -----------------------------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
        epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

    # if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
    #     print('Save best model to best_epoch_weights.pth')
    #     torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

