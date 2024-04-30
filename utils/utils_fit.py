import torch
from tqdm import tqdm


def fit_one_epoch(model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, data_loader, data_loader_val, Epoch, cuda, fp16, scaler, save_period, save_dir, device, local_rank=0):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model.train()

    for iteration, batch in enumerate(data_loader): #
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]

        with torch.no_grad():
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]

        optimizer.zero_grad()
        outputs = model(images)

        loss_value_all = 0

        for l in range(len(outputs)):
            loss_item = yolo_loss(l, outputs[l], targets)
            loss_value_all += loss_item
        loss_value = loss_value_all
        loss_value.backward()
        optimizer.step()