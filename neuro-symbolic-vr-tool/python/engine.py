import math
import sys
import torch
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        try:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not math.isfinite(losses.item()):
                print(f"⚠️ Non-finite loss detected: {losses.item()}, skipping update.")
                continue

            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            metric_logger.update(loss=losses_reduced.item(), **{k: v.item() for k, v in loss_dict_reduced.items()})
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        except AssertionError as e:
            print(f"⚠️ Skipping batch due to invalid box: {e}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error during training batch: {e}")
            continue


def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type=device.type):
                model(images, targets)

    torch.set_num_threads(n_threads)
