import numpy as np
import torch
from tqdm import tqdm
from . import metrics

from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
    """Warmup 学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, base_scheduler):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增加学习率
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Warmup 结束后，使用基础调度器
            return self.base_scheduler.get_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：手动更新学习率
            super().step(epoch)
        else:
            # Warmup 结束后，调用基础调度器的 step 方法
            self.base_scheduler.step(epoch)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_logs(logs):
    str_logs = ["{}={:.3}".format(k, v) for k, v in logs.items()]
    return ", ".join(str_logs)


def metric(input, target):
    """
    Args:
        input (tensor): prediction
        target (tensor): reference data

    Returns:
        float: harmonic fscore without including backgorund
    """
    input = torch.softmax(input, dim=1)
    scores = []

    for i in range(1, input.shape[1]):  # background is not included
        ypr = input[:, i, :, :].view(input.shape[0], -1)
        ygt = target[:, i, :, :].view(target.shape[0], -1)
        scores.append(metrics.iou(ypr, ygt).item())

    return np.mean(scores)


def train_epoch(model, optimizer, criterion, dataloader, device="cpu"):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        criterion (_type_): _description_
        dataloader (_type_): _description_
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()

    iterator = tqdm(dataloader, desc="Train")
    for x, y, *_ in iterator:
        x = x.to(device).float()
        y = y.to(device).float()
        n = x.shape[0]

        optimizer.zero_grad()
        outputs = model.forward(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=n)

        with torch.no_grad():
            score_meter.update(metric(outputs, y), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs


def train_epoch_with_scheduler(
    model,
    optimizer,
    scheduler,
    criterion,
    dataloader,
    device="cpu",
    scheduler_type='epoch',
):
    """带有调度器的训练epoch。支持不同类型的学习率调度器。

    Args:
        model (_type_): 训练模型
        optimizer (_type_): 优化器
        scheduler (_type_): 学习率调度器
        criterion (_type_): 损失函数
        dataloader (_type_): 数据加载器
        device (str, optional): 设备类型 (cpu or cuda)。默认为 "cpu"。
        scheduler_type (str, optional): 调度器类型。'batch' 或 'epoch'。默认为 'epoch'。

    Returns:
        logs (dict): 记录损失和评分的日志字典。
    """
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()  # 设置模型为训练模式

    iterator = tqdm(dataloader, desc="Train")  # 初始化进度条
    for x, y, *_ in iterator:
        x = x.to(device).float()  # 将输入数据移动到指定设备
        y = y.to(device).float()  # 将标签移动到指定设备
        n = x.shape[0]

        optimizer.zero_grad()  # 清空梯度
        outputs = model(x)  # 通过模型进行前向传播
        loss = criterion(outputs, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        loss_meter.update(loss.item(), n=n)  # 更新损失的平均值

        with torch.no_grad():  # 在评估时不需要计算梯度
            score_meter.update(metric(outputs, y), n=n)  # 更新评分

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))  # 更新进度条上的显示

        # 调度器更新
        if scheduler_type == 'batch':
            scheduler.step()  # 对于一些调度器，如 StepLR，更新每个batch
        elif scheduler_type == 'epoch':
            # 对于 ReduceLROnPlateau 等调度器，通常在 epoch 结束时更新
            pass

    if scheduler_type == 'epoch':
        scheduler.step()  # 如果是 ReduceLROnPlateau 类型调度器，通常在epoch结束时调用

    return logs


def train_epoch_with_scheduler_and_warmup(
    model,
    optimizer,
    scheduler,
    criterion,
    dataloader,
    device="cpu",
    scheduler_type='epoch',
    warmup_epochs=3,  # Warmup 的 epoch 数
):
    """带有调度器和 Warmup 的训练 epoch。

    Args:
        model (_type_): 训练模型
        optimizer (_type_): 优化器
        scheduler (_type_): 学习率调度器
        criterion (_type_): 损失函数
        dataloader (_type_): 数据加载器
        epoch_num (int): 训练的 epoch 数
        device (str, optional): 设备类型 (cpu or cuda)。默认为 "cpu"。
        scheduler_type (str, optional): 调度器类型。'batch' 或 'epoch'。默认为 'epoch'。
        warmup_epochs (int, optional): Warmup 的 epoch 数。默认为 5。

    Returns:
        logs (dict): 记录损失和评分的日志字典。
    """
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()  # 设置模型为训练模式

    # 初始化 Warmup 调度器
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, scheduler)

    iterator = tqdm(dataloader, desc="Train")  # 初始化进度条
    for x, y, *_ in iterator:
        x = x.to(device).float()  # 将输入数据移动到指定设备
        y = y.to(device).float()  # 将标签移动到指定设备
        n = x.shape[0]

        optimizer.zero_grad()  # 清空梯度
        outputs = model(x)  # 通过模型进行前向传播
        loss = criterion(outputs, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        loss_meter.update(loss.item(), n=n)  # 更新损失的平均值

        with torch.no_grad():  # 在评估时不需要计算梯度
            score_meter.update(metric(outputs, y), n=n)  # 更新评分

        logs.update({"Loss": loss_meter.avg})
        logs.update({"lr": optimizer.param_groups[0]["lr"]})  # 记录当前学习率
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))  # 更新进度条上的显示

        # 调度器更新
        if scheduler_type == 'batch':
            warmup_scheduler.step()  # 每个 batch 更新一次学习率
        elif scheduler_type == 'epoch':
            # 对于 ReduceLROnPlateau 等调度器，通常在 epoch 结束时更新
            pass

    if scheduler_type == 'epoch':
        warmup_scheduler.step()  # 每个 epoch 更新一次学习率

    return logs


def valid_epoch(model=None, criterion=None, dataloader=None, device="cpu"):
    """_summary_

    Args:
        model (_type_, optional): _description_. Defaults to None.
        criterion (_type_, optional): _description_. Defaults to None.
        dataloader (_type_, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.to(device).eval()

    iterator = tqdm(dataloader, desc="Valid")
    for x, y, *_ in iterator:
        x = x.to(device).float()
        y = y.to(device).float()
        n = x.shape[0]

        with torch.no_grad():
            outputs = model.forward(x)
            loss = criterion(outputs, y)

            loss_meter.update(loss.item(), n=n)
            score_meter.update(metric(outputs, y), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs
