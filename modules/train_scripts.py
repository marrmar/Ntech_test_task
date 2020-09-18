import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def fit_epoch(model, train_loader, criterion, optimizer):
    """
    :param model: модель для тренировки
    :param train_loader: генератор батчей
    :param criterion: функция потерь
    :param optimizer: оптимизатор
    :return: лосс и accuracy на этой эпохе на train_loader
    """
    cur_loss = 0.0
    cur_right = 0
    processed_data = 0

    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = np.squeeze(model(x))
        loss = criterion(outputs, y.float())
        loss.backward()
        optimizer.step()

        pred = (outputs > 0.).int()
        cur_loss += loss.item() * x.size(0)
        cur_right += torch.sum(pred == y.data.int())
        processed_data += x.size(0)

    train_loss = cur_loss / processed_data
    train_acc = cur_right.numpy() / processed_data
    return train_loss, train_acc


def eval_epoch(model, val_loader, criterion):
    """
    :param model: модель для валидации
    :param val_loader: генератор батчей
    :param criterion: функция потерь
    :return: лосс и accuracy на этой эпохе на val_loader
    """
    model.eval()
    cur_loss = 0.0
    cur_right = 0
    processed_size = 0

    for x, y in tqdm(val_loader):
        with torch.set_grad_enabled(False):
            outputs = np.squeeze(model(x))
            loss = criterion(outputs, y.float())
            pred = (outputs > 0.).int()


        cur_loss += loss.item() * x.size(0)
        cur_right += torch.sum(pred == y.data.int())
        processed_size += x.size(0)
    val_loss = cur_loss / processed_size
    val_acc = cur_right.numpy() / processed_size
    return val_loss, val_acc


def train(train_dataset, val_dataset, model, epochs, batch_size, writer, lr=0.001, drop_after_epoch=[],
          checkpoint_after=1, checkpoints_folder='', start_from_checkpoint=False, checkpoint_path='',
          weights_only=False, ):
    """
    :param train_dataset: -
    :param val_dataset: -
    :param model: модель для тренировки
    :param epochs: количество эпох
    :param batch_size: -
    :param writer: для отслеживания прогерсса через tensorboard
    :param lr: -
    :param drop_after_epoch: номера эпох для скедулера, когда нужно дропнуть lr
    :param checkpoint_after: через сколько эпох сохранять сеть
    :param checkpoints_folder: куда сохранять чекпойнты
    :param start_from_checkpoint: начинаем с чекпойнта или с начала
    :param checkpoint_path: путь до чекпойнта
    :param weights_only: при start_from_checkpoint=True, True, если хотим загрузить только веса и поставить оптимизатор и скедулер свои, False иначе
    :return: итоговые лосс и accuracy на трейне и вале
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    current_epoch = 0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=drop_after_epoch)
    criterion = nn.BCEWithLogitsLoss()

    if start_from_checkpoint:
        checkpoint = torch.load(checkpoint_path)
        if weights_only:
            model.load_state_dict(checkpoint['state_dict'])
            current_epoch = checkpoint['current_epoch']
        else:
            model.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            current_epoch = checkpoint['current_epoch']

    for epoch in range(current_epoch + 1, epochs + current_epoch + 1):
        train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt)

        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                       v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))

        if epoch % checkpoint_after == 0:
            snapshot_name = '{}/checkpoint_epoch_{}.pth'.format(checkpoints_folder, epoch)
            torch.save({'state_dict': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'current_epoch': epoch},
                       snapshot_name)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    return train_loss, train_acc, val_loss, val_acc
