from modules.dataset import Male_Dataset
from modules.train_scripts import train
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import argparse
from modules.net import DenseNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='path project')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--resize_size', type=int, default=112, help='net input image size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--start_from_checkpoint', type=bool, default=False, help='from checkpoint or from scratch')
    parser.add_argument('--weights_only', type=bool, default=False, help='if start_from_checkpoint use saved optimizer and scheduler or not')
    parser.add_argument('--drop_after_epoch', type=list, default=[], help='when drop lr')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoints path')
    args = parser.parse_args()

    path = args.path
    checkpoint_path = args.checkpoint_path
    epochs = args.epochs
    batch_size = args.batch_size
    resize_size = args.resize_size
    start_from_checkpoint = args.start_from_checkpoint
    lr = args.lr
    weights_only = args.weights_only
    drop_after_epoch = args.drop_after_epoch

    checkpoint_folder = path+'\\checkpoints'
    train_path = path + '\\train_files.txt'
    val_path = path + '\\val_files.txt'

    writer = SummaryWriter()
    train_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.Normalize([0.3767, 0.4692, 0.6215], [0.2218, 0.2273, 0.2501])]) # средние и дисперсии по датасету
    val_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize([0.3767, 0.4692, 0.6215], [0.2218, 0.2273, 0.2501])]) # средние и дисперсии по трейн датасету

    train_dataset = Male_Dataset(train_path, transforms=train_transforms, mode='train', resize_size=resize_size)
    val_dataset = Male_Dataset(val_path, transforms=val_transforms, mode='val', resize_size=resize_size)


    model = DenseNet(38, 1)
    train_loss, train_acc, val_loss, val_acc = train(train_dataset, val_dataset, model=model, epochs=epochs, batch_size=batch_size, writer=writer,
                                                     lr=lr, drop_after_epoch=drop_after_epoch,
                                                     checkpoints_folder=checkpoint_folder,
                                                     start_from_checkpoint=start_from_checkpoint,
                                                     checkpoint_path=checkpoint_path,  weights_only=weights_only,)

    print("\nTOTAL: train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}".format(
        ep=epochs + 1, t_loss=train_loss,
        v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))

