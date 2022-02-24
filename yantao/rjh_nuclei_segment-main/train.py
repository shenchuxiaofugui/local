import os
#os.environ["CUDA_VISIBLE_DEVICES"] = 'MIG-GPU-4ef25685-6248-e9a8-f4bd-86cbf2763f38/1/0'
import sys
sys.path.append('/homes/ydwang/projects')
from Dataset_all_roi import make_data_loaders
from models.network import UNet3D, CA_UNet3D

import torch
from utils.losses import total_loss_6ROI
from torch.utils.tensorboard import SummaryWriter
from utils.early_stopping import EarlyStopping
import tqdm
from torchsummary import summary

def test(loader, model, optimizer, total_dice_dict, total_loss_dict, phase):
    dice, loss = 0, 0
    for (batch_images, batch_rois, batch_infos) in tqdm.tqdm(loader):
        batch_images, batch_rois = batch_images.cuda(), batch_rois.cuda()
        with torch.set_grad_enabled(phase == 'train'):
            output = model(batch_images)
            loss_dict = total_loss_6ROI(output, batch_rois, c=2)

        loss += loss_dict['total_loss'].data.item()
        dice += loss_dict['dice_loss'].data.item()

        if phase == 'train':
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()

    loss_mean = loss / len(loader)
    dice_mean = dice / len(loader)
    total_loss_dict[phase] = loss_mean
    total_dice_dict[phase] = dice_mean

    return loss_mean, dice_mean


def train(epoch, loader, model, optimizer, all_dices, all_losses):
    loss_mean, dice_mean = test(loader, model, optimizer, all_dices, all_losses, 'train')
    print(f'\nTrain: Epoch is : {epoch}, loss is :{loss_mean}, dice is {dice_mean}')


def evaluate(epoch, loader, model, optimizer, all_dices, all_losses):
    loss_mean, dice_mean = test(loader, model, optimizer, all_dices, all_losses, 'eval')
    print(f'\nEvaluate: Epoch is : {epoch}, loss is :{loss_mean}, dice is {dice_mean}')
    return loss_mean

def on_epoch_end(writer, model, optimizer, log_path, epoch, early_stopping, loss_mean, all_losses, all_dices):
    state = dict()
    state['model'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    model_path = os.path.join(log_path, 'model')
    os.makedirs(model_path, exist_ok=True)
    if (epoch + 1) % 10 == 0:
        file_name = os.path.join(model_path, 'epoch' + str(epoch + 1) + '_model.pth')
        torch.save(state, file_name)
    early_stopping(loss_mean, state, epoch, model_path)
    writer.add_scalars('loss', all_losses, epoch)
    writer.add_scalars('dice', all_dices, epoch)

    return early_stopping.early_stop

def train_val(model, loaders, optimizer,  early_stopping,  log_path, n_epochs=300):
    writer = SummaryWriter(os.path.join(log_path, 'log_dir'))

    for epoch in range(n_epochs):
        all_losses, all_dices = {}, {}

        train(epoch, loaders['train'], model, optimizer, all_dices, all_losses)
        loss_mean = evaluate(epoch, loaders['eval'], model, optimizer, all_dices, all_losses)
        if on_epoch_end(writer, model, optimizer, log_path, epoch, early_stopping, loss_mean, all_losses, all_dices):
            print('Early stopping')
            break

    writer.close()
    return model

def main(params):
    train_csv_path = os.path.join(params['index_path'], 'train_index.csv')
    val_csv_path = os.path.join(params['index_path'], 'val_index.csv')
    loaders = make_data_loaders(params['data_root'], train_csv_path, val_csv_path,
                                params['data_modes'], params['roi_modes'], params['input_shape'],
                                batch_size=params['batch_size'])

    network = UNet3D(input_shape=params['input_shape'], in_channels=1, out_channels=2, init_channels=8)
    # network = CA_UNet3D(input_shape=params['input_shape'], in_channels=1, out_channels=7, init_channels=8)
    model = network.cuda()
    summary(model, (1, 320, 320, 32))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    train_val(model, loaders, optimizer, early_stopping, params['log_path'],  n_epochs=500)


if __name__ == "__main__":
    params = {
        'data_root':  r'/homes/syli/dataset/EC_seg/EC-old1',
        'index_path': r'/homes/syli/python/yantao/dada/index',
        'data_modes': ['T1CE.nii'],
        'roi_modes': ['T1CE_roi.nii.gz'],
        'log_path': r'/homes/syli/python/yantao/dada/Unet_ALL_ROI_logs',
        'input_shape': [320, 320, 32],
        'batch_size': 2,
    }
    main(params)