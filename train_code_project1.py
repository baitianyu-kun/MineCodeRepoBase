import os

import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.timer
import utils.logger
from utils.train_utils import setup_seed
from code_project1.threeDmatchDataset import ThreeDMatch
from metrics import benchmarks_R_t
from se_math import se3
from code_project1.rie_model import RIENET


def write_scalar(tuple_metrics, tuple_losses, epoch, isTrain='Train'):
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = tuple_metrics
    loss1, loss2, loss3, total_loss = tuple_losses
    writer.add_scalar(f'{isTrain}/TOTAL_LOSS', total_loss, global_step=epoch)
    writer.add_scalar(f'{isTrain}/LOSS_1', loss1, global_step=epoch)
    writer.add_scalar(f'{isTrain}/LOSS_2', loss2, global_step=epoch)
    writer.add_scalar(f'{isTrain}/LOSS_3', loss3, global_step=epoch)
    writer.add_scalar(f'{isTrain}/R_MSE', r_mse, global_step=epoch)
    writer.add_scalar(f'{isTrain}/R_MAE', r_mae, global_step=epoch)
    writer.add_scalar(f'{isTrain}/T_MSE', t_mse, global_step=epoch)
    writer.add_scalar(f'{isTrain}/T_MAE', t_mae, global_step=epoch)
    writer.add_scalar(f'{isTrain}/R_IOS', r_isotropic, global_step=epoch)
    writer.add_scalar(f'{isTrain}/T_IOS', t_isotropic, global_step=epoch)
    logger.info(f'{isTrain} Epoch {epoch} Loss: %.4f, %.4f, %.4f, %.4f' % (
        loss1, loss2, loss3, total_loss))
    logger.info(f'{isTrain} Epoch {epoch} Metrics: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic))


def load(model, path):
    checkpoint = torch.load(path, map_location='cpu')  # 加载断点
    model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数
    opt.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
    loss = checkpoint['loss']
    return model.to(device), opt, start_epoch, loss


def save(model, opt, epoch, loss, path):
    checkpoint = {
        "model": model.state_dict(),
        'optimizer': opt.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    os.makedirs(path, exist_ok=True)
    torch.save(checkpoint, path + f'/weight_e_{epoch}' + '.pth')


def train_one_epoch(model, opt, train_loader, epoch):
    model = model.train()
    total_loss = 0
    total_global_alignment_loss = 0
    total_consensus_loss = 0
    total_spatial_consistency_loss = 0
    bar_format = '{desc}{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=bar_format)
    train_loop.set_description(f'Train [{epoch}]')
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    for i, data in train_loop:
        ps, pt, iGgt, iRgt, itgt, gt_mask_src, gt_mask_tgt = data
        ps = ps.to(device).transpose(1, 2)
        pt = pt.to(device).transpose(1, 2)
        iGgt = iGgt.to(device)
        iRgt = iRgt.to(device)
        itgt = itgt.to(device)
        gt_mask_src = gt_mask_src.to(device)
        gt_mask_tgt = gt_mask_tgt.to(device)

        Rpre, tpre, global_alignment_loss, consensus_loss, spatial_consistency_loss = model(ps, pt)

        loss = global_alignment_loss.sum() + consensus_loss.sum() + spatial_consistency_loss.sum()
        opt.zero_grad()

        total_loss += loss.item()
        total_global_alignment_loss += global_alignment_loss.sum().item()
        total_consensus_loss += consensus_loss.sum().item()
        total_spatial_consistency_loss += spatial_consistency_loss.sum().item()

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
        opt.step()

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = benchmarks_R_t.compute_R_t_metrics(Rpre, tpre, iRgt, torch.squeeze(itgt))
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        benchmarks_R_t.summary_R_t_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    return r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic, \
           total_global_alignment_loss, total_consensus_loss, total_spatial_consistency_loss, total_loss


def test_one_epoch(model, test_loader, epoch):
    model = model.eval()
    total_loss = 0
    total_global_alignment_loss = 0
    total_consensus_loss = 0
    total_spatial_consistency_loss = 0
    bar_format = '{desc}{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
    test_loop = tqdm(enumerate(test_loader), total=len(test_loader), bar_format=bar_format)
    test_loop.set_description(f'Test [{epoch}]')
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for i, data in test_loop:
            ps, pt, iGgt, iRgt, itgt, gt_mask_src, gt_mask_tgt = data
            ps = ps.to(device).transpose(1, 2)
            pt = pt.to(device).transpose(1, 2)
            iGgt = iGgt.to(device)
            iRgt = iRgt.to(device)
            itgt = itgt.to(device)
            gt_mask_src = gt_mask_src.to(device)
            gt_mask_tgt = gt_mask_tgt.to(device)

            Rpre, tpre, global_alignment_loss, consensus_loss, spatial_consistency_loss = model(ps, pt)
            loss = global_alignment_loss.sum() + consensus_loss.sum() + spatial_consistency_loss.sum()
            total_loss += loss.item()
            total_global_alignment_loss += global_alignment_loss.sum().item()
            total_consensus_loss += consensus_loss.sum().item()
            total_spatial_consistency_loss += spatial_consistency_loss.sum().item()

            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = benchmarks_R_t.compute_R_t_metrics(Rpre, tpre, iRgt, torch.squeeze(itgt))
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        benchmarks_R_t.summary_R_t_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    return r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic, \
           total_global_alignment_loss, total_consensus_loss, total_spatial_consistency_loss, total_loss


def run(model, opt, train_loader, test_loader):
    for epoch in range(epoches):
        train_r_mse, train_r_mae, train_t_mse, train_t_mae, train_r_isotropic, train_t_isotropic, \
        train_total_global_alignment_loss, train_total_consensus_loss, train_total_spatial_consistency_loss, \
        train_total_loss = train_one_epoch(model, opt, train_loader, epoch)

        test_r_mse, test_r_mae, test_t_mse, test_t_mae, test_r_isotropic, test_t_isotropic, \
        test_total_global_alignment_loss, test_total_consensus_loss, test_total_spatial_consistency_loss, \
        test_total_loss = test_one_epoch(model, test_loader, epoch)

        write_scalar((train_r_mse, train_r_mae, train_t_mse, train_t_mae, train_r_isotropic, train_t_isotropic)
                     , (train_total_global_alignment_loss, train_total_consensus_loss,
                        train_total_spatial_consistency_loss, train_total_loss), epoch=epoch, isTrain='Train')
        write_scalar((test_r_mse, test_r_mae, test_t_mse, test_t_mae, test_r_isotropic, test_t_isotropic),
                     (test_total_global_alignment_loss, test_total_consensus_loss,
                      test_total_spatial_consistency_loss, test_total_loss), epoch=epoch, isTrain='Test')
        save(model, opt, epoch, test_total_loss, model_save_path)


if __name__ == '__main__':
    epoches = 100
    batch_size = 2
    lr = 0.01
    num_pts = 1024
    max_angle = 45
    max_t = 0.5
    noise = 0.01
    shuffle_pts = True
    subsampled_rate_src = 0.7
    subsampled_rate_tgt = 0.7
    workers = 4
    DATA_DIR = 'D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1'
    device = 'cuda:0'
    model_save_path = './code_project1/weights_saved/'
    log_save_path = './code_project1/log_saved/'
    log_name = 'default_name'

    setup_seed(1234)
    writer = SummaryWriter(log_save_path)
    # timer = utils.timer.Timer()
    logger, log_path = utils.logger.prepare_logger(log_save_path, log_name)

    trainloader = torch.utils.data.DataLoader(
        ThreeDMatch(DATA_DIR, partition='train', partial_overlap=2, num_pts=num_pts, max_angle=max_angle,
                    max_t=max_t, shuffle_pts=shuffle_pts, subsampled_rate_src=subsampled_rate_src,
                    subsampled_rate_tgt=subsampled_rate_tgt),
        batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = torch.utils.data.DataLoader(
        ThreeDMatch(DATA_DIR, partition='test', partial_overlap=2, num_pts=num_pts, max_angle=max_angle,
                    max_t=max_t, shuffle_pts=shuffle_pts, subsampled_rate_src=subsampled_rate_src,
                    subsampled_rate_tgt=subsampled_rate_tgt),
        batch_size=batch_size, shuffle=True, num_workers=workers)
    model = RIENET().to(device)
    opt = optim.AdamW(params=model.parameters(), lr=lr)
    run(model, opt, trainloader, testloader)
