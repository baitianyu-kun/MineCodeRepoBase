import os

import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.logger
from utils.train_utils import setup_seed
from RIGARework.threeDmatchDataset import ThreeDMatchRIGA
from metrics import benchmarks_R_t
from se_math import se3
from RIGARework.bak.RIGAModel_origin_ver import PPFLocalAndGlobal


def write_scalar(tuple_metrics, tuple_losses, epoch, isTrain='Train'):
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = tuple_metrics
    total_loss = tuple_losses
    writer.add_scalar(f'{isTrain}/TOTAL_LOSS', total_loss, global_step=epoch)
    writer.add_scalar(f'{isTrain}/R_MSE', r_mse, global_step=epoch)
    writer.add_scalar(f'{isTrain}/R_MAE', r_mae, global_step=epoch)
    writer.add_scalar(f'{isTrain}/T_MSE', t_mse, global_step=epoch)
    writer.add_scalar(f'{isTrain}/T_MAE', t_mae, global_step=epoch)
    writer.add_scalar(f'{isTrain}/R_IOS', r_isotropic, global_step=epoch)
    writer.add_scalar(f'{isTrain}/T_IOS', t_isotropic, global_step=epoch)
    logger.info(f'{isTrain} Epoch {epoch} Loss: %.4f' % (total_loss))
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
    loss_fun = nn.MSELoss(reduction='mean')
    total_loss = 0
    bar_format = '{desc}{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=bar_format)
    train_loop.set_description(f'Train [{epoch}]')
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    for i, data in train_loop:
        ps, pt, ns, nt, iGgt, iRgt, itgt = data
        ps = ps.to(device)
        pt = pt.to(device)
        ns = ns.to(device)
        nt = nt.to(device)
        iGgt = iGgt.to(device)
        iRgt = iRgt.to(device)
        itgt = itgt.to(device)

        Gpre = model(ps, pt, ns, nt)
        loss = loss_fun(se3.transform_torch(se3.inverse(iGgt), ps), se3.transform_torch(Gpre, ps))
        Rpre, tpre = se3.decompose_trans(Gpre)

        opt.zero_grad()
        total_loss += loss.item()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)
        opt.step()

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = benchmarks_R_t.compute_R_t_metrics(Rpre, torch.squeeze(tpre), iRgt, torch.squeeze(itgt))
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        benchmarks_R_t.summary_R_t_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    return r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic, total_loss


def test_one_epoch(model, test_loader, epoch):
    model = model.eval()
    loss_fun = nn.MSELoss(reduction='mean')
    total_loss = 0
    bar_format = '{desc}{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
    test_loop = tqdm(enumerate(test_loader), total=len(test_loader), bar_format=bar_format)
    test_loop.set_description(f'Test [{epoch}]')
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for i, data in test_loop:
            ps, pt, ns, nt, iGgt, iRgt, itgt = data
            ps = ps.to(device)
            pt = pt.to(device)
            ns = ns.to(device)
            nt = nt.to(device)
            iGgt = iGgt.to(device)
            iRgt = iRgt.to(device)
            itgt = itgt.to(device)

            Gpre = model(ps, pt, ns, nt)
            loss = loss_fun(se3.transform_torch(se3.inverse(iGgt), ps), se3.transform_torch(Gpre, ps))
            Rpre, tpre = se3.decompose_trans(Gpre)

            total_loss += loss.item()

            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = benchmarks_R_t.compute_R_t_metrics(Rpre, torch.squeeze(tpre), iRgt, torch.squeeze(itgt))
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        benchmarks_R_t.summary_R_t_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    return r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic, total_loss


def run(model, opt, train_loader, test_loader):
    for epoch in range(epoches):
        train_r_mse, train_r_mae, train_t_mse, train_t_mae, train_r_isotropic, train_t_isotropic, \
        train_total_loss = train_one_epoch(model, opt, train_loader, epoch)

        test_r_mse, test_r_mae, test_t_mse, test_t_mae, test_r_isotropic, test_t_isotropic, \
        test_total_loss = test_one_epoch(model, test_loader, epoch)

        write_scalar((train_r_mse, train_r_mae, train_t_mse, train_t_mae, train_r_isotropic, train_t_isotropic)
                     , (train_total_loss), epoch=epoch, isTrain='Train')
        write_scalar((test_r_mse, test_r_mae, test_t_mse, test_t_mae, test_r_isotropic, test_t_isotropic),
                     (test_total_loss), epoch=epoch, isTrain='Test')
        save(model, opt, epoch, test_total_loss, model_save_path)


if __name__ == '__main__':
    setup_seed(1234)

    epoches = 100
    batch_size = 2
    lr = 0.01
    num_pts = 512
    max_angle = 45
    max_t = 0.5
    noise = 0.01
    shuffle_pts = True
    subsampled_rate_src = 0.8
    subsampled_rate_tgt = 0.8
    workers = 4
    DATA_DIR = 'D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1'
    device = 'cuda:0'

    log_save_path = './RIGARework/log_saved/'
    log_name = 'default_name'
    logger, log_path = utils.logger.prepare_logger(log_save_path, log_name)
    writer = SummaryWriter(log_path)
    model_save_path = os.path.join(log_path, 'weights_saved')

    trainloader = torch.utils.data.DataLoader(
        ThreeDMatchRIGA(DATA_DIR, partition='train', partial_overlap=0, num_pts=num_pts, max_angle=max_angle,
                        max_t=max_t, shuffle_pts=shuffle_pts, subsampled_rate_src=subsampled_rate_src,
                        subsampled_rate_tgt=subsampled_rate_tgt),
        batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = torch.utils.data.DataLoader(
        ThreeDMatchRIGA(DATA_DIR, partition='test', partial_overlap=0, num_pts=num_pts, max_angle=max_angle,
                        max_t=max_t, shuffle_pts=shuffle_pts, subsampled_rate_src=subsampled_rate_src,
                        subsampled_rate_tgt=subsampled_rate_tgt),
        batch_size=batch_size, shuffle=True, num_workers=workers)
    model = PPFLocalAndGlobal().to(device)
    opt = optim.AdamW(params=model.parameters(), lr=lr)
    run(model, opt, trainloader, testloader)
