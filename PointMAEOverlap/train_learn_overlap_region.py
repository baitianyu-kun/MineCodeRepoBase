import os
import torch
from torch import optim, nn
from tqdm import tqdm
from metrics.losses import chamfer_loss
import utils.logger
from utils.train_utils import setup_seed
from RIGARework.partial_threeDmatchDataset import ThreeDMatchPartial
from RIGARework.learnOverlapRegion import LearnOverlapRegion


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
    bar_format = '{desc}{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=bar_format)
    train_loop.set_description(f'Train [{epoch}]')
    for i, data in train_loop:
        ps, pt, iGgt, iRgt, itgt, gt_mask_src, gt_mask_tgt = data
        batch, num_pts = ps.shape[0], ps.shape[1]
        ps = ps.to(device)
        pt = pt.to(device)
        iGgt = iGgt.to(device)
        iRgt = iRgt.to(device)
        itgt = itgt.to(device)
        gt_mask_src = gt_mask_src.to(device)
        gt_mask_tgt = gt_mask_tgt.to(device)
        decode_ps_overlap, decode_pt_overlap = model(ps, pt)

        gt_mask_src = gt_mask_src.repeat_interleave(3, dim=1).view(batch, num_pts, 3)
        gt_mask_tgt = gt_mask_tgt.repeat_interleave(3, dim=1).view(batch, num_pts, 3)

        ps_overlap_region = ps[gt_mask_src == 1].view(batch, -1, 3).clone()
        pt_overlap_region = pt[gt_mask_tgt == 1].view(batch, -1, 3).clone()

        loss = chamfer_loss(decode_ps_overlap, ps_overlap_region) + chamfer_loss(decode_pt_overlap, pt_overlap_region)
        opt.zero_grad()
        total_loss += loss
        loss.backward()
        opt.step()
    return total_loss


def test_one_epoch(model, test_loader, epoch):
    model = model.eval()
    total_loss = 0
    bar_format = '{desc}{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
    test_loop = tqdm(enumerate(test_loader), total=len(test_loader), bar_format=bar_format)
    test_loop.set_description(f'Test [{epoch}]')
    with torch.no_grad():
        for i, data in test_loop:
            ps, pt, iGgt, iRgt, itgt, gt_mask_src, gt_mask_tgt = data
            batch, num_pts = ps.shape[0], ps.shape[1]
            ps = ps.to(device)
            pt = pt.to(device)
            iGgt = iGgt.to(device)
            iRgt = iRgt.to(device)
            itgt = itgt.to(device)
            gt_mask_src = gt_mask_src.to(device)
            gt_mask_tgt = gt_mask_tgt.to(device)
            decode_ps_overlap, decode_pt_overlap = model(ps, pt)
            gt_mask_src = gt_mask_src.repeat_interleave(3, dim=1).view(batch, num_pts, 3)
            gt_mask_tgt = gt_mask_tgt.repeat_interleave(3, dim=1).view(batch, num_pts, 3)
            ps_overlap_region = ps[gt_mask_src == 1].view(batch, -1, 3).clone()
            pt_overlap_region = pt[gt_mask_tgt == 1].view(batch, -1, 3).clone()
            loss = chamfer_loss(decode_ps_overlap, ps_overlap_region) + chamfer_loss(decode_pt_overlap,
                                                                                     pt_overlap_region)
            total_loss += loss
    return total_loss


def run(model, opt, train_loader, test_loader):
    for epoch in range(epoches):
        train_total_loss = train_one_epoch(model, opt, train_loader, epoch)

        logger.info(f'Train Epoch {epoch} Loss: %.6f' % (train_total_loss))

        test_total_loss = test_one_epoch(model, test_loader, epoch)

        logger.info(f'Test Epoch {epoch} Loss: %.6f' % (test_total_loss))

        save(model, opt, epoch, test_total_loss, model_save_path)


if __name__ == '__main__':
    setup_seed(1234)
    epoches = 100
    batch_size = 2
    lr = 0.01
    num_pts = 1024
    max_angle = 45
    max_t = 0.5
    noise = 0.01
    shuffle_pts = True
    subsampled_rate_src = 0.6
    subsampled_rate_tgt = 0.6
    workers = 4
    DATA_DIR = 'D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1'
    device = 'cuda:0'

    log_save_path = '../RIGARework/overlap_log/'
    log_name = 'default_name'
    logger, log_path = utils.logger.prepare_logger(log_save_path, log_name)
    model_save_path = os.path.join(log_path, 'weights_saved')
    trainloader = torch.utils.data.DataLoader(
        ThreeDMatchPartial(DATA_DIR, partition='train', partial_overlap=2, num_pts=num_pts, max_angle=max_angle,
                           max_t=max_t, shuffle_pts=shuffle_pts, subsampled_rate_src=subsampled_rate_src,
                           subsampled_rate_tgt=subsampled_rate_tgt),
        batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = torch.utils.data.DataLoader(
        ThreeDMatchPartial(DATA_DIR, partition='test', partial_overlap=2, num_pts=num_pts, max_angle=max_angle,
                           max_t=max_t, shuffle_pts=shuffle_pts, subsampled_rate_src=subsampled_rate_src,
                           subsampled_rate_tgt=subsampled_rate_tgt),
        batch_size=batch_size, shuffle=True, num_workers=workers)
    model = LearnOverlapRegion(partial_num_points=204).to(device)
    opt = optim.AdamW(params=model.parameters(), lr=lr)
    run(model, opt, trainloader, testloader)
