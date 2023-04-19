import torch
from torch import optim, nn
from tqdm import tqdm
from utils.train_utils import setup_seed
from threeDmatchDataset import ThreeDMatch
from model import ModelTest
from metrics import benchmarks_R_t
from se_math import se3

epoches = 100
batch_size = 32
lr = 0.01
dim_k = 1024
num_pts = 1024
max_angle = 45
max_t = 0.5
noise = 0.01
shuffle_pts = True
workers = 4
DATA_DIR = '/home/bty/datasets'
device = 'cuda'

setup_seed(1234)
trainloader = torch.utils.data.DataLoader(
    ThreeDMatch(DATA_DIR, 'train', num_pts=num_pts, max_angle=max_angle, max_t=max_t, noise=noise,
                shuffle_pts=shuffle_pts),
    batch_size=batch_size, shuffle=True, num_workers=workers)
testloader = torch.utils.data.DataLoader(
    ThreeDMatch(DATA_DIR, 'test', num_pts=num_pts, max_angle=max_angle, max_t=max_t, noise=noise,
                shuffle_pts=shuffle_pts),
    batch_size=batch_size, shuffle=True, num_workers=workers)

model = ModelTest().to(device)
opt = optim.AdamW(params=model.parameters(), lr=lr)


def write_scalar():
    pass


def load(path):
    pass


def save(model, path):
    pass


def run(model, opt, train_loader, test_loader):
    pass


def train_one_epoch(model, opt, train_loader, epoch):
    pass


def test_one_epoch(model, opt, test_loader, epoch):
    model = model.eval()
    total_loss = 0
    loss_fn = nn.MSELoss()
    bar_format = '{desc}{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
    test_loop = tqdm(enumerate(testloader), total=len(testloader), bar_format=bar_format)
    test_loop.set_description(f'Test [{epoch}]')
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for i, data in test_loop:
            pt, ps, Ggt, Rgt, tgt = data
            pt = pt.to(device)
            ps = ps.to(device)
            Ggt = Ggt.to(device)
            Rgt = Rgt.to(device)
            tgt = tgt.to(device)
            Gpre = model()
            Rpre, tpre = se3.decompose_trans(Gpre)
            loss = loss_fn(se3.inverse(Gpre), Ggt)
            total_loss += loss.item()
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = benchmarks_R_t.compute_R_t_metrics(Rpre, tpre, Rgt, tgt)
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        benchmarks_R_t.summary_R_t_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    return r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic
