import torch

batch = 1
channel = 3
N = 10
idx_num_pts = 5

# features_shape: (batch, channel, N)
features = torch.rand((batch, channel, N))

# idx_shape: (batch, idx_num_pts)
idx = torch.randint(0, 9, (batch, idx_num_pts))
idx = idx[:, None, :].repeat(1, channel, 1)

# gathers shape: (batch, channel, idx_num_pts)
gathers = torch.gather(features, dim=2, index=idx)
print(gathers)
print(gathers.shape)
