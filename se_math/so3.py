""" 3-d rotation group and corresponding Lie algebra """
import torch
from . import sinc
from .sinc import sinc1, sinc2, sinc3
from scipy.spatial.transform import Rotation
import numpy as np


def dcm2euler(mats: np.ndarray, seq: str = 'zyx', degrees: bool = True):
    """Converts rotation matrix to euler angles
    Args:
        mats: (B, 3, 3) containing the B rotation matricecs
        seq: Sequence of euler rotations (default: 'zyx')
        degrees (bool): If true (default), will return in degrees instead of radians
    Returns:
    """
    eulers = []
    for i in range(mats.shape[0]):
        # r = Rotation.from_dcm(mats[i])
        # CHANGED THIS MAY BE RIGHT
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=degrees))
    return np.stack(eulers)


def quat2mat(quat):
    """ Quaternions to Rotation Matrix
    Args:
        quat: (B x 4)
    Returns:
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def quatmul(q, r):
    """ Multiply quaternion(s) q with quaternion(s) r.
    Args: two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        q:
        r:
    Returns: q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    original_shape = q.shape
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def RG2angle(G):
    """ rotation R or transformation G -> angle 轴角
    Args:
        G: rotation R or transformation G
    Returns: angle
    """
    rot_trace = G[:, 0, 0] + G[:, 1, 1] + G[:, 2, 2]
    rot_deg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
    # return torch.rad2deg(rot_deg) can also use this
    return rot_deg


def cross_prod(x, y):
    z = torch.cross(x.view(-1, 3), y.view(-1, 3), dim=1).view_as(x)
    return z


def liebracket(x, y):
    return cross_prod(x, y)


def mat(x):
    """Generate Rotation R from Twist Vector x
    Args:
        x: Twist Vector x (batch, 3)
    Returns:
        R: Rotation R (batch, 3, 3)
    """
    # size: [*, 3] -> [*, 3, 3]
    x_ = x.view(-1, 3)
    x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
    O = torch.zeros_like(x1)

    X = torch.stack((
        torch.stack((O, -x3, x2), dim=1),
        torch.stack((x3, O, -x1), dim=1),
        torch.stack((-x2, x1, O), dim=1)), dim=1)
    return X.view(*(x.size()[0:-1]), 3, 3)


def vec(X):
    """ Generate Twist Vector from Rotation R
    Args:
        X: Rotation R (batch, 3, 3)
    Returns:
        x: Twist Vector (batch, 3)
    """
    X_ = X.view(-1, 3, 3)
    x1, x2, x3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
    x = torch.stack((x1, x2, x3), dim=1)
    return x.view(*X.size()[0:-2], 3)


def genvec():
    return torch.eye(3)


def genmat():
    return mat(genvec())


def RodriguesRotation(x):
    """ Same as so3 -> SO3
    Args:
        x:
    Returns:
    """
    # for autograd
    w = x.view(-1, 3)
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)
    # Rodrigues' rotation formula.
    # R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
    # R = eye(3) + sinc1(t)*W + sinc2(t)*S
    R = I + sinc.Sinc1(t) * W + sinc.Sinc2(t) * S
    return R.view(*(x.size()[0:-1]), 3, 3)


def exp(x):
    """ so3 -> SO3
    Args:
        x: twist vector (batch, 3) (w1, w2, w3)
    Returns:
        R: Rotation R (batch, 3, 3)
    """
    w = x.view(-1, 3)
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)
    # Rodrigues' rotation formula.
    # R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
    # R = eye(3) + sinc1(t)*W + sinc2(t)*S
    R = I + sinc1(t) * W + sinc2(t) * S
    return R.view(*(x.size()[0:-1]), 3, 3)


def inverse(r):
    """ Inverse R = Transpose R 旋转矩阵的逆矩阵是它的转置矩阵
    Args:
        r: Rotation R (batch, 3, 3)
    Returns:
        rt: Transposed Rotation R (batch, 3, 3)
    """
    R = r.view(-1, 3, 3)
    Rt = R.transpose(1, 2)
    return Rt.view_as(r)


def btrace(X):
    """
    Args:
        X: Matrix (batch, N, N)
    Returns:
        tr: Trace of the batch Matrix (batch)
    """
    # batch-trace: [B, N, N] -> [B]
    n = X.size(-1)
    X_ = X.view(-1, n, n)
    tr = torch.zeros(X_.size(0)).to(X)
    for i in range(tr.size(0)):
        m = X_[i, :, :]
        tr[i] = torch.trace(m)
    return tr.view(*(X.size()[0:-2]))


def log(g):
    """ SO3 -> se3
    Args:
        g: Rotation R (batch, 3, 3)
    Returns:
        x: Twist Vector (batch, 3)
    """
    eps = 1.0e-7
    R = g.view(-1, 3, 3)
    tr = btrace(R)
    c = (tr - 1) / 2
    t = torch.acos(c)
    sc = sinc1(t)
    idx0 = (torch.abs(sc) <= eps)
    idx1 = (torch.abs(sc) > eps)
    sc = sc.view(-1, 1, 1)

    X = torch.zeros_like(R)
    if idx1.any():
        X[idx1] = (R[idx1] - R[idx1].transpose(1, 2)) / (2 * sc[idx1])

    if idx0.any():
        # t[idx0] == se_math.pi
        t2 = t[idx0] ** 2
        A = (R[idx0] + torch.eye(3).type_as(R).unsqueeze(0)) * t2.view(-1, 1, 1) / 2
        aw1 = torch.sqrt(A[:, 0, 0])
        aw2 = torch.sqrt(A[:, 1, 1])
        aw3 = torch.sqrt(A[:, 2, 2])
        sgn_3 = torch.sign(A[:, 0, 2])
        sgn_3[sgn_3 == 0] = 1
        sgn_23 = torch.sign(A[:, 1, 2])
        sgn_23[sgn_23 == 0] = 1
        sgn_2 = sgn_23 * sgn_3
        w1 = aw1
        w2 = aw2 * sgn_2
        w3 = aw3 * sgn_3
        w = torch.stack((w1, w2, w3), dim=-1)
        W = mat(w)
        X[idx0] = W

    x = vec(X.view_as(g))
    return x


def transform(g, a):
    """ Rotate a use g
    Args:
        g: Rotation R (batch, 3, 3)
        a: Points (batch, 3, N) or (N, 3)
    Returns:
    """
    # g in SO(3):  * x 3 x 3
    # a in R^3:    * x 3[x N]
    if len(g.size()) == len(a.size()):
        b = g.matmul(a)
    else:
        b = g.matmul(a.unsqueeze(-1)).squeeze(-1)
    return b


def transform_np(g: np.ndarray, pts: np.ndarray):
    """ Use g to transform a numpy version
    Args:
        g: SO3 transformation matrix of size ([B,] 3, 3)
        pts: Points to be transformed ([B,] N, 3)
    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2)
    return transformed


def transform_torch(g: torch.Tensor, pts: torch.Tensor):
    """ Use g to transform a torch version
    Args:
        g: SO3 transformation matrix of size ([B,] 3, 3)
        pts: Points to be transformed ([B,] N, 3)
    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    transformed = pts[..., :3] @ torch.swapaxes(rot, -1, -2)
    return transformed


def group_prod(g, h):
    # g, h : SO(3)
    g1 = g.matmul(h)
    return g1


def vecs_Xg_ig(x):
    """ Vi = vec(dg/dxi * inv(g)), where g = exp(x)
        (== [Ad(exp(x))] * vecs_ig_Xg(x))
    """
    t = x.view(-1, 3).norm(p=2, dim=1).view(-1, 1, 1)
    X = mat(x)
    S = X.bmm(X)
    # B = x.view(-1,3,1).bmm(x.view(-1,1,3))  # B = x*x'
    I = torch.eye(3).to(X)

    # V = sinc1(t)*eye(3) + sinc2(t)*X + sinc3(t)*B
    # V = eye(3) + sinc2(t)*X + sinc3(t)*S

    V = I + sinc2(t) * X + sinc3(t) * S

    return V.view(*(x.size()[0:-1]), 3, 3)


def inv_vecs_Xg_ig(x):
    """ H = inv(vecs_Xg_ig(x)) """
    t = x.view(-1, 3).norm(p=2, dim=1).view(-1, 1, 1)
    X = mat(x)
    S = X.bmm(X)
    I = torch.eye(3).to(x)

    e = 0.01
    eta = torch.zeros_like(t)
    s = (t < e)
    c = (s == 0)
    t2 = t[s] ** 2
    eta[s] = ((t2 / 40 + 1) * t2 / 42 + 1) * t2 / 720 + 1 / 12  # O(t**8)
    eta[c] = (1 - (t[c] / 2) / torch.tan(t[c] / 2)) / (t[c] ** 2)

    H = I - 1 / 2 * X + eta * S
    return H.view(*(x.size()[0:-1]), 3, 3)


class ExpMap(torch.autograd.Function):
    """ Exp: so(3) -> SO(3)
    """

    @staticmethod
    def forward(ctx, x):
        """ Exp: R^3 -> M(3),
            size: [B, 3] -> [B, 3, 3],
              or  [B, 1, 3] -> [B, 1, 3, 3]
        """
        ctx.save_for_backward(x)
        g = exp(x)
        return g

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        g = exp(x)
        gen_k = genmat().to(x)
        # gen_1 = gen_k[0, :, :]
        # gen_2 = gen_k[1, :, :]
        # gen_3 = gen_k[2, :, :]

        # Let z = f(g) = f(exp(x))
        # dz = df/dgij * dgij/dxk * dxk
        #    = df/dgij * (d/dxk)[exp(x)]_ij * dxk
        #    = df/dgij * [gen_k*g]_ij * dxk
        dg = gen_k.matmul(g.view(-1, 1, 3, 3))
        # (k, i, j)
        dg = dg.to(grad_output)
        go = grad_output.contiguous().view(-1, 1, 3, 3)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)
        return grad_input


Exp = ExpMap.apply

# EOF
