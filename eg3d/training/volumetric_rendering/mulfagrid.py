import math
import torch
import torch.nn.functional as F

def fourier_features(coords, df=8):
    # coords: (N, M, 2)
    f_feature_list = []
    for i in range(df):
        if i % 2 == 0:
            f_feature_list.append(torch.sin(2 ** (i // 2) * math.pi * coords))
        else:
            f_feature_list.append(torch.cos(2 ** (i // 2) * math.pi * coords))
    return torch.cat(f_feature_list, dim=-1)

def mulfagrid_kernel_fn(coords, # (N, M, 2)
                        indices, # (N, M, 1)
                        w_list,
                        phi_list,
                        W_list,
                        b_list
                        ):
    # Note W and b is one length longer than w and phi
    z = fourier_features(coords) # (N, M, 2 * 8)
    for w, phi, W, b in zip(w_list, phi_list, W_list, b_list):
        z = F.linear(z, W, b) * torch.sin(F.linear(indices.float(), w, phi))
    z = F.linear(z, W_list[-1], b_list[-1])
    return z

def _triplane_indices(indices, plane_size):
    N, M, _ = indices.shape
    indices = indices.reshape(N // 3, 3, M, 1) 
    indices[:, 1, :, :] = indices[:, 1, :, :] + plane_size
    indices[:, 2, :, :] = indices[:, 2, :, :] + plane_size * 2
    return indices.reshape(N, M, 1).detach()

def mulfagrid_fn(plane_features, projected_coordinates,
                 w_list, phi_list, W_list, b_list):
    """ The filters notations are the same with the paper https://arxiv.org/abs/2403.20002
        Equation (7)
    """
    # Get the indices
    N, _, H, W = plane_features.shape
    coords = (projected_coordinates + 1.0) / 2.0
    xf = torch.clamp((coords[:, :, 0] * (H -1)), 0, H - 1)
    yf = torch.clamp((coords[:, :, 1] * (W -1)), 0, W - 1)
    x = xf.floor().long()
    y = yf.floor().long()
    x0 = torch.clamp(x, 0, H - 2)
    x1 = torch.clamp(x + 1, 0, H - 1)
    y0 = torch.clamp(y, 0, W - 2)
    y1 = torch.clamp(y + 1, 0, W - 1)

    # Get the features
    batch_indices = torch.arange(N, device=plane_features.device).view(N, 1).expand(N, M)
    # Coords
    coordf = torch.stack([xf, yf], dim=-1)
    # Indices, (N, M, 2)
    top_left_indices = _triplane_indices((x0 * W + y0).unsqueeze(dim=-1), H * W)
    top_right_indices = _triplane_indices((x0 * W + y1).unsqueeze(dim=-1), H * W)
    bottom_left_indices = _triplane_indices((x1 * W + y0).unsqueeze(dim=-1), H * W)
    bottom_right_indices = _triplane_indices((x1 * W + y1).unsqueeze(dim=-1), H * W)
    # Gather features, (N, C, M) 
    top_left_feature = plane_features[batch_indices, :, x0, y0].permute(0, 2, 1)
    top_right_feature = plane_features[batch_indices, :, x0, y1].permute(0, 2, 1)
    bottom_left_feature = plane_features[batch_indices, :, x1, y0].permute(0, 2, 1)
    bottom_right_feature = plane_features[batch_indices, :, x1, y1].permute(0, 2, 1)

    # Get the kernel function results
    tl_varphi = mulfagrid_kernel_fn(coordf, top_left_indices, w_list, phi_list, W_list, b_list).permute(0, 2, 1)
    tr_varphi = mulfagrid_kernel_fn(coordf, top_right_indices, w_list, phi_list, W_list, b_list).permute(0, 2, 1)
    bl_varphi = mulfagrid_kernel_fn(coordf, bottom_left_indices, w_list, phi_list, W_list, b_list).permute(0, 2, 1)
    br_varphi = mulfagrid_kernel_fn(coordf, bottom_right_indices, w_list, phi_list, W_list, b_list).permute(0, 2, 1)

    # Normalization the kernel function result
    sum_varphi = tl_varphi + tr_varphi + bl_varphi + br_varphi
    tl_varphi = tl_varphi / sum_varphi
    tr_varphi = tr_varphi / sum_varphi
    bl_varphi = bl_varphi / sum_varphi
    br_varphi = br_varphi / sum_varphi

    # Aggregate them, out of (N*nplanes, C, M)
    out = (top_left_feature * tl_varphi + 
           top_right_feature * tr_varphi +
           bottom_left_feature * bl_varphi +
           bottom_right_feature * br_varphi)

    return out


if __name__ == "__main__":
    # Example input tensors
    N, C, H, W = 6, 3, 5, 5
    M = 10 # Number of sampled points
    plane_features = torch.randn(N, C, H, W)
    projected_coordinates = torch.rand(N, M, 2)  # coordinates in range [0, 1]

    n = 4
    w_list = []
    phi_list = []
    W_list = []
    b_list = []
    for _ in range(n):
        w_list.append(torch.randn(16, 1))
        phi_list.append(torch.randn(1))
        W_list.append(torch.randn(16, 16))
        b_list.append(torch.randn(1))
    W_list.append(torch.randn(1, 16))
    b_list.append(torch.randn(1))
    mulfagrid_fn(plane_features, projected_coordinates, w_list, phi_list, W_list, b_list)
