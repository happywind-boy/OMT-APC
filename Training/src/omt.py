import torch


def InverseOMT(omt_tensor, binary_tensor, idx, inv_idx, device):
    """Inverse OMT
    Args:
        omt_tensor    (torch.FloatTensor): with shape (1, nc, n, n, n).
        binary_tensor (torch.FloatTensor): with shape (1, nx, ny, nz).
        idx           (torch.FloatTensor).
        inv_idx       (torch.FloatTensor) with shape (1, n^3, 1).

    Returns:
        inv_tensor (torch.FloatTensor): with shape (1, nc, nx, ny, nz).
    """
    omt_tensor = omt_tensor.float()
    idx = idx.long().squeeze()
    inv_idx = inv_idx.long().squeeze()
    _, nx, ny, nz = binary_tensor.shape
    nc = omt_tensor.shape[1]

    binary_tensor = binary_tensor.reshape(-1)
    omt_tensor = omt_tensor.reshape(nc, -1)
    inv_tensor = torch.zeros((nc, nx*ny*nz), dtype=torch.float, device=device)
    inv_tensor[:, binary_tensor] = omt_tensor[:, idx]

    vote_pred = torch.zeros((nc, nx*ny*nz), dtype=torch.float, device=device)
    max_bound = torch.max(inv_idx) + 1

    for i in range(nc):
        vote_pred[i, :max_bound] = torch.bincount(inv_idx, weights=omt_tensor[i, :])

    unique_id, occur_times = torch.unique(inv_idx, return_counts=True)
    unique_id = unique_id.long()
    vote_pred[:, unique_id] /= occur_times
    inv_tensor[:, unique_id] = vote_pred[:, unique_id]
    inv_tensor = inv_tensor.reshape((1, nc, nx, ny, nz))

    return inv_tensor
