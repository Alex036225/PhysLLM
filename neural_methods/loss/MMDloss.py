import torch
import itertools

def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(x.size(0)) + int(y.size(0))
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_loss(source, target):
    batch_size = int(source.size(0))
    kernels = gaussian_kernel(source, target)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    return torch.mean(XX + YY - XY - YX)

def pairwise_mmd_loss(feature_list):
    assert isinstance(feature_list, list) and len(feature_list) >= 2, "Need at least two domains"
    
    total_loss = 0.0
    count = 0
    
    for feat1, feat2 in itertools.combinations(feature_list, 2):
        # Flatten features to [N, D]
        feat1_flat = feat1.view(-1, feat1.shape[-1])
        feat2_flat = feat2.view(-1, feat2.shape[-1])
        
        total_loss += mmd_loss(feat1_flat, feat2_flat)
        count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0, device=feature_list[0].device)

