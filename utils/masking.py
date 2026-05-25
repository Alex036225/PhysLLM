import torch


class TriangularCausalMask:
    def __init__(self, batch_size, seq_len, device="cpu"):
        mask_shape = (batch_size, 1, seq_len, seq_len)
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1
            )

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, batch_size, num_heads, seq_len, index, scores, device="cpu"):
        base_mask = torch.ones(
            (seq_len, scores.shape[-1]), dtype=torch.bool, device=device
        ).triu(1)
        expanded_mask = base_mask[None, None, :, :].expand(batch_size, num_heads, -1, -1)
        selector = index[:, :, :, None].expand(-1, -1, -1, scores.shape[-1])
        self._mask = expanded_mask.gather(2, selector)

    @property
    def mask(self):
        return self._mask
