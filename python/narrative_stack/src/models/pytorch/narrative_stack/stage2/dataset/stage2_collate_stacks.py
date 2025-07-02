import torch
from torch.nn.utils.rnn import pad_sequence
from us_gaap_store import STAGE2_CATEGORY_STACKS

def stage2_collate_stacks(batch):
    """
    Collate function for Stage 2 autoencoder dataset.

    Args:
        batch: list of tuples of six tensors, one per category:
            Tuple[Tensor[N_i, latent_dim] x6]

    Returns:
        stacks_batch: List[Tensor] of shape [B, N_i, latent_dim]
        masks_batch: List[BoolTensor] of shape [B, N_i]
        balance_batch: List[LongTensor] of shape [B, N_i]
        period_batch: List[LongTensor] of shape [B, N_i]
    """

    # TODO: Extract into reusable utility
    # Dynamically compute balance/period maps
    balance_map = [ {"credit": 0, "debit": 1, "none": 2}[k.split("::")[0]]
                    for k in STAGE2_CATEGORY_STACKS ]
    period_map = [ {"instant": 1, "duration": 0}[k.split("::")[1]]
                   for k in STAGE2_CATEGORY_STACKS ]

    stacks_batch = []
    masks_batch = []
    balance_batch = []
    period_batch = []

    for cat_idx in range(len(STAGE2_CATEGORY_STACKS)):
        cat_stacks = []
        cat_masks = []
        cat_balance = []
        cat_period = []

        for sample in batch:
            stack = sample[cat_idx]
            N = stack.size(0)
            cat_stacks.append(stack)
            cat_masks.append(torch.ones(N, dtype=torch.bool))
            cat_balance.append(torch.full((N,), balance_map[cat_idx], dtype=torch.long))
            cat_period.append(torch.full((N,), period_map[cat_idx], dtype=torch.long))

        stacks_batch.append(pad_sequence(cat_stacks, batch_first=True))
        masks_batch.append(pad_sequence(cat_masks, batch_first=True))
        balance_batch.append(pad_sequence(cat_balance, batch_first=True))
        period_batch.append(pad_sequence(cat_period, batch_first=True))

    return stacks_batch, masks_batch, balance_batch, period_batch
