import torch

def stage2_collate_stacks(batch):
    """
    Collate function for Stage 2 autoencoder dataset.

    Args:
        batch: list of tuples, each:
          (stacks, masks, balance_idxs, period_idxs)

    Returns:
        stacks_batch: List[Tensor] [B, N_i, latent_dim]
        masks_batch: List[Tensor] [B, N_i]
        balance_batch: List[Tensor] [B, N_i]
        period_batch: List[Tensor] [B, N_i]
    """
    num_categories = max(len(s[0]) for s in batch)

    stacks_batch = []
    masks_batch = []
    balance_batch = []
    period_batch = []

    for cat_idx in range(num_categories):
        cat_stacks = []
        cat_masks = []
        cat_balance = []
        cat_period = []

        for stacks, masks, bal_idxs, per_idxs in batch:
            if cat_idx < len(stacks):
                cat_stacks.append(stacks[cat_idx])
                cat_masks.append(masks[cat_idx])
                cat_balance.append(bal_idxs[cat_idx])
                cat_period.append(per_idxs[cat_idx])
            else:
                cat_stacks.append(torch.zeros(0, stacks[0].shape[-1]))
                cat_masks.append(torch.zeros(0, dtype=torch.bool))
                cat_balance.append(torch.zeros(0, dtype=torch.long))
                cat_period.append(torch.zeros(0, dtype=torch.long))

        stacks_batch.append(torch.nn.utils.rnn.pad_sequence(cat_stacks, batch_first=True))
        masks_batch.append(torch.nn.utils.rnn.pad_sequence(cat_masks, batch_first=True))
        balance_batch.append(torch.nn.utils.rnn.pad_sequence(cat_balance, batch_first=True))
        period_batch.append(torch.nn.utils.rnn.pad_sequence(cat_period, batch_first=True))

    return stacks_batch, masks_batch, balance_batch, period_batch
