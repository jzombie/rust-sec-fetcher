import torch

def stage1_collate_with_scaler(batch):
    """
    Custom collate function that correctly handles a list of individual samples.
    Each sample is a tuple: (x, y, scaler, concept_unit).
    """

    # Unzip the list of tuples into separate lists
    i_cell_list, xs, ys, scalers_list, concept_units = zip(*batch)

    # Stack the tensors to create a batch
    xs_batch = torch.stack(xs)
    ys_batch = torch.stack(ys)

    # The scalers and concept_units remain as lists
    return i_cell_list, xs_batch, ys_batch, scalers_list, list(concept_units)
