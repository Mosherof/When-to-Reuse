import torch
import numpy as np

def process_R_list(R_list):
    # R_list: list of N_epochs tensors, each of shape (B, T+1, H)
    # Returns: list of N_epochs tensors, each of shape (B*(T+1), H)
    # This flattens batch × time into a single seed dimension while
    # preserving the epoch structure (so indices align with weight_snapshots).
    print("Shape of R_list:", len(R_list), "x", R_list[0].shape)

    R_list = [r.reshape(-1, r.shape[-1]) for r in R_list]
    print("Shape of R_list after processing:", len(R_list), "x", R_list[0].shape)
    
    return R_list
    
    