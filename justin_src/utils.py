import torch
import numpy as np

def process_R_list(R_list):
    print("Shape of R_list:", np.array(R_list).shape)

    R_list = torch.cat(R_list, dim=0)
    R_list = list(torch.unbind(R_list, dim=0))
    print("Shape of R_list after processing:", np.array(R_list).shape)
    
    return R_list
    
    