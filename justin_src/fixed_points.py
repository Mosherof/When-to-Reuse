import torch
import numpy as np

def find_fixed_points_KE_min(model, sample_states, lr=0.1, max_iter=1000, tol=1e-9):
    """
    model: trained RNN model
    sample_states: tensor of shape (num_samples, hidden_size)
    """
    
    W_rec = model.W_rec.detach()
    b_rec = model.b_rec.detach()
    
    # Flatten to 2D: (total_samples, hidden_size)
    if sample_states.dim() == 3:
        sample_states = sample_states.reshape(-1, model.hidden_size)
        
    n_inits = sample_states.shape[0]
    
    fixed_points = []
    for i in range(n_inits):
        r = sample_states[i:i+1].clone().detach().requires_grad_(True)  # shape (1, hidden_size)
        
        optimizer = torch.optim.Adam([r], lr=lr)
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            r_next = torch.tanh(r @ W_rec.t() + b_rec)
            energy = torch.sum(torch.square(r_next - r))
            
            energy.backward()
            optimizer.step()
            
            if energy.item() < tol:
                break
            
        if energy.item() < tol:
            fixed_points.append(r.detach().clone())
            
    # 3. Remove duplicates
    unique_fps = []
    for fp in fixed_points:
        is_duplicate = False
        for ufp in unique_fps:
            if torch.norm(fp - ufp) < 0.1:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_fps.append(fp)
            
    print(n_inits, "initializations, found", len(unique_fps), "unique fixed points")
            
    unique_fps = torch.cat(unique_fps, dim=0).cpu().numpy() if len(unique_fps) > 0 else np.array([])
            
    return unique_fps