import torch
import numpy as np

def find_fixed_points_KE_min(model, sample_states, lr=0.1, max_iter=1000, tol=1e-9, dup_thresh=0.1):
    """
    Vectorized fixed-point finder via kinetic-energy minimization.

    model: trained RNN model
    sample_states: tensor of shape (num_samples, hidden_size) or (batch, T, hidden_size)
    lr: learning rate for Adam
    max_iter: maximum optimisation steps
    tol: energy threshold to declare convergence
    dup_thresh: L2 distance below which two fixed points are considered duplicates
    """

    W_rec = model.W_rec.detach()
    b_rec = model.b_rec.detach()

    # Flatten to 2D: (total_samples, hidden_size)
    if sample_states.dim() == 3:
        sample_states = sample_states.reshape(-1, model.hidden_size)

    # ---- batched optimisation ------------------------------------------------
    r = sample_states.clone().detach().requires_grad_(True)  # (N, hidden_size)
    optimizer = torch.optim.Adam([r], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()

        r_next = torch.tanh(r @ W_rec.t() + b_rec)           # (N, hidden_size)
        energies = torch.sum(torch.square(r_next - r), dim=1) # (N,)
        total_energy = energies.sum()

        total_energy.backward()
        optimizer.step()

        # stop early if every candidate has converged
        if energies.max().item() < tol:
            break

    # ---- filter converged points ---------------------------------------------
    with torch.no_grad():
        r_next = torch.tanh(r @ W_rec.t() + b_rec)
        energies = torch.sum(torch.square(r_next - r), dim=1)
        converged_mask = energies < tol

    if not converged_mask.any():
        return np.array([])

    fps = r[converged_mask].detach()  # (M, hidden_size)

    # ---- remove duplicates (vectorized pairwise check) -----------------------
    dists = torch.cdist(fps, fps)                       # (M, M)
    keep = torch.ones(fps.shape[0], dtype=torch.bool)
    for i in range(fps.shape[0]):
        if not keep[i]:
            continue
        # mark later duplicates
        keep[i + 1:] &= dists[i, i + 1:] >= dup_thresh

    unique_fps = fps[keep].cpu().numpy()
    return unique_fps



def g_residual(r, W, b):
    """g(r) = -r + tanh(r W^T + b)."""
    a = r @ W.t() + b
    g = -r + torch.tanh(a)
    return g, a


def S_and_residual(r, W, b):
    """S(r) = ||g(r)||^2 and residual g(r). Supports r of shape (H,) or (N,H)."""
    if r.dim() == 1:
        r_ = r.unsqueeze(0)
        squeeze = True
    else:
        r_ = r
        squeeze = False

    a = r_ @ W.t() + b
    g = -r_ + torch.tanh(a)
    S = (g ** 2).sum(dim=-1)

    if squeeze:
        return S[0], g[0]
    return S, g


def solve_fixed_point_gd(
    r0, W, b,
    lr=1e-2,
    max_steps=4000,
    tol_S=1e-10,
    patience=300,
):
    """Minimize S(r) by Adam starting from r0."""
    Wc = W.detach()
    bc = b.detach()

    r = nn.Parameter(r0.detach().clone())
    opt = torch.optim.Adam([r], lr=lr)

    best_S = float("inf")
    best_r = None
    stall = 0

    for _ in range(max_steps):
        opt.zero_grad()
        S, _ = S_and_residual(r, Wc, bc)
        S.backward()
        opt.step()

        sval = float(S.detach().cpu())
        if sval < best_S:
            best_S = sval
            best_r = r.detach().clone()
            stall = 0
        else:
            stall += 1

        if sval < tol_S:
            break
        if stall >= patience:
            break

    r_star = best_r if best_r is not None else r.detach().clone()
    with torch.no_grad():
        S_final, g_final = S_and_residual(r_star, Wc, bc)
        return r_star, float(S_final.cpu()), float(torch.norm(g_final).cpu())






@torch.no_grad()
def newton_refine(r_init, W, b, steps=20, ridge=1e-6, damping=1.0, tol=1e-12):
    """Newton refinement for g(r)=0.

    Uses analytic Jacobian:
      J_g = -I + diag(sech^2(a)) W, where a = r W^T + b.

    ridge adds a small multiple of I for numerical stability.
    damping allows a damped Newton step if needed.
    """
    r = r_init.clone()
    H = r.numel()
    I = torch.eye(H, device=r.device, dtype=r.dtype)

    for _ in range(steps):
        g, a = g_residual(r, W, b)
        if torch.norm(g) < tol:
            break

        d = 1.0 - torch.tanh(a) ** 2           # sech^2(a)
        J = -I + (d.unsqueeze(1) * W)          # diag(d) @ W

        J_reg = J + ridge * I
        try:
            delta = torch.linalg.solve(J_reg, g)
        except RuntimeError:
            delta = torch.linalg.lstsq(J_reg, g).solution

        r_new = r - damping * delta

        # basic safeguard: keep the step if residual decreases
        g_new, _ = g_residual(r_new, W, b)
        if torch.norm(g_new) <= torch.norm(g):
            r = r_new
        else:
            r = r_new  # accept anyway; you can implement stronger line search if desired

    return r


def solve_fixed_point_hybrid(
    r0, W, b,
    gd_steps=1000,
    gd_lr=1e-2,
    tol_S=1e-10,
    patience=300,
    newton_steps=20,
    newton_ridge=1e-6,
    newton_damping=1.0,
):
    """Gradient descent on S(r) followed by Newton refinement on g(r)=0."""
    # run GD with a hard cap gd_steps
    Wc = W.detach()
    bc = b.detach()

    r = nn.Parameter(r0.detach().clone())
    opt = torch.optim.Adam([r], lr=gd_lr)

    best_S = float("inf")
    best_r = None
    stall = 0

    for _ in range(gd_steps):
        opt.zero_grad()
        S, _ = S_and_residual(r, Wc, bc)
        S.backward()
        opt.step()

        sval = float(S.detach().cpu())
        if sval < best_S:
            best_S = sval
            best_r = r.detach().clone()
            stall = 0
        else:
            stall += 1

        if sval < tol_S:
            break
        if stall >= patience:
            break

    r_star = best_r if best_r is not None else r.detach().clone()

    # Newton refinement (optional)
    r_star = newton_refine(
        r_star, Wc, bc,
        steps=newton_steps,
        ridge=newton_ridge,
        damping=newton_damping,
    )

    with torch.no_grad():
        S_final, g_final = S_and_residual(r_star, Wc, bc)
        return r_star, float(S_final.cpu()), float(torch.norm(g_final).cpu())


@torch.no_grad()
def spectral_radius_leaky(r_star, W, b, alpha):
    """Spectral radius of the Jacobian of the leaky update at r_star."""
    H = r_star.numel()
    I = torch.eye(H, device=r_star.device, dtype=r_star.dtype)

    a = r_star @ W.t() + b
    d = 1.0 - torch.tanh(a) ** 2         # sech^2(a)
    JF = d.unsqueeze(1) * W              # derivative of F(r)=tanh(r W^T + b)
    J_leaky = (1.0 - alpha) * I + alpha * JF

    eigvals = torch.linalg.eigvals(J_leaky)
    return torch.max(torch.abs(eigvals)).item()
