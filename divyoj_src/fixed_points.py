import torch
import torch.nn as nn
import numpy as np


def _flatten_states(sample_states, hidden_size):
    if sample_states.dim() == 3:
        return sample_states.reshape(-1, hidden_size)
    if sample_states.dim() == 2:
        return sample_states
    raise ValueError(f"sample_states must have shape (N,H) or (B,T,H), got {tuple(sample_states.shape)}")


def _deduplicate_points(points, scores, dup_thresh):
    """Greedy deduplication keeping the lowest-score point in each neighborhood."""
    if points.shape[0] == 0:
        return points, scores

    order = torch.argsort(scores)  # best first
    points_sorted = points[order]
    scores_sorted = scores[order]

    dists = torch.cdist(points_sorted, points_sorted)
    keep = torch.ones(points_sorted.shape[0], dtype=torch.bool, device=points.device)

    for i in range(points_sorted.shape[0]):
        if not keep[i]:
            continue
        keep[i + 1:] &= dists[i, i + 1:] >= dup_thresh

    return points_sorted[keep], scores_sorted[keep]


def g_residual(r, W, b):
    """g(r) = -r + tanh(r W^T + b). Supports (H,) and (N,H)."""
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
    r0,
    W,
    b,
    lr=1e-2,
    max_steps=4000,
    tol_S=1e-10,
    patience=300,
):
    """Minimize S(r) by Adam starting from a single seed r0."""
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

        if sval < tol_S or stall >= patience:
            break

    r_star = best_r if best_r is not None else r.detach().clone()
    with torch.no_grad():
        S_final, g_final = S_and_residual(r_star, Wc, bc)
    return r_star, float(S_final.cpu()), float(torch.norm(g_final).cpu())


@torch.no_grad()
def newton_refine(r_init, W, b, steps=20, ridge=1e-6, damping=1.0, tol=1e-12):
    """Newton refinement for a single point r_init."""
    r = r_init.clone()
    H = r.numel()
    I = torch.eye(H, device=r.device, dtype=r.dtype)

    for _ in range(steps):
        g, a = g_residual(r, W, b)
        if torch.norm(g) < tol:
            break

        d = 1.0 - torch.tanh(a) ** 2
        J = -I + (d.unsqueeze(1) * W)
        J_reg = J + ridge * I

        try:
            delta = torch.linalg.solve(J_reg, g)
        except RuntimeError:
            delta = torch.linalg.lstsq(J_reg, g).solution

        r = r - damping * delta

    return r


def solve_fixed_point_hybrid(
    r0,
    W,
    b,
    gd_steps=100,
    gd_lr=1e-2,
    tol_S=1e-10,
    patience=300,
    newton_steps=5,
    newton_ridge=1e-6,
    newton_damping=1.0,
):
    """Single-seed GD->Newton hybrid solve."""
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

        if sval < tol_S or stall >= patience:
            break

    r_star = best_r if best_r is not None else r.detach().clone()
    r_star = newton_refine(
        r_star,
        Wc,
        bc,
        steps=newton_steps,
        ridge=newton_ridge,
        damping=newton_damping,
    )

    with torch.no_grad():
        S_final, g_final = S_and_residual(r_star, Wc, bc)
    return r_star, float(S_final.cpu()), float(torch.norm(g_final).cpu())


@torch.no_grad()
def _newton_refine_batched(r_init, W, b, steps=5, ridge=1e-6, damping=1.0, tol=1e-10):
    """Batched Newton refinement for r of shape (N,H)."""
    r = r_init.clone()
    if r.numel() == 0:
        return r

    N, H = r.shape
    I = torch.eye(H, device=r.device, dtype=r.dtype).unsqueeze(0).expand(N, -1, -1)

    for _ in range(steps):
        a = r @ W.t() + b
        g = -r + torch.tanh(a)
        g_norm = torch.linalg.norm(g, dim=1)
        if torch.max(g_norm).item() < tol:
            break

        d = 1.0 - torch.tanh(a) ** 2  # (N,H)
        J = -I + d.unsqueeze(2) * W.unsqueeze(0)  # (N,H,H)
        J_reg = J + ridge * I

        rhs = g.unsqueeze(-1)
        try:
            delta = torch.linalg.solve(J_reg, rhs).squeeze(-1)
        except RuntimeError:
            delta = torch.linalg.lstsq(J_reg, rhs).solution.squeeze(-1)

        r = r - damping * delta

    return r


def find_fixed_points_hybrid(
    model,
    sample_states,
    gd_lr=1e-2,
    gd_steps=100,
    newton_steps=5,
    tol=1e-9,
    dup_thresh=0.1,
    newton_ridge=1e-6,
    newton_damping=1.0,
    return_details=False,
):
    """
    Vectorized fixed-point finder with 2 stages:
      1) batched GD minimization of S(r)=||-r+tanh(rW^T+b)||^2
      2) batched Newton refinement on g(r)=0

    Defaults use 100 GD steps + 5 Newton steps.
    """
    W = model.W_rec.detach()
    b = model.b_rec.detach()

    seeds = _flatten_states(sample_states, model.hidden_size).to(W.device, dtype=W.dtype)
    if seeds.shape[0] == 0:
        if return_details:
            return np.empty((0, model.hidden_size)), {
                "energies": np.array([]),
                "residual_norms": np.array([]),
                "converged_mask": np.array([], dtype=bool),
            }
        return np.empty((0, model.hidden_size))

    r = seeds.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([r], lr=gd_lr)

    for _ in range(gd_steps):
        opt.zero_grad()
        r_next = torch.tanh(r @ W.t() + b)
        residual = r_next - r
        energies = torch.sum(residual * residual, dim=1)
        loss = energies.sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        r_refined = _newton_refine_batched(
            r.detach(),
            W,
            b,
            steps=newton_steps,
            ridge=newton_ridge,
            damping=newton_damping,
            tol=max(1e-12, np.sqrt(tol)),
        )

        r_next = torch.tanh(r_refined @ W.t() + b)
        residual = r_next - r_refined
        energies = torch.sum(residual * residual, dim=1)
        residual_norms = torch.linalg.norm(residual, dim=1)
        converged_mask = energies < tol

        fps = r_refined[converged_mask]
        fps_scores = energies[converged_mask]

        if fps.shape[0] > 0:
            fps, fps_scores = _deduplicate_points(fps, fps_scores, dup_thresh=dup_thresh)

        fps_np = fps.cpu().numpy() if fps.shape[0] > 0 else np.empty((0, model.hidden_size))

    if return_details:
        details = {
            "energies": energies.cpu().numpy(),
            "residual_norms": residual_norms.cpu().numpy(),
            "converged_mask": converged_mask.cpu().numpy(),
            "unique_energies": fps_scores.cpu().numpy() if fps.shape[0] > 0 else np.array([]),
        }
        return fps_np, details

    return fps_np


def find_fixed_points_KE_min(model, sample_states, lr=0.1, max_iter=1000, tol=1e-9, dup_thresh=0.1):
    """
    Backward-compatible wrapper.

    Keeps old signature but now uses the hybrid solver internally.
    """
    return find_fixed_points_hybrid(
        model,
        sample_states,
        gd_lr=lr,
        gd_steps=max_iter,
        newton_steps=5,
        tol=tol,
        dup_thresh=dup_thresh,
        return_details=False,
    )


@torch.no_grad()
def spectral_radius_leaky(r_star, W, b, alpha):
    """Spectral radius of the Jacobian of the leaky update at r_star."""
    H = r_star.numel()
    I = torch.eye(H, device=r_star.device, dtype=r_star.dtype)

    a = r_star @ W.t() + b
    d = 1.0 - torch.tanh(a) ** 2
    JF = d.unsqueeze(1) * W
    J_leaky = (1.0 - alpha) * I + alpha * JF

    eigvals = torch.linalg.eigvals(J_leaky)
    return torch.max(torch.abs(eigvals)).item()


@torch.no_grad()
def classify_fixed_points_stability(model, fixed_points, alpha=None, marginal_eps=1e-3):
    """
    Classify fixed points by Jacobian spectral radius.

    Returns dict with:
      - spectral_radius: (M,)
      - is_stable: (M,) boolean
      - labels: (M,) string in {"stable", "marginal", "unstable"}
    """
    if fixed_points is None:
        return {
            "spectral_radius": np.array([]),
            "is_stable": np.array([], dtype=bool),
            "labels": np.array([], dtype=object),
        }

    if isinstance(fixed_points, np.ndarray):
        if fixed_points.size == 0:
            return {
                "spectral_radius": np.array([]),
                "is_stable": np.array([], dtype=bool),
                "labels": np.array([], dtype=object),
            }
        fps = torch.from_numpy(fixed_points).to(model.W_rec.device, dtype=model.W_rec.dtype)
    else:
        fps = fixed_points.to(model.W_rec.device, dtype=model.W_rec.dtype)
        if fps.numel() == 0:
            return {
                "spectral_radius": np.array([]),
                "is_stable": np.array([], dtype=bool),
                "labels": np.array([], dtype=object),
            }

    if fps.dim() == 1:
        fps = fps.unsqueeze(0)

    if alpha is None:
        alpha = float(getattr(model, "alpha", 1.0))

    W = model.W_rec.detach()
    b = model.b_rec.detach()

    N, H = fps.shape
    I = torch.eye(H, device=fps.device, dtype=fps.dtype).unsqueeze(0).expand(N, -1, -1)

    a = fps @ W.t() + b
    d = 1.0 - torch.tanh(a) ** 2
    JF = d.unsqueeze(2) * W.unsqueeze(0)
    J = (1.0 - alpha) * I + alpha * JF

    eigvals = torch.linalg.eigvals(J)
    rho = torch.abs(eigvals).amax(dim=1)

    rho_np = rho.cpu().numpy()
    labels = np.full(rho_np.shape, "unstable", dtype=object)
    labels[rho_np < (1.0 - marginal_eps)] = "stable"
    labels[np.abs(rho_np - 1.0) <= marginal_eps] = "marginal"

    return {
        "spectral_radius": rho_np,
        "is_stable": rho_np < 1.0,
        "labels": labels,
    }
