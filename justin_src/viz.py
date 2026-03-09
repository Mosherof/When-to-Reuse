import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from fixed_points import find_fixed_points_KE_min, find_fixed_points_hybrid, find_slow_points

def fit_pca(R_list):
    pca = PCA(n_components=2)
    pca.fit(R_list)
    return pca

def animate_R(R_list, interval=50, fixed_points=None, save_path=None, fps=20, stride=1, dpi=160, show=False, pca=None, lim=None, title=None):
    """
    Animate the neural activity R trajectory over epochs and time for a given batch index.
    Uses PCA to reduce dimensionality to 2D and animates the trajectory.
    
    Parameters:
    -----------
    R_list : list of torch.Tensor
        List of neural activity tensors of shape (num_samples, T+1, hidden_size)
    batch_idx : int
        Index of the batch to visualize
    interval : int
        Delay between frames in milliseconds
    save_path : str or None
        If provided, saves the animation to this path (e.g., .mp4 or .gif)
    fps : int
        Frames per second for saving the animation
    stride : int
        Subsample frames by this factor to reduce save time/size (default 1)
    dpi : int
        DPI for saving (lower = faster, default 80)
    show : bool
        Whether to display the animation window (default False)
    """
    
    # Stack all epochs and flatten over the epoch dimension
    #R_stacked = torch.cat([R[batch_idx] for R in R_list], dim=0)  # Shape: (epoch * (T+1), hidden_size)
    #r_batch = R_list.detach().numpy()
    r_batch = R_list
    
    print("r_batch shape: ", np.array(r_batch).shape)
    
    # Apply PCA to reduce to 2D
    if pca is None:
        pca = PCA(n_components=2)
        r_2d = pca.fit_transform(r_batch)
    else:
        r_2d = pca.transform(r_batch)
        
    fp_2d = None
    if fixed_points is not None:
        if isinstance(fixed_points, torch.Tensor):
            fixed_points = fixed_points.detach().cpu().numpy()
        if isinstance(fixed_points, np.ndarray) and fixed_points.ndim == 2 and fixed_points.shape[0] > 0:
            fp_2d = pca.transform(fixed_points)
    
    # Subsample frames to speed up encoding
    if stride > 1:
        r_2d = r_2d[::stride]
    
    print(f"Generating animation with {len(r_2d)} frames...")
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set limits
    if lim is None:
        lim = (r_2d[:, 0].min() - 0.5, r_2d[:, 0].max() + 0.5, r_2d[:, 1].min() - 0.5, r_2d[:, 1].max() + 0.5)
    
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    
    if fp_2d is not None:
        ax.scatter(fp_2d[:, 0], fp_2d[:, 1], c='red', marker='X', s=100, label='Fixed Points')
    
    # Initialize scatter plot for all points
    scatter = ax.scatter([], [], c=[], cmap='viridis', s=10, alpha=0.6)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title(f'Neural Activity Trajectory Animation' if title is None else title)
    ax.grid(True, alpha=0.3)
    
    def update(frame):
        # Update scatter plot with points up to current frame
        scatter.set_offsets(r_2d[:frame+1, :])
        scatter.set_array(np.arange(frame+1))
        
        return scatter,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(r_2d), interval=interval, 
                        blit=True, repeat=False)
    
    plt.tight_layout()
    
    if save_path:
        # Cap fps for stability and choose writer
        save_fps = min(fps, 60)
        if save_path.lower().endswith(".gif"):
            writer = PillowWriter(fps=save_fps)
        else:
            writer = FFMpegWriter(fps=save_fps, bitrate=1800, codec='libx264', 
                                 extra_args=['-preset', 'fast'])
        
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer=writer, dpi=dpi)
        print(f"Animation saved successfully!")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return pca, lim


def find_fixed_points_over_epochs(rnn, train_R_list, stride=1, weight_snapshots=None,
                                  fp_finder=None, fp_kwargs=None,
                                  n_random_seeds=200, n_carry_forward=3):
    """
    Find fixed points at each sampled epoch over training.

    Parameters:
    -----------
    rnn : RNN
        The trained RNN model
    train_R_list : list
        List of hidden state tensors, one per epoch
    stride : int
        Subsample epochs by this factor
    weight_snapshots : list of dict or None
        Per-epoch weight snapshots from train_rnn. Each dict has 'W_rec' and 'b_rec'.
        If None, uses rnn's current (final) weights for all epochs.
    fp_finder : callable or None
        Function to find fixed points. Default is find_fixed_points_hybrid.
    fp_kwargs : dict or None
        Extra kwargs passed to fp_finder
    n_random_seeds : int
        Number of random initial seeds to include (default 200)
    n_carry_forward : int
        Number of previous epochs' FPs to carry forward as seeds (default 3)

    Returns:
    --------
    fps_per_epoch : list of np.ndarray
        Fixed points found at each sampled epoch. Each array has shape (n_fps, hidden_size).
    """
    if fp_finder is None:
        fp_finder = find_fixed_points_hybrid
    if fp_kwargs is None:
        fp_kwargs = {}

    if stride > 1:
        train_R_list = train_R_list[::stride]
        if weight_snapshots is not None:
            weight_snapshots = weight_snapshots[::stride]

    n_epochs = len(train_R_list)
    fps_per_epoch = []
    recent_fps = []

    print(f"Finding fixed points for {n_epochs} sampled epochs (stride={stride})...")
    for i in range(n_epochs):
        # Temporarily swap in per-epoch weights if available
        if weight_snapshots is not None:
            orig_W_rec = rnn.W_rec.data.clone()
            orig_b_rec = rnn.b_rec.data.clone()
            rnn.W_rec.data = weight_snapshots[i]['W_rec'].to(rnn.W_rec.device)
            rnn.b_rec.data = weight_snapshots[i]['b_rec'].to(rnn.b_rec.device)

        # Gather seeds: current + neighboring epochs + random seeds
        start = max(0, i - 2)
        initial_states = torch.cat(train_R_list[start:i+1], dim=0)
        if n_random_seeds > 0:
            random_seeds = torch.randn(n_random_seeds, rnn.hidden_size) * 0.5
            initial_states = torch.cat([initial_states, random_seeds], dim=0)

        # Include fixed points from previous epochs as seeds
        for prev_fps in recent_fps[-n_carry_forward:]:
            if prev_fps.shape[0] > 0:
                fps_tensor = torch.from_numpy(prev_fps).to(initial_states.dtype)
                initial_states = torch.cat([initial_states, fps_tensor], dim=0)

        fps_result = fp_finder(rnn, initial_states, **fp_kwargs)

        # Restore original weights
        if weight_snapshots is not None:
            rnn.W_rec.data = orig_W_rec
            rnn.b_rec.data = orig_b_rec

        # Convert to numpy array
        if isinstance(fps_result, list) and len(fps_result) > 0:
            fps_arr = torch.cat(fps_result, dim=0).detach().cpu().numpy()
        elif isinstance(fps_result, torch.Tensor):
            fps_arr = fps_result.detach().cpu().numpy()
        elif isinstance(fps_result, np.ndarray):
            fps_arr = fps_result
        else:
            fps_arr = np.empty((0, rnn.hidden_size))

        fps_per_epoch.append(fps_arr)
        recent_fps.append(fps_arr)
        if (i + 1) % 10 == 0:
            print(f"  Epoch {(i+1) * stride}/{n_epochs * stride}: {fps_arr.shape[0]} fixed points")

    return fps_per_epoch


def animate_fixed_points(rnn, train_R_list, pca=None, save_path=None, fps=20, dpi=160,
                         show=False, lim=None, title=None, interval=100,
                         fp_finder=None, fp_kwargs=None, stride=1, weight_snapshots=None):
    """
    Animate the trajectory of fixed points over training epochs.
    Finds fixed points at each epoch and animates how they move in PCA space.
    
    Parameters:
    -----------
    rnn : RNN
        The trained RNN model (used for weights in fixed point finding)
    train_R_list : list
        List of hidden state tensors, one per epoch
    pca : PCA or None
        Pre-fitted PCA object. If None, fits on all fixed points.
    save_path : str or None
        If provided, saves the animation to this path
    fps : int
        Frames per second for saving
    dpi : int
        DPI for saving
    show : bool
        Whether to display the animation
    lim : tuple or None
        (xmin, xmax, ymin, ymax) axis limits
    title : str or None
        Plot title
    interval : int
        Delay between frames in milliseconds
    fp_finder : callable or None
        Function to find fixed points. Default is find_fixed_points_KE_min.
    fp_kwargs : dict or None
        Extra kwargs passed to fp_finder
    stride : int
        Subsample epochs by this factor
    weight_snapshots : list of dict or None
        Per-epoch weight snapshots from train_rnn. Each dict has 'W_rec' and 'b_rec'.
        If None, uses rnn's current (final) weights for all epochs.
    """

    fps_per_epoch = find_fixed_points_over_epochs(
        rnn, train_R_list, stride=stride, weight_snapshots=weight_snapshots,
        fp_finder=fp_finder, fp_kwargs=fp_kwargs,
    )
    n_epochs = len(fps_per_epoch)

    # Fit PCA on all fixed points combined
    if pca is None:
        all_fps = np.concatenate([fp for fp in fps_per_epoch if fp.shape[0] > 0], axis=0)
        pca = PCA(n_components=2)
        pca.fit(all_fps)

    # Project all fixed points to 2D
    fps_2d_per_epoch = []
    for fp in fps_per_epoch:
        if fp.shape[0] > 0:
            fps_2d_per_epoch.append(pca.transform(fp))
        else:
            fps_2d_per_epoch.append(np.empty((0, 2)))

    # Set axis limits
    if lim is None:
        all_2d = np.concatenate([fp for fp in fps_2d_per_epoch if fp.shape[0] > 0], axis=0)
        margin = 0.5
        lim = (all_2d[:, 0].min() - margin, all_2d[:, 0].max() + margin,
               all_2d[:, 1].min() - margin, all_2d[:, 1].max() + margin)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title(title or 'Fixed Point Trajectories Over Training')
    ax.grid(True, alpha=0.3)

    # Current fixed points (red X)
    scatter_current = ax.scatter([], [], c='red', marker='X', s=150, zorder=5, label='Current FPs')
    # Trail of past fixed points (fading)
    scatter_trail = ax.scatter([], [], c=[], cmap='coolwarm', s=30, alpha=0.3)
    epoch_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                         verticalalignment='top', fontweight='bold')
    ax.legend(loc='upper right')

    def update(frame):
        # Current epoch fixed points
        fp_now = fps_2d_per_epoch[frame]
        if fp_now.shape[0] > 0:
            scatter_current.set_offsets(fp_now)
        else:
            scatter_current.set_offsets(np.empty((0, 2)))

        # Trail: all previous fixed points
        trail_pts = []
        trail_colors = []
        for t in range(frame + 1):
            fp_t = fps_2d_per_epoch[t]
            if fp_t.shape[0] > 0:
                trail_pts.append(fp_t)
                trail_colors.extend([t] * fp_t.shape[0])

        if trail_pts:
            trail_all = np.concatenate(trail_pts, axis=0)
            scatter_trail.set_offsets(trail_all)
            scatter_trail.set_array(np.array(trail_colors, dtype=float))
            scatter_trail.set_clim(0, n_epochs)

        epoch_text.set_text(f'Epoch {(frame + 1) * stride}/{n_epochs * stride}')
        return scatter_current, scatter_trail, epoch_text

    print(f"Generating animation with {n_epochs} frames...")
    anim = FuncAnimation(fig, update, frames=n_epochs, interval=interval,
                         blit=True, repeat=False)

    plt.tight_layout()

    if save_path:
        save_fps = min(fps, 60)
        if save_path.lower().endswith(".gif"):
            writer = PillowWriter(fps=save_fps)
        else:
            writer = FFMpegWriter(fps=save_fps, bitrate=1800, codec='libx264',
                                  extra_args=['-preset', 'fast'])
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer=writer, dpi=dpi)
        print(f"Animation saved successfully!")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pca, lim


def plot_num_fixed_points(rnn, train_R_list, stride=1, weight_snapshots=None,
                          fp_finder=None, fp_kwargs=None, save_path=None,
                          title=None, show=True):
    """
    Plot the number of fixed points found at each epoch over training.

    Parameters:
    -----------
    rnn : RNN
        The trained RNN model
    train_R_list : list
        List of hidden state tensors, one per epoch
    stride : int
        Subsample epochs by this factor
    weight_snapshots : list of dict or None
        Per-epoch weight snapshots from train_rnn
    fp_finder : callable or None
        Fixed-point finder function. Default is find_fixed_points_KE_min.
    fp_kwargs : dict or None
        Extra kwargs passed to fp_finder
    save_path : str or None
        If provided, saves the figure to this path
    title : str or None
        Plot title
    show : bool
        Whether to display the plot
    """
    fps_per_epoch = find_fixed_points_over_epochs(
        rnn, train_R_list, stride=stride, weight_snapshots=weight_snapshots,
        fp_finder=fp_finder, fp_kwargs=fp_kwargs,
    )
    n_epochs = len(fps_per_epoch)
    epoch_indices = [(i + 1) * stride for i in range(n_epochs)]
    num_fps = [fp.shape[0] for fp in fps_per_epoch]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epoch_indices, num_fps, 'o-', markersize=4, linewidth=1.5, color='C0')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Number of Fixed Points', fontsize=12)
    ax.set_title(title or 'Number of Fixed Points Over Training', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return epoch_indices, num_fps


def find_slow_points_over_epochs(rnn, train_R_list, stride=1, weight_snapshots=None,
                                 n_random_seeds=2000, random_scale=1.0,
                                 slow_kwargs=None):
    """
    Find fixed points and slow points at each sampled epoch over training.

    Parameters:
    -----------
    rnn : RNN
        The trained RNN model
    train_R_list : list
        List of hidden state tensors, one per epoch
    stride : int
        Subsample epochs by this factor
    weight_snapshots : list of dict or None
        Per-epoch weight snapshots from train_rnn
    n_random_seeds : int
        Number of random initial seeds to include (default 2000)
    random_scale : float
        Scale for random seeds (default 1.0)
    slow_kwargs : dict or None
        Extra kwargs passed to find_slow_points

    Returns:
    --------
    fps_per_epoch : list of np.ndarray
        Fixed points at each sampled epoch
    slow_per_epoch : list of np.ndarray
        Slow points at each sampled epoch
    epoch_indices : list of int
        The actual epoch indices sampled
    """
    if slow_kwargs is None:
        slow_kwargs = {}

    if stride > 1:
        sampled_indices = list(range(0, len(train_R_list), stride))
        if sampled_indices[-1] != len(train_R_list) - 1:
            sampled_indices.append(len(train_R_list) - 1)
    else:
        sampled_indices = list(range(len(train_R_list)))

    fps_per_epoch = []
    slow_per_epoch = []

    print(f"Finding slow points at {len(sampled_indices)} sampled epochs (stride={stride})...")
    for ep in sampled_indices:
        if weight_snapshots is not None:
            orig_W_rec = rnn.W_rec.data.clone()
            orig_b_rec = rnn.b_rec.data.clone()
            rnn.W_rec.data = weight_snapshots[ep]['W_rec'].to(rnn.W_rec.device)
            rnn.b_rec.data = weight_snapshots[ep]['b_rec'].to(rnn.b_rec.device)

        seeds = train_R_list[ep]
        if n_random_seeds > 0:
            random_seeds = torch.randn(n_random_seeds, rnn.hidden_size) * random_scale
            seeds = torch.cat([seeds, random_seeds], dim=0)

        fps_np, slow_np = find_slow_points(rnn, seeds, **slow_kwargs)

        if weight_snapshots is not None:
            rnn.W_rec.data = orig_W_rec
            rnn.b_rec.data = orig_b_rec

        fps_per_epoch.append(fps_np)
        slow_per_epoch.append(slow_np)
        print(f"  Epoch {ep:>4d}: {fps_np.shape[0]} FPs, {slow_np.shape[0]} slow points")

    return fps_per_epoch, slow_per_epoch, sampled_indices


def plot_slow_points_state_space(fps_per_epoch, slow_per_epoch, epoch_indices,
                                 pca=None, save_path=None, show=True, title=None):
    """
    Tiled panel plot of slow points and fixed points in PCA state space over epochs.

    Parameters:
    -----------
    fps_per_epoch : list of np.ndarray
        Fixed points at each epoch
    slow_per_epoch : list of np.ndarray
        Slow points at each epoch
    epoch_indices : list of int
        Epoch labels for each panel
    pca : PCA or None
        Pre-fitted PCA. If None, fits on all points.
    save_path : str or None
        If provided, saves figure
    show : bool
        Whether to display
    title : str or None
        Suptitle

    Returns:
    --------
    pca : PCA
        The fitted PCA object
    """
    # Fit PCA on all found points
    all_points_list = (
        [fp for fp in fps_per_epoch if fp.shape[0] > 0] +
        [sp for sp in slow_per_epoch if sp.shape[0] > 0]
    )
    if not all_points_list:
        print("No points found at any epoch.")
        return pca

    all_points = np.concatenate(all_points_list, axis=0)
    if pca is None:
        pca = PCA(n_components=2)
        pca.fit(all_points)

    # Project each epoch
    fps_2d = [pca.transform(fp) if fp.shape[0] > 0 else np.empty((0, 2)) for fp in fps_per_epoch]
    slow_2d = [pca.transform(sp) if sp.shape[0] > 0 else np.empty((0, 2)) for sp in slow_per_epoch]

    # Layout
    n_panels = len(epoch_indices)
    ncols = min(6, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    # Axis limits
    all_2d = np.concatenate([a for a in fps_2d + slow_2d if a.shape[0] > 0], axis=0)
    margin = 0.5
    xlim = (all_2d[:, 0].min() - margin, all_2d[:, 0].max() + margin)
    ylim = (all_2d[:, 1].min() - margin, all_2d[:, 1].max() + margin)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for i, (ep, fp2, sp2) in enumerate(zip(epoch_indices, fps_2d, slow_2d)):
        ax = axes[i // ncols][i % ncols]
        if sp2.shape[0] > 0:
            ax.scatter(sp2[:, 0], sp2[:, 1], c='dodgerblue', s=15, alpha=0.5, label='Slow pts')
        if fp2.shape[0] > 0:
            ax.scatter(fp2[:, 0], fp2[:, 1], c='red', marker='X', s=100, zorder=5, label='FPs')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'Epoch {ep}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for i in range(n_panels, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.supxlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    fig.supylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    fig.suptitle(title or 'Slow Points & FPs in State Space', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pca


def animate_slow_points_state_space(fps_per_epoch, slow_per_epoch, epoch_indices,
                                    pca=None, save_path=None, show=False, title=None,
                                    fps=5, interval=200, figsize=(7, 6)):
    """
    Animated version of plot_slow_points_state_space: one frame per epoch in PCA space.

    Parameters:
    -----------
    fps_per_epoch : list of np.ndarray
        Fixed points at each epoch
    slow_per_epoch : list of np.ndarray
        Slow points at each epoch
    epoch_indices : list of int
        Epoch labels for each frame
    pca : PCA or None
        Pre-fitted 2-component PCA. If None, fits on all points.
    save_path : str or None
        If provided, saves the animation (.mp4 or .gif)
    show : bool
        Whether to display the animation (default False)
    title : str or None
        Base title for the plot
    fps : int
        Frames per second for saved animation (default 5)
    interval : int
        Milliseconds between frames when displaying (default 200)
    figsize : tuple
        Figure size (default (7, 6))

    Returns:
    --------
    pca : PCA
        The fitted PCA object
    """
    # Fit PCA on all found points
    all_points_list = (
        [fp for fp in fps_per_epoch if fp.shape[0] > 0] +
        [sp for sp in slow_per_epoch if sp.shape[0] > 0]
    )
    if not all_points_list:
        print("No points found at any epoch.")
        return pca

    all_points = np.concatenate(all_points_list, axis=0)
    if pca is None:
        pca = PCA(n_components=2)
        pca.fit(all_points)

    # Project each epoch
    fps_2d = [pca.transform(fp) if fp.shape[0] > 0 else np.empty((0, 2)) for fp in fps_per_epoch]
    slow_2d = [pca.transform(sp) if sp.shape[0] > 0 else np.empty((0, 2)) for sp in slow_per_epoch]

    # Axis limits (fixed across all frames)
    all_2d = np.concatenate([a for a in fps_2d + slow_2d if a.shape[0] > 0], axis=0)
    margin = 0.5
    xlim = (all_2d[:, 0].min() - margin, all_2d[:, 0].max() + margin)
    ylim = (all_2d[:, 1].min() - margin, all_2d[:, 1].max() + margin)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.grid(True, alpha=0.3)

    slow_scatter = ax.scatter([], [], c='dodgerblue', s=15, alpha=0.5, label='Slow pts')
    fp_scatter = ax.scatter([], [], c='red', marker='X', s=100, zorder=5, label='FPs')
    ax.legend(fontsize=9, loc='upper right')

    base_title = title or 'Slow Points & FPs in State Space'
    epoch_text = ax.set_title(f'{base_title} — Epoch {epoch_indices[0]}',
                              fontsize=13, fontweight='bold')

    def update(frame_idx):
        ep = epoch_indices[frame_idx]
        sp2 = slow_2d[frame_idx]
        fp2 = fps_2d[frame_idx]

        slow_scatter.set_offsets(sp2 if sp2.shape[0] > 0 else np.empty((0, 2)))
        fp_scatter.set_offsets(fp2 if fp2.shape[0] > 0 else np.empty((0, 2)))
        ax.set_title(f'{base_title} — Epoch {ep}', fontsize=13, fontweight='bold')
        return slow_scatter, fp_scatter

    n_frames = len(epoch_indices)
    print(f"Generating animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps, metadata={'title': base_title})
        anim.save(save_path, writer=writer, dpi=160)
        print(f"Animation saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pca


def plot_slow_points_state_space_3d(fps_per_epoch, slow_per_epoch, epoch_indices,
                                    pca=None, save_path=None, show=True, title=None,
                                    elev=25, azim=135):
    """
    Tiled 3D panel plot of slow points and fixed points in PCA state space over epochs.

    Parameters:
    -----------
    fps_per_epoch : list of np.ndarray
        Fixed points at each epoch
    slow_per_epoch : list of np.ndarray
        Slow points at each epoch
    epoch_indices : list of int
        Epoch labels for each panel
    pca : PCA or None
        Pre-fitted 3-component PCA. If None, fits on all points.
    save_path : str or None
        If provided, saves figure
    show : bool
        Whether to display
    title : str or None
        Suptitle

    Returns:
    --------
    pca : PCA
        The fitted PCA object (n_components=3)
    """
    all_points_list = (
        [fp for fp in fps_per_epoch if fp.shape[0] > 0] +
        [sp for sp in slow_per_epoch if sp.shape[0] > 0]
    )
    if not all_points_list:
        print("No points found at any epoch.")
        return pca

    all_points = np.concatenate(all_points_list, axis=0)
    if pca is None:
        pca = PCA(n_components=3)
        pca.fit(all_points)

    fps_3d = [pca.transform(fp) if fp.shape[0] > 0 else np.empty((0, 3)) for fp in fps_per_epoch]
    slow_3d = [pca.transform(sp) if sp.shape[0] > 0 else np.empty((0, 3)) for sp in slow_per_epoch]

    n_panels = len(epoch_indices)
    ncols = min(6, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    all_3d = np.concatenate([a for a in fps_3d + slow_3d if a.shape[0] > 0], axis=0)
    margin = 0.5
    xlim = (all_3d[:, 0].min() - margin, all_3d[:, 0].max() + margin)
    ylim = (all_3d[:, 1].min() - margin, all_3d[:, 1].max() + margin)
    zlim = (all_3d[:, 2].min() - margin, all_3d[:, 2].max() + margin)

    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    for i, (ep, fp3, sp3) in enumerate(zip(epoch_indices, fps_3d, slow_3d)):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        if sp3.shape[0] > 0:
            ax.scatter(sp3[:, 0], sp3[:, 1], sp3[:, 2],
                       c='dodgerblue', s=10, alpha=0.4, label='Slow pts')
        if fp3.shape[0] > 0:
            ax.scatter(fp3[:, 0], fp3[:, 1], fp3[:, 2],
                       c='red', marker='X', s=120, zorder=5, label='FPs')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_title(f'Epoch {ep}', fontsize=10, fontweight='bold')
        ax.set_xlabel(f'PC1', fontsize=8)
        ax.set_ylabel(f'PC2', fontsize=8)
        ax.set_zlabel(f'PC3', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=elev, azim=azim)
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused panels
    for i in range(n_panels, nrows * ncols):
        ax_hide = fig.add_subplot(nrows, ncols, i + 1)
        ax_hide.set_visible(False)

    evr = pca.explained_variance_ratio_
    fig.suptitle(
        (title or 'Slow Points & FPs in 3D State Space') +
        f'\n(PC1: {evr[0]:.1%}, PC2: {evr[1]:.1%}, PC3: {evr[2]:.1%})',
        fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pca


def viz_R(R_list, batch_idx=0):
    """
    Visualize the neural activity R over epochs and time for a given batch index.
    Uses PCA to reduce dimensionality to 2D and plots the trajectory of neural activity.
    
    Parameters:
    -----------
    R_list : list of torch.Tensor
        List of neural activity tensors of shape (epoch, B, T+1, hidden_size)
    batch_idx : int
        Index of the batch to visualize
    """
    
    # Handle single tensor (not a list)
    if isinstance(R_list, torch.Tensor):
        R_list = [R_list]
    
    # Stack all epochs and flatten over the epoch dimension
    R_stacked = torch.cat([R[batch_idx] for R in R_list], dim=0)[:100]  # Shape: (epoch * (T+1), hidden_size)
    r_batch = R_stacked.detach().numpy()
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    r_2d = pca.fit_transform(r_batch)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color points by time (index)
    time_indices = np.arange(len(r_2d))
    scatter = ax.scatter(r_2d[:, 0], r_2d[:, 1], c=time_indices, cmap='viridis', 
                         alpha=0.7, s=20)
    ax.plot(r_2d[0, 0], r_2d[0, 1], 'go', markersize=10, label='Start')
    ax.plot(r_2d[-1, 0], r_2d[-1, 1], 'ro', markersize=10, label='End')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time', fontsize=10)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title(f'Neural Activity Trajectory (Batch {batch_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    
    return pca

def plot_example_trial(task, seed=None):
    """
    Plot an example trial showing input train and output train for the flip-flop task.
    
    Parameters:
    -----------
    task : FlipFlopTask
        Task generator
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate a single trial
    original_batch_size = task.batch_size
    task.batch_size = 1
    U, O = task.generate()
    task.batch_size = original_batch_size
    
    # Convert to numpy
    U_np = U[0].numpy()
    O_np = O[0].numpy()
    time = np.arange(task.T)
    
    # Create figure with 4 subplots (2 inputs, 2 outputs)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    colors = ['#E63946', '#457B9D']  # Red for channel 1, Blue for channel 2
    
    # Plot Input Channel 1
    ax = axes[0]
    flip_times_1 = time[U_np[:, 0] != 0]
    flip_values_1 = U_np[U_np[:, 0] != 0, 0]
    markerline, stemlines, baseline = ax.stem(flip_times_1, flip_values_1, 
                                               linefmt=colors[0], markerfmt='o', basefmt='k-')
    plt.setp(stemlines, linewidth=2)
    plt.setp(markerline, markersize=8, color=colors[0])
    ax.set_ylabel('Input u₁', fontsize=12, fontweight='bold')
    ax.set_ylim([-1.5, 1.5])
    ax.set_yticks([-1, 0, 1])
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_title('Input Train - Channel 1', fontsize=12, color=colors[0])
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot Input Channel 2
    ax = axes[1]
    flip_times_2 = time[U_np[:, 1] != 0]
    flip_values_2 = U_np[U_np[:, 1] != 0, 1]
    markerline, stemlines, baseline = ax.stem(flip_times_2, flip_values_2, 
                                               linefmt=colors[1], markerfmt='o', basefmt='k-')
    plt.setp(stemlines, linewidth=2)
    plt.setp(markerline, markersize=8, color=colors[1])
    ax.set_ylabel('Input u₂', fontsize=12, fontweight='bold')
    ax.set_ylim([-1.5, 1.5])
    ax.set_yticks([-1, 0, 1])
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_title('Input Train - Channel 2', fontsize=12, color=colors[1])
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot Output Channel 1
    ax = axes[2]
    ax.step(time, O_np[:, 0], where='post', linewidth=2.5, color=colors[0])
    ax.fill_between(time, 0, O_np[:, 0], step='post', alpha=0.3, color=colors[0])
    ax.set_ylabel('Output o₁', fontsize=12, fontweight='bold')
    ax.set_ylim([-1.5, 1.5])
    ax.set_yticks([-1, 0, 1])
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_title('Output Train - Channel 1 (Memory State)', fontsize=12, color=colors[0])
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot Output Channel 2
    ax = axes[3]
    ax.step(time, O_np[:, 1], where='post', linewidth=2.5, color=colors[1])
    ax.fill_between(time, 0, O_np[:, 1], step='post', alpha=0.3, color=colors[1])
    ax.set_ylabel('Output o₂', fontsize=12, fontweight='bold')
    ax.set_ylim([-1.5, 1.5])
    ax.set_yticks([-1, 0, 1])
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_title('Output Train - Channel 2 (Memory State)', fontsize=12, color=colors[1])
    ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xlim([0, task.T])
    plt.suptitle('2-Bit Flip-Flop Task: Example Trial\n(Input pulses trigger memory state changes)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    return fig


def plot_results(rnn, task, losses):
    """
    Plot training results including loss curve, predicted vs target outputs,
    and neural activity raster plot.
    """
    # Generate a test trial
    task.batch_size = 1
    task.T = 500
    U, O_target = task.generate()
    
    with torch.no_grad():
        R, O_pred = rnn(U)
    
    # Convert to numpy
    U_np = U[0].numpy()
    O_target_np = O_target[0].numpy()
    O_pred_np = O_pred[0].numpy()
    R_np = R[0].numpy()
    
    time = np.arange(500)
    time_r = np.arange(501)
    
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)
    
    # Loss curve
    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(losses, linewidth=1.5, color='C0')
    ax_loss.set_xlabel('Epoch', fontsize=11)
    ax_loss.set_ylabel('Loss', fontsize=11)
    ax_loss.set_title('Training Loss Curve', fontsize=12, fontweight='bold')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_yscale('log')
    
    # Inputs
    ax_input1 = fig.add_subplot(gs[1, 0])
    flip_times_1 = time[U_np[:, 0] != 0]
    flip_values_1 = U_np[U_np[:, 0] != 0, 0]
    ax_input1.stem(flip_times_1, flip_values_1, linefmt='C0-', 
                   markerfmt='C0o', basefmt='k-')
    ax_input1.set_ylabel('Input u₁', fontsize=11)
    ax_input1.set_ylim([-1.5, 1.5])
    ax_input1.set_xlim([0, 500])
    ax_input1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax_input1.grid(True, alpha=0.3)
    ax_input1.set_title('Input Channel 1', fontsize=11)
    
    ax_input2 = fig.add_subplot(gs[1, 1])
    flip_times_2 = time[U_np[:, 1] != 0]
    flip_values_2 = U_np[U_np[:, 1] != 0, 1]
    ax_input2.stem(flip_times_2, flip_values_2, linefmt='C1-', 
                   markerfmt='C1o', basefmt='k-')
    ax_input2.set_ylabel('Input u₂', fontsize=11)
    ax_input2.set_ylim([-1.5, 1.5])
    ax_input2.set_xlim([0, 500])
    ax_input2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax_input2.grid(True, alpha=0.3)
    ax_input2.set_title('Input Channel 2', fontsize=11)
    
    # Predicted vs Target Outputs - Channel 1
    ax_out1 = fig.add_subplot(gs[2, 0])
    ax_out1.plot(time, O_target_np[:, 0], 'k-', linewidth=2.5, 
                 label='Target', alpha=0.7)
    ax_out1.plot(time, O_pred_np[:, 0], 'C0--', linewidth=2, 
                 label='Predicted')
    ax_out1.set_ylabel('Output o₁', fontsize=11)
    ax_out1.set_ylim([-1.5, 1.5])
    ax_out1.set_xlim([0, 500])
    ax_out1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax_out1.legend(fontsize=9)
    ax_out1.grid(True, alpha=0.3)
    ax_out1.set_title('Output Channel 1', fontsize=11)
    
    # Predicted vs Target Outputs - Channel 2
    ax_out2 = fig.add_subplot(gs[2, 1])
    ax_out2.plot(time, O_target_np[:, 1], 'k-', linewidth=2.5, 
                 label='Target', alpha=0.7)
    ax_out2.plot(time, O_pred_np[:, 1], 'C1--', linewidth=2, 
                 label='Predicted')
    ax_out2.set_ylabel('Output o₂', fontsize=11)
    ax_out2.set_ylim([-1.5, 1.5])
    ax_out2.set_xlim([0, 500])
    ax_out2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax_out2.legend(fontsize=9)
    ax_out2.grid(True, alpha=0.3)
    ax_out2.set_title('Output Channel 2', fontsize=11)
    
    # Neural activity raster plot
    ax_raster = fig.add_subplot(gs[3:, :])
    
    # Create raster plot by showing activity of all neurons
    im = ax_raster.imshow(R_np.T, aspect='auto', cmap='RdBu_r', 
                          interpolation='nearest', vmin=-1, vmax=1,
                          extent=[0, 101, 0, rnn.hidden_size])
    ax_raster.set_xlabel('Time (ms)', fontsize=11)
    ax_raster.set_ylabel('Neuron Index', fontsize=11)
    ax_raster.set_title('Neural Activity Raster Plot', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_raster)
    cbar.set_label('Activity', fontsize=10)
    
    plt.suptitle('RNN Performance on 2-Bit Flip Flop Task', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.show()
    
    return fig