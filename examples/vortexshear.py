"""
Comparison between QG and SW solutions in vortex shear instability.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from helmholtz import compute_laplace_dstI, dstI2D
from qg import QG
from sw import SW


def parse_args():
    parser = argparse.ArgumentParser(
        description='Vortex shear instability: compare QG and SW solutions'
    )
    
    # Grid parameters
    parser.add_argument('--nx', type=int, default=192, help='Number of grid points in x')
    parser.add_argument('--ny', type=int, default=192, help='Number of grid points in y')
    parser.add_argument('--L', type=float, default=100e3, help='Domain size (m)')
    
    # Physical parameters
    parser.add_argument('--Bu', type=float, default=1.0, help='Burger number')
    parser.add_argument('--Ro', type=float, default=0.1, help='Rossby number')
    parser.add_argument('--flip-sign', action='store_true', help='Flip sign of f0')
    
    # Simulation parameters
    parser.add_argument('--n-tau', type=float, default=8.0, help='Number of tau periods to simulate')
    parser.add_argument('--compile', action='store_true', default=False, help='Enable torch.compile')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float64'], help='Data type for tensors')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='run_outputs/vortexshear',
                        help='Output directory')
    parser.add_argument('--no-plot', action='store_true', help='Disable interactive plotting')
    parser.add_argument('--save-images', action='store_true', 
                        help='Save plots as images to output_dir/images')
    parser.add_argument('--save-video', action='store_true',
                        help='Save plots as MP4 video to output_dir')
    parser.add_argument('--fps', type=int, default=10, help='FPS for video output')
    parser.add_argument('--freq-plot', type=int, default=100,
                        help='Plot frequency (in timesteps, 0 to disable)')
    
    return parser.parse_args()


def grad_perp(f, dx, dy):
    """Orthogonal gradient"""
    return (f[...,:-1] - f[...,1:]) / dy, (f[...,1:,:] - f[...,:-1,:]) / dx


def main():
    args = parse_args()
    
    torch.backends.cudnn.deterministic = True
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Dtype setup
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    
    print(f'Using device: {device}, dtype: {args.dtype}')
    
    # Grid
    nx, ny = args.nx, args.ny
    nl = 1
    Lx = Ly = args.L
    dx = torch.tensor(Lx / nx, device=device, dtype=dtype)
    dy = torch.tensor(Ly / ny, device=device, dtype=dtype)
    x_, y_ = torch.linspace(-Lx/2, Lx/2, nx+1, dtype=dtype, device=device), \
             torch.linspace(-Ly/2, Ly/2, ny+1, dtype=dtype, device=device)
    x, y = torch.meshgrid(x_, y_, indexing='ij')
    
    xc_ = 0.5 * (x_[1:] + x_[:-1])
    yc_ = 0.5 * (y_[1:] + y_[:-1])
    xc, yc = torch.meshgrid(xc_, yc_, indexing='ij')
    rc = torch.sqrt(xc**2 + yc**2)
    
    # Mask (rectangular domain by default)
    mask = torch.ones_like(xc)
    
    # Layer parameters
    H = torch.zeros(nl,1,1, dtype=dtype, device=device)
    H[0,0,0] = 1000  # 1km
    
    rho = 1000.
    g_prime = torch.zeros(nl,1,1, dtype=dtype, device=device)
    g_prime[0,0,0] = 10.
    
    Bu, Ro = args.Bu, args.Ro
    print(f'Ro={Ro} Bu={Bu}')
    
    r0, r1, r2 = 0.1*Lx, 0.1*Lx, 0.14*Lx
    
    # Set coriolis with burger number
    f0 = torch.sqrt(g_prime[0,0,0] * H[0,0,0] / Bu / r0**2)
    if args.flip_sign:
        f0 *= -1
    beta = 0
    f = f0 + beta * (y - Ly/2)
    
    z = x + 1j*y
    theta = torch.angle(z)
    r = torch.sqrt(x**2 + y**2)
    
    # Create rankine vortex with tripolar perturbation
    epsilon = 1e-3
    r *= (1+epsilon*torch.cos(theta*3))
    soft_step = lambda x: torch.sigmoid(x/100)
    mask_core = soft_step(r0 - r)
    mask_ring = soft_step(r-r1) * soft_step(r2-r)
    vor = 1. * (-mask_core / mask_core.mean() + mask_ring / mask_ring.mean())
    if args.flip_sign:
        vor *= -1
    laplace_dstI = compute_laplace_dstI(nx, ny, dx, dy, {'device': device, 'dtype': dtype})
    psi_hat = dstI2D(vor[1:-1,1:-1]) / laplace_dstI
    psi = F.pad(dstI2D(psi_hat), (1,1,1,1)).unsqueeze(0).unsqueeze(0)
    
    # Set psi amplitude to have correct Rossby number
    u, v = grad_perp(psi, dx, dy)
    u_norm_max = max(torch.abs(u).max(), torch.abs(v).max())
    psi *= Ro * f0 * r0 / u_norm_max
    p_init = psi * f0
    
    # Wind forcing, bottom drag
    taux = 0.
    tauy = 0.
    bottom_drag_coef = 0.
    
    param_qg = {
        'nx': nx, 'ny': ny, 'nl': nl,
        'dx': dx, 'dy': dy,
        'H': H, 'g_prime': g_prime, 'f': f,
        'taux': taux, 'tauy': tauy,
        'bottom_drag_coef': bottom_drag_coef,
        'device': device, 'dtype': dtype,
        'mask': mask,
        'compile': args.compile,
        'slip_coef': 1.,
        'dt': 0.,
    }
    
    param_sw = {
        'nx': nx, 'ny': ny, 'nl': nl,
        'dx': dx, 'dy': dy,
        'H': H, 'rho': rho, 'g_prime': g_prime, 'f': f,
        'taux': taux, 'tauy': tauy,
        'mask': mask,
        'bottom_drag_coef': bottom_drag_coef,
        'device': device, 'dtype': dtype,
        'slip_coef': 1,
        'compile': args.compile,
        'barotropic_filter': False,
        'barotropic_filter_spectral': False,
        'dt': 0.,
    }
    
    qg_multilayer = QG(param_qg)
    u_init, v_init, h_init = qg_multilayer.G(p_init)
    
    u_max, v_max, c = torch.abs(u_init).max().item() / dx, \
                    torch.abs(v_init).max().item() / dy, \
                    torch.sqrt(g_prime[0,0,0] * H.sum())
    print(f'u_max {u_max:.2e}, v_max {v_max:.2e}, c {c:.2e}')
    cfl_adv = 0.5
    cfl_gravity = 5 if param_sw['barotropic_filter'] else 0.5
    dt = min(cfl_adv * dx / u_max, cfl_adv * dy / v_max, cfl_gravity * dx / c)
    
    qg_multilayer.dt = dt
    qg_multilayer.u = torch.clone(u_init)
    qg_multilayer.v = torch.clone(v_init)
    qg_multilayer.h = torch.clone(h_init)
    qg_multilayer.compute_diagnostic_variables()
    
    # Initialize SW
    param_sw['dt'] = dt
    sw_multilayer = SW(param_sw)
    sw_multilayer.u = torch.clone(u_init)
    sw_multilayer.v = torch.clone(v_init)
    sw_multilayer.h = torch.clone(h_init)
    sw_multilayer.compute_diagnostic_variables()
    
    # Time parameters
    t = 0
    qg_multilayer.compute_time_derivatives()
    
    w_0 = qg_multilayer.omega.squeeze() / qg_multilayer.dx / qg_multilayer.dy
    tau = 1. / torch.sqrt(w_0.pow(2).mean()).cpu().item()
    print(f'tau = {tau *f0:.2f} f0-1')
    
    t_end = args.n_tau * tau
    freq_checknan = 100
    freq_log = int(t_end / 10 / dt) + 1
    n_steps = int(t_end / dt) + 1
    
    # Determine actual plot frequency
    if args.freq_plot > 0:
        freq_plot = args.freq_plot
    else:
        freq_plot = int(t_end / 100 / dt) + 1 if not args.no_plot or args.save_images or args.save_video else 0
    
    # Setup output directory
    if args.save_images or args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.save_images:
            image_dir = os.path.join(args.output_dir, 'images')
            os.makedirs(image_dir, exist_ok=True)
            print(f'Images will be saved to {image_dir}')
    
    # Setup plotting
    plot_enabled = (not args.no_plot) or args.save_images or args.save_video
    if plot_enabled and freq_plot > 0:
        import matplotlib
        import matplotlib.pyplot as plt
        
        if args.no_plot:
            matplotlib.use('Agg')  # Non-interactive backend
        else:
            matplotlib.rcParams.update({'font.size': 18})
            plt.ion()
        
        palette = plt.cm.bwr
        fig, axes = plt.subplots(1, 3, figsize=(18,8))
        axes[0].set_title(r'$\omega_{qg}$')
        axes[1].set_title(r'$\omega_{sw}$')
        axes[2].set_title(r'$\omega_{qg} - \omega_{sw}$')
        [(axes[i].set_xticks([]), axes[i].set_yticks([])) for i in range(3)]
        fig.tight_layout()
        
        if not args.no_plot:
            plt.pause(0.1)
        
        # Video writer setup
        if args.save_video:
            from matplotlib.animation import FFMpegWriter
            video_path = os.path.join(args.output_dir, f'vortexshear_Ro{Ro}_Bu{Bu}.mp4')
            writer = FFMpegWriter(fps=args.fps, metadata={'artist': 'pyQGSW'})
            writer.setup(fig, video_path, dpi=100)
            print(f'Video will be saved to {video_path}')
    
    # Main loop
    print(f'Starting simulation for {n_steps} steps...')
    for n in range(0, n_steps+1):
        
        if plot_enabled and freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
            mask_w = sw_multilayer.masks.not_w[0,0].cpu().numpy()
            w_qg = (qg_multilayer.omega / qg_multilayer.area / qg_multilayer.f0).cpu().numpy()
            w_sw = (sw_multilayer.omega / sw_multilayer.area / sw_multilayer.f0).cpu().numpy()
            wM = max(np.abs(w_qg).max(), np.abs(w_sw).max())
            
            kwargs = dict(cmap=palette, origin='lower', vmin=-wM, vmax=wM)
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            axes[0].imshow(np.ma.masked_where(mask_w, w_qg[0,0]).T, **kwargs)
            axes[1].imshow(np.ma.masked_where(mask_w, w_sw[0,0]).T, **kwargs)
            axes[2].imshow(np.ma.masked_where(mask_w, (w_qg - w_sw)[0,0]).T, **kwargs)
            axes[0].set_title(r'$\omega_{qg}$')
            axes[1].set_title(r'$\omega_{sw}$')
            axes[2].set_title(r'$\omega_{qg} - \omega_{sw}$')
            [(axes[i].set_xticks([]), axes[i].set_yticks([])) for i in range(3)]
            fig.suptitle(rf'Ro={Ro:.2f}, Bu={Bu:.2f}, t={t/tau:.2f}$\tau$, '
                       f'{"neg." if args.flip_sign else "pos"} $f_0$')
            
            if args.save_images:
                img_path = os.path.join(image_dir, f'frame_{n:06d}.png')
                fig.savefig(img_path, dpi=100, bbox_inches='tight')
            
            if args.save_video:
                writer.grab_frame()
            
            if not args.no_plot:
                plt.pause(0.05)
        
        if n < n_steps:  # Don't step after last plot
            qg_multilayer.step()
            sw_multilayer.step()
            t += dt
        
        if n % freq_checknan == 0:
            if torch.isnan(qg_multilayer.h).any():
                raise ValueError(f'Stopping, NAN number in QG h at iteration {n}.')
            if torch.isnan(sw_multilayer.h).any():
                raise ValueError(f'Stopping, NAN number in SW h at iteration {n}.')
        
        if freq_log > 0 and n % freq_log == 0:
            print(f'n={n:05d}/{n_steps}, {qg_multilayer.get_print_info()}')
    
    # Cleanup
    if args.save_video:
        writer.finish()
        print(f'Video saved to {video_path}')
    
    if not args.no_plot and plot_enabled:
        print('Simulation complete. Close the plot window to exit.')
        plt.ioff()
        plt.show()
    
    print('Done!')


if __name__ == '__main__':
    main()
