"""
Double-gyre on beta plane with idealized wind forcing.
"""
import argparse
import os
import numpy as np
import torch

from sw import SW
from qg import QG


def parse_args():
    parser = argparse.ArgumentParser(
        description='Double-gyre configuration with idealized wind forcing'
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='qg', choices=['qg', 'sw'],
                        help='Model to use: QG or SW')
    
    # Grid parameters
    parser.add_argument('--nx', type=int, default=256, help='Number of grid points in x')
    parser.add_argument('--ny', type=int, default=256, help='Number of grid points in y')
    parser.add_argument('--Lx', type=float, default=5120e3, help='Domain size in x (m)')
    parser.add_argument('--Ly', type=float, default=5120e3, help='Domain size in y (m)')
    
    # Physical parameters
    parser.add_argument('--f0', type=float, default=9.375e-5, help='Mean Coriolis parameter (s^-1)')
    parser.add_argument('--beta', type=float, default=1.754e-11, 
                        help='Coriolis gradient (m^-1 s^-1)')
    parser.add_argument('--wind-mag', type=float, default=0.08,
                        help='Wind stress magnitude (Pa)')
    parser.add_argument('--slip-coef', type=float, default=1.0,
                        help='Slip coefficient (0=no-slip, 1=free-slip)')
    
    # Simulation parameters
    parser.add_argument('--dt', type=float, default=4000, help='Time step (s)')
    parser.add_argument('--n-years', type=float, default=10.0, help='Number of years to simulate')
    parser.add_argument('--compile', action='store_true', default=False, help='Enable torch.compile')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float64'], help='Data type for tensors')
    
    # I/O parameters
    parser.add_argument('--output-dir', type=str, default='',
                        help='Output directory (default: run_outputs/{model}_{nx}x{ny}_...)')
    parser.add_argument('--start-file', type=str, default='',
                        help='Restart file (npz format)')
    parser.add_argument('--freq-save', type=float, default=15.0,
                        help='Save frequency in days (0 to disable)')
    parser.add_argument('--save-after-years', type=float, default=2.0,
                        help='Start saving after this many years')
    
    # Plotting parameters
    parser.add_argument('--no-plot', action='store_true', help='Disable interactive plotting')
    parser.add_argument('--save-images', action='store_true',
                        help='Save plots as images to output_dir/images')
    parser.add_argument('--save-video', action='store_true',
                        help='Save plots as MP4 video to output_dir')
    parser.add_argument('--fps', type=int, default=10, help='FPS for video output')
    parser.add_argument('--freq-plot', type=float, default=15.0,
                        help='Plot frequency in days (0 to disable)')
    parser.add_argument('--layer', type=int, default=0, help='Layer to plot (0-indexed)')
    
    return parser.parse_args()


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
    Lx, Ly = args.Lx, args.Ly
    dx, dy = Lx / nx, Ly / ny
    
    mask = torch.ones(nx, ny, dtype=dtype, device=device)
    
    # Layer parameters
    H = torch.zeros(3,1,1, dtype=dtype, device=device)
    H[0,0,0] = 400.
    H[1,0,0] = 1100.
    H[2,0,0] = 2600.
    
    rho = 1000
    g_prime = torch.zeros(3,1,1, dtype=dtype, device=device)
    g_prime[0,0,0] = 9.81
    g_prime[1,0,0] = 0.025
    g_prime[2,0,0] = 0.0125
    
    # Bottom drag
    bottom_drag_coef = 0.5 * args.f0 * 2. / 2600
    
    # Model selection
    model_class = QG if args.model == 'qg' else SW
    model_name = args.model
    
    param = {
        'nx': nx, 'ny': ny, 'nl': 3,
        'dx': dx, 'dy': dy,
        'H': H, 'rho': rho, 'g_prime': g_prime,
        'bottom_drag_coef': bottom_drag_coef,
        'device': device, 'dtype': dtype,
        'slip_coef': args.slip_coef,
        'interp_fd': False,
        'dt': args.dt,
        'compile': args.compile,
        'barotropic_filter': False,
        'barotropic_filter_spectral': False,
        'mask': mask
    }
    
    # Grid for Coriolis and wind
    x, y = torch.meshgrid(
        torch.linspace(0, Lx, nx+1, dtype=dtype, device=device),
        torch.linspace(0, Ly, ny+1, dtype=dtype, device=device),
        indexing='ij')
    
    # Adjust time step for SW
    if model_class == SW:
        c = torch.sqrt(H.sum() * g_prime[0,0,0]).cpu().item()
        cfl = 20 if param['barotropic_filter'] else 0.5
        dt = float(int(cfl * min(dx, dy) / c))
        print(f'SW model: adjusted dt = {dt:.1f} s.')
        param['dt'] = dt
    else:
        dt = args.dt
    
    print(f'Double gyre config, {model_name} model, {nx}x{ny} grid, dt {dt:.1f}s.')
    
    # Coriolis
    f = args.f0 + args.beta * (y - Ly/2)
    param['f'] = f
    
    # Wind forcing
    tau0 = args.wind_mag / rho
    y_ugrid = 0.5*(y[:,1:] + y[:,:-1])
    taux = tau0 * torch.cos(2*torch.pi*(y_ugrid - Ly/2)/Ly)[1:-1,:]
    tauy = 0.
    param['taux'] = taux
    param['tauy'] = tauy
    
    # Initialize model
    qgsw_multilayer = model_class(param)
    
    # Load restart file if provided
    if args.start_file:
        print(f'Starting from file {args.start_file}...')
        zipf = np.load(args.start_file)
        qgsw_multilayer.set_physical_uvh(zipf['u'], zipf['v'], zipf['h'])
    
    t = 0
    freq_checknan = 10
    freq_log = 100
    
    # Time parameters
    n_steps = int(args.n_years * 365 * 24 * 3600 / dt) + 1
    n_steps_save = int(args.save_after_years * 365 * 24 * 3600 / dt)
    freq_save = int(args.freq_save * 24 * 3600 / dt) if args.freq_save > 0 else 0
    freq_plot = int(args.freq_plot * 24 * 3600 / dt) if args.freq_plot > 0 else 0
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f'run_outputs/{model_name}_{nx}x{ny}_dt{int(dt)}_slip{args.slip_coef}'
    
    if freq_save > 0 or args.save_images or args.save_video:
        os.makedirs(output_dir, exist_ok=True)
        print(f'Outputs will be saved to {output_dir}')
        
        if args.save_images:
            image_dir = os.path.join(output_dir, 'images')
            os.makedirs(image_dir, exist_ok=True)
            print(f'Images will be saved to {image_dir}')
    
    # Setup plotting
    plot_enabled = (not args.no_plot) or args.save_images or args.save_video
    if plot_enabled and freq_plot > 0:
        import matplotlib
        import matplotlib.pyplot as plt
        
        if args.no_plot:
            matplotlib.use('Agg')
        else:
            plt.ion()
        
        nl_plot = args.layer
        if model_class == QG:
            npx, npy = 2, 1
            fig, axes = plt.subplots(npy, npx, figsize=(12, 12))
            axes[0].set_title(r'$\omega_g$')
            axes[1].set_title(r'$\omega_a$')
            [(axes[i].set_xticks([]), axes[i].set_yticks([])) for i in range(npx)]
        else:
            npx, npy = 3, 1
            fig, axes = plt.subplots(npy, npx, figsize=(16, 6))
            axes[0].set_title(r'$u$')
            axes[1].set_title(r'$v$')
            axes[2].set_title(r'$h$')
            [(axes[i].set_xticks([]), axes[i].set_yticks([])) for i in range(npx)]
        
        plt.tight_layout()
        if not args.no_plot:
            plt.pause(0.1)
        
        plot_kwargs = {'cmap': 'bwr', 'origin': 'lower'}
        uM, vM, hM = 0, 0, 0
        
        # Video writer setup
        if args.save_video:
            from matplotlib.animation import FFMpegWriter
            video_path = os.path.join(output_dir, f'{model_name}_doublegyre.mp4')
            writer = FFMpegWriter(fps=args.fps, metadata={'artist': 'pyQGSW'})
            writer.setup(fig, video_path, dpi=100)
            print(f'Video will be saved to {video_path}')
    
    # Main loop
    print(f'Starting simulation for {n_steps} steps ({args.n_years} years)...')
    for n in range(1, n_steps+1):
        qgsw_multilayer.step()
        t += dt
        
        n_years = int(t // (365*24*3600))
        n_days = int(t % (365*24*3600) // (24*3600))
        
        if n % freq_checknan == 0 and torch.isnan(qgsw_multilayer.p).any():
            raise ValueError(f'Stopping, NAN number in p at iteration {n}.')
        
        if freq_log > 0 and n % freq_log == 0:
            print(f'n={n:05d}/{n_steps}, t={n_years:02d}y{n_days:03d}d, {qgsw_multilayer.get_print_info()}')
        
        if plot_enabled and freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
            u, v, h = qgsw_multilayer.get_physical_uvh(numpy=True)
            uM, vM = max(uM, 0.8*np.abs(u).max()), max(vM, 0.8*np.abs(v).max())
            hM = max(hM, 0.8*np.abs(h).max())
            
            if model_class == QG:
                wM = 0.2
                w = (qgsw_multilayer.omega / qgsw_multilayer.area / qgsw_multilayer.f0).cpu().numpy()
                w_a = (qgsw_multilayer.omega_a / qgsw_multilayer.f0).cpu().numpy()
                
                axes[0].clear()
                axes[1].clear()
                axes[0].imshow(w[0, nl_plot].T, vmin=-wM, vmax=wM, **plot_kwargs)
                axes[1].imshow(w_a[0, nl_plot].T, vmin=-0.2*wM, vmax=0.2*wM, **plot_kwargs)
                axes[0].set_title(r'$\omega_g$')
                axes[1].set_title(r'$\omega_a$')
            else:
                axes[0].clear()
                axes[1].clear()
                axes[2].clear()
                axes[0].imshow(u[0, nl_plot].T, vmin=-uM, vmax=uM, **plot_kwargs)
                axes[1].imshow(v[0, nl_plot].T, vmin=-vM, vmax=vM, **plot_kwargs)
                axes[2].imshow(h[0, nl_plot].T, vmin=-hM, vmax=hM, **plot_kwargs)
                axes[0].set_title(r'$u$')
                axes[1].set_title(r'$v$')
                axes[2].set_title(r'$h$')
            
            [(axes[i].set_xticks([]), axes[i].set_yticks([])) for i in range(npx)]
            fig.suptitle(f'{n_years} yrs, {n_days:03d} days')
            
            if args.save_images:
                img_path = os.path.join(image_dir, f'frame_{n:08d}.png')
                fig.savefig(img_path, dpi=100, bbox_inches='tight')
            
            if args.save_video:
                writer.grab_frame()
            
            if not args.no_plot:
                plt.pause(0.05)
        
        if freq_save > 0 and n > n_steps_save and n % freq_save == 0:
            filename = os.path.join(output_dir, f'uvh_{n_years:03d}y_{n_days:03d}d.npz')
            u, v, h = qgsw_multilayer.get_physical_uvh(numpy=True)
            if model_class == QG:
                u_a = qgsw_multilayer.u_a.cpu().numpy()
                v_a = qgsw_multilayer.v_a.cpu().numpy()
                np.savez(filename,
                        u=u.astype('float32'), v=v.astype('float32'),
                        u_a=u_a.astype('float32'), v_a=v_a.astype('float32'),
                        h=h.astype('float32'))
                print(f'Saved u,v,h,u_a,v_a to {filename}')
            else:
                np.savez(filename,
                        u=u.astype('float32'), v=v.astype('float32'),
                        h=h.astype('float32'))
                print(f'Saved u,v,h to {filename}')
    
    # Cleanup
    if args.save_video and plot_enabled:
        writer.finish()
        print(f'Video saved to {video_path}')
    
    if not args.no_plot and plot_enabled:
        print('Simulation complete. Close the plot window to exit.')
        plt.ioff()
        plt.show()
    
    print('Done!')


if __name__ == '__main__':
    main()
