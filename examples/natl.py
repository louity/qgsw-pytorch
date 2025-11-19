"""
Simplified north atlantic configuration with realistic coastal geometries.
"""
import argparse
import os
import numpy as np
import skimage.morphology
import torch
import torch.nn.functional as F
import urllib.request
import netCDF4
import scipy.interpolate
import scipy.io

from sw import SW
from qg import QG


def parse_args():
    parser = argparse.ArgumentParser(
        description='North Atlantic configuration with realistic coastal geometries'
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='qg', choices=['qg', 'sw'],
                        help='Model to use: QG or SW')
    
    # Grid parameters
    parser.add_argument('--nx', type=int, default=1024, help='Number of grid points in x')
    parser.add_argument('--ny', type=int, default=512, help='Number of grid points in y')
    parser.add_argument('--lat-min', type=float, default=9.0, help='Minimum latitude')
    parser.add_argument('--lat-max', type=float, default=48.0, help='Maximum latitude')
    
    # Physical parameters
    parser.add_argument('--f0', type=float, default=9.375e-5, help='Mean Coriolis parameter (s^-1)')
    parser.add_argument('--beta', type=float, default=1.754e-11,
                        help='Coriolis gradient (m^-1 s^-1)')
    parser.add_argument('--slip-coef', type=float, default=0.6,
                        help='Slip coefficient (0=no-slip, 1=free-slip)')
    
    # Simulation parameters
    parser.add_argument('--dt', type=float, default=2000, help='Time step (s)')
    parser.add_argument('--n-years', type=float, default=10.0, help='Number of years to simulate')
    parser.add_argument('--compile', action='store_true', default=False, help='Enable torch.compile')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float64'], help='Data type for tensors')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for data files')
    parser.add_argument('--topo-url', type=str,
                        default='https://www.di.ens.fr/louis.thiry/AN_etopo1.mat',
                        help='URL for topography data')
    parser.add_argument('--wind-url', type=str,
                        default='http://iridl.ldeo.columbia.edu/SOURCES/.HELLERMAN/data.nc',
                        help='URL for wind stress data')
    
    # I/O parameters
    parser.add_argument('--output-dir', type=str, default='',
                        help='Output directory (default: run_outputs/qgna_{nx}x{ny}_...)')
    parser.add_argument('--start-file', type=str, default='',
                        help='Restart file (npz format)')
    parser.add_argument('--freq-save', type=float, default=5.0,
                        help='Save frequency in days (0 to disable)')
    parser.add_argument('--save-after-years', type=float, default=0.0,
                        help='Start saving after this many years')
    
    # Plotting parameters
    parser.add_argument('--no-plot', action='store_true', help='Disable interactive plotting')
    parser.add_argument('--save-images', action='store_true',
                        help='Save plots as images to output_dir/images')
    parser.add_argument('--save-video', action='store_true',
                        help='Save plots as MP4 video to output_dir')
    parser.add_argument('--fps', type=int, default=10, help='FPS for video output')
    parser.add_argument('--freq-plot', type=float, default=5.0,
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
    
    # Grids
    deg_to_km = 111e3
    nx, ny = args.nx, args.ny
    lat_min, lat_max = args.lat_min, args.lat_max
    Ly = (lat_max - lat_min) * deg_to_km
    Lx = Ly * nx / ny
    lon_min = -98
    lon_max = lon_min + Lx / (0.5 * (np.cos(lat_min/180*np.pi) + np.cos(lat_max/180*np.pi)) * deg_to_km)
    htop_ocean = -250
    dx = Lx / nx
    dy = Ly / ny
    print(f'Grid lat: {lat_min:.1f}, {lat_max:.1f}, '
          f'lon: {lon_min:.1f}, {lon_max:.1f}, '
          f'dx={dx/1e3:.1f}km, dy={dy/1e3:.1f}km.')
    
    x_cor = np.linspace(lon_min, lon_max, nx+1)
    y_cor = np.linspace(lat_min, lat_max, ny+1)
    lon_w, lat_w = np.meshgrid(x_cor, y_cor, indexing='ij')
    x_cen = 0.5 * (x_cor[1:] + x_cor[:-1])
    y_cen = 0.5 * (y_cor[1:] + y_cor[:-1])
    lon_h, lat_h = np.meshgrid(x_cen, y_cen, indexing='ij')
    lon_u, lat_u = np.meshgrid(x_cor, y_cen, indexing='ij')
    lon_v, lat_v = np.meshgrid(x_cen, y_cor, indexing='ij')
    
    # Data and output directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs('./run_outputs', exist_ok=True)
    
    # Topography
    topo_file = os.path.join(args.data_dir, 'northatlantic_topo.mat')
    if not os.path.isfile(topo_file):
        print(f'Downloading topo file from {args.topo_url}...')
        urllib.request.urlretrieve(args.topo_url, topo_file)
        print('Done.')
    
    data = scipy.io.loadmat(topo_file)
    lon_bath = data['lon_bathy'][:, 0]
    lat_bath = data['lat_bathy'][:, 0]
    bathy = data['bathy'].T
    island_min_area = int(4000 * nx * ny / 1024 / 512)
    lake_min_area = int(40000 * nx * ny / 1024 / 512)
    
    method = 'linear'
    print(f'Interpolating bathymetry on grid with {method} interpolation...')
    bathmetry = scipy.interpolate.RegularGridInterpolator((lon_bath, lat_bath), bathy, method=method)
    
    mask_land = bathmetry((lon_h, lat_h)) > 0
    mask_land = skimage.morphology.area_closing(mask_land, area_threshold=lake_min_area)
    mask_land = np.logical_not(skimage.morphology.area_closing(
        np.logical_not(mask_land), area_threshold=island_min_area))
    mask_land_w = (F.avg_pool2d(
        F.pad(torch.from_numpy(mask_land).type(torch.float64).unsqueeze(0).unsqueeze(0),
              (1,1,1,1), value=1.), (2,2), stride=(1,1))[0,0] > 0.5).numpy()
    
    mask_ocean = bathmetry((lon_h, lat_h)) < htop_ocean
    mask_ocean = skimage.morphology.area_closing(mask_ocean, area_threshold=island_min_area)
    mask_ocean = np.logical_not(skimage.morphology.area_closing(
        np.logical_not(mask_ocean), area_threshold=lake_min_area)).astype('float64')
    
    # Remove ocean cells surrounded by 3 non-ocean cells
    for _ in range(100):
        mask_ocean[1:-1,1:-1] += (1 - mask_ocean[1:-1,1:-1]) * (
            (mask_ocean[:-2,1:-1] + mask_ocean[2:,1:-1] +
             mask_ocean[1:-1,2:] + mask_ocean[1:-1,:-2]) > 2.5)
    mask_ocean = (mask_ocean > 0.5).astype('float64')
    mask = mask_ocean
    
    # Read wind stress climatology
    wind_file = os.path.join(args.data_dir, 'windstress_HellermanRosenstein83.nc')
    if not os.path.isfile(wind_file):
        print(f'Downloading wind file from {args.wind_url}...')
        urllib.request.urlretrieve(args.wind_url, wind_file)
        print('Done.')
    
    ds = netCDF4.Dataset(wind_file, 'r')
    X = ds.variables['X'][:].data
    Y = ds.variables['Y'][:].data
    T = ds.variables['T'][:].data
    
    taux = np.zeros((T.shape[0]+1, nx+1, ny))
    tauy = np.zeros((T.shape[0]+1, nx, ny+1))
    
    print(f'Interpolating wind forcing on grid with {method} interpolation...')
    for t in range(T.shape[0]):
        taux_ref = ds.variables['taux'][:].data[t].T
        tauy_ref = ds.variables['tauy'][:].data[t].T
        taux_interpolator = scipy.interpolate.RegularGridInterpolator((X, Y), taux_ref, method=method)
        tauy_interpolator = scipy.interpolate.RegularGridInterpolator((X, Y), tauy_ref, method=method)
        taux_i = taux_interpolator((lon_u+360, lat_u))
        tauy_i = tauy_interpolator((lon_v+360, lat_v))
        taux[t,:,:] = taux_i
        tauy[t,:,:] = tauy_i
    
    taux *= 1e-4
    tauy *= 1e-4
    taux[-1][:] = taux[0][:]
    tauy[-1][:] = tauy[0][:]
    
    taux = torch.from_numpy(taux).type(dtype).to(device)
    tauy = torch.from_numpy(tauy).type(dtype).to(device)
    
    # Reference layer thickness
    H = torch.zeros(3,1,1, dtype=dtype, device=device)
    H[0,0,0] = 400.
    H[1,0,0] = 1100.
    H[2,0,0] = 2600.
    
    rho = 1000
    g_prime = torch.zeros(3,1,1, dtype=dtype, device=device)
    g_prime[0,0,0] = 9.81
    g_prime[1,0,0] = 0.025
    g_prime[2,0,0] = 0.0125
    
    # Coriolis beta plane
    y = torch.from_numpy(lat_w).type(dtype).to(device) * deg_to_km
    f = args.f0 + args.beta * (y - y.mean())
    print(f'Coriolis param min {f.min().cpu().item():.2e}, max {f.max().cpu().item():.2e}')
    
    # Bottom drag
    bottom_drag_coef = 0.5 * args.f0 * 2. / 2600
    
    # Set model
    model_class = QG if args.model == 'qg' else SW
    model_name = args.model
    
    param = {
        'nx': nx, 'ny': ny, 'nl': 3,
        'H': H, 'dx': dx, 'dy': dy,
        'rho': rho, 'g_prime': g_prime,
        'bottom_drag_coef': bottom_drag_coef,
        'f': f,
        'device': device, 'dtype': dtype,
        'slip_coef': args.slip_coef,
        'dt': args.dt,
        'compile': args.compile,
        'mask': torch.from_numpy(mask).type(dtype).to(device),
        'taux': taux[0,1:-1,:],
        'tauy': tauy[0,:,1:-1],
    }
    
    model_instance = model_class(param)
    
    # Initial condition from rest or restart file
    if args.start_file:
        print(f'Starting from file {args.start_file}...')
        zipf = np.load(args.start_file)
        model_instance.u = torch.from_numpy(zipf['u']).type(dtype).to(device) * model_instance.masks.u * model_instance.dx
        model_instance.v = torch.from_numpy(zipf['v']).type(dtype).to(device) * model_instance.masks.v * model_instance.dy
        model_instance.h = torch.from_numpy(zipf['h']).type(dtype).to(device) * model_instance.masks.h * model_instance.area
        model_instance.compute_diagnostic_variables()
        zipf.close()
    
    # Time and control params
    t = 0
    freq_checknan = 1000
    freq_log = int(5*24*3600 / args.dt)  # each 5 days
    n_steps = int(args.n_years * 365 * 24 * 3600 / args.dt) + 1
    n_steps_save = int(args.save_after_years * 365 * 24 * 3600 / args.dt)
    freq_save = int(args.freq_save * 24 * 3600 / args.dt) if args.freq_save > 0 else 0
    freq_plot = int(args.freq_plot * 24 * 3600 / args.dt) if args.freq_plot > 0 else 0
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f'run_outputs/{model_name}na_{nx}x{ny}_dt{int(args.dt)}_slip{args.slip_coef}'
    
    if freq_save > 0 or args.save_images or args.save_video:
        os.makedirs(output_dir, exist_ok=True)
        print(f'Outputs will be saved to {output_dir}')
        
        if freq_save > 0:
            np.save(os.path.join(output_dir, 'mask_land_h.npy'), mask_land)
            np.save(os.path.join(output_dir, 'mask_land_w.npy'), mask_land_w)
            torch.save(param, os.path.join(output_dir, 'param.pth'))
        
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
        
        palette = plt.cm.bwr
        palette.set_bad(color='grey')
        nl_plot = args.layer
        
        if model_class == QG:
            npx, npy = 1, 1
            fig, ax = plt.subplots(npy, npx, figsize=(12, 12))
            ax.set_title(r'$\omega_g$')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            npx, npy = 3, 1
            fig, axes = plt.subplots(npy, npx, figsize=(16, 6))
            axes[0].set_title(r'$u$')
            axes[1].set_title(r'$v$')
            axes[2].set_title(r'$h$')
            [(axes[i].set_xticks([]), axes[i].set_yticks([])) for i in range(npx)]
        
        uM, vM, hM = 0, 0, 0
        plt.tight_layout()
        if not args.no_plot:
            plt.pause(0.1)
        
        plot_kwargs = {'cmap': palette, 'origin': 'lower'}
        
        # Video writer setup
        if args.save_video:
            from matplotlib.animation import FFMpegWriter
            video_path = os.path.join(output_dir, f'{model_name}_natl.mp4')
            writer = FFMpegWriter(fps=args.fps, metadata={'artist': 'pyQGSW'})
            writer.setup(fig, video_path, dpi=100)
            print(f'Video will be saved to {video_path}')
    
    # Time stepping
    print(f'Starting simulation for {n_steps} steps ({args.n_years} years)...')
    for n in range(1, n_steps+1):
        # Update wind forcing
        month = 12 * (t % (365*24*3600)) / (365*24*3600)
        m_i, m_r = int(month), month - int(month)
        taux_t = (1 - m_r) * taux[m_i,1:-1,:] + m_r * taux[m_i+1,1:-1,:]
        tauy_t = (1 - m_r) * tauy[m_i,:,1:-1] + m_r * tauy[m_i+1,:,1:-1]
        model_instance.set_wind_forcing(taux_t, tauy_t)
        
        # Model forward
        model_instance.step()
        t += args.dt
        
        n_years = int(t // (365*24*3600))
        n_days = int(t % (365*24*3600) // (24*3600))
        
        if n % freq_checknan == 0 and torch.isnan(model_instance.p).any():
            raise ValueError(f'Stopping, NAN number in p at iteration {n}.')
        
        if freq_log > 0 and n % freq_log == 0:
            print(f'n={n:05d}/{n_steps}, t={n_years:02d}y{n_days:03d}d, {model_instance.get_print_info()}')
        
        if plot_enabled and freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
            u, v, h = model_instance.get_physical_uvh(numpy=True)
            uM, vM = max(uM, 0.8*np.abs(u).max()), max(vM, 0.8*np.abs(v).max())
            hM = max(hM, 0.8*np.abs(h).max())
            
            if model_class == QG:
                wM = 0.1
                w = (model_instance.omega / model_instance.area / model_instance.f0).cpu().numpy()[0, nl_plot]
                w = np.ma.masked_where(mask_land_w, w)
                ax.clear()
                ax.imshow(w.T, vmin=-wM, vmax=wM, **plot_kwargs)
                ax.set_title(r'$\omega_g$')
                ax.set_xticks([])
                ax.set_yticks([])
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
            u, v, h = model_instance.get_physical_uvh(numpy=True)
            if model_class == QG:
                u_a = model_instance.u_a.cpu().numpy()
                v_a = model_instance.v_a.cpu().numpy()
                np.savez(filename,
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
