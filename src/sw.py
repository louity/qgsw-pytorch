"""
Shallow-water implementation.
Louis Thiry, Nov 2023 for IFREMER.
"""
import numpy as np
import torch
import torch.nn.functional as F

from finite_diff import interp_TP, comp_ke, div_nofluxbc
from flux import flux
from helmholtz import HelmholtzNeumannSolver
from helmholtz_multigrid import MG_Helmholtz
from masks import Masks
from reconstruction import linear2_centered, wenoz4_left, wenoz6_left

def replicate_pad(f, mask):
    f_ = F.pad(f, (1,1,1,1))
    mask_ = F.pad(mask, (1,1,1,1))
    mask_sum = F.avg_pool2d(
        F.avg_pool2d(mask_, (3,1), stride=(1,1), padding=(1,0), divisor_override=1),
        (1,3), stride=(1,1), padding=(0,1), divisor_override=1)
    f_sum = F.avg_pool2d(
        F.avg_pool2d(f_, (3,1), stride=(1,1), padding=(1,0), divisor_override=1),
        (1,3), stride=(1,1), padding=(0,1), divisor_override=1)
    f_out = f_sum / torch.maximum(torch.ones_like(mask_sum), mask_sum)
    return mask_ * f_ + (1 - mask_) * f_out


def reverse_cumsum(x, dim):
    """Pytorch cumsum in the reverse order
    Example:
    reverse_cumsum(torch.arange(1,4), dim=-1)
    >>> tensor([6, 5, 3])
    """

    return x + torch.sum(x, dim=dim, keepdim=True) - torch.cumsum(x, dim=dim)


def inv_reverse_cumsum(x, dim):
    """Inverse of reverse cumsum function"""
    return torch.cat([-torch.diff(x, dim=dim), x.narrow(dim, -1, 1)], dim=dim)


class SW:
    """
    # Implementation of multilayer rotating shallow-water model

    Following https://doi.org/10.1029/2021MS002663 .

    ## Main ingredients
        - vector invariant formulation
        - velocity RHS using vortex force upwinding with wenoz-5 reconstruction
        - mass continuity RHS with finite volume using wenoz-5 recontruction

    ## Variables
    Prognostic variables u, v, h differ from physical variables
    u_phys, v_phys (velocity components) and
    h_phys (layer thickness perturbation) as they include
    metric terms dx and dy :
      - u = u_phys x dx
      - v = v_phys x dy
      - h = g_phys x dx x dy

    Diagnostic variables are :
      - U = u_phys / dx
      - V = v_phys / dx
      - omega = omega_phys x dx x dy    (rel. vorticity)
      - eta = eta_phys                  (interface height)
      - p = p_phys                      (hydrostratic pressure)
      - k_energy = k_energy_phys        (kinetic energy)
      - pv = pv_phys                    (potential vorticity)

    ## Time integration
    Explicit time integration with RK3-SSP scheme.

    """

    def __init__(self, param):
        """
        Parameters

        param: python dict. with following keys
            'nx':       int, number of grid points in dimension x
            'ny':       int, number grid points in dimension y
            'nl':       nl, number of stacked layer
            'dx':       float or Tensor (nx, ny), dx metric term
            'dy':       float or Tensor (nx, ny), dy metric term
            'H':        Tensor (nl,) or (nl, nx, ny), unperturbed layer thickness
            'g_prime':  Tensor (nl,), reduced gravities
            'f':        Tensor (nx, ny), Coriolis parameter
            'taux':     float or Tensor (nx-1, ny), top-layer forcing, x component
            'tauy':     float or Tensor (nx, ny-1), top-layer forcing, y component
            'dt':       float > 0., integration time-step
            'n_ens':    int, number of ensemble member
            'device':   'str', torch devicee e.g. 'cpu', 'cuda', 'cuda:0'
            'dtype':    torch.float32 of torch.float64
            'slip_coef':    float, 1 for free slip, 0 for no-slip, inbetween for
                        partial free slip.
            'bottom_drag_coef': float, linear bottom drag coefficient
            'barotropic_filter': boolean, i true applies implicit FS calculation
        """

        print(f'Creating {self.__class__.__name__} model...')
        self.device = param['device']
        self.dtype = param['dtype'] if 'dtype' in param.keys() else torch.float64
        self.arr_kwargs = {
            'dtype': self.dtype,
            'device': self.device
        }
        print(f'  - dtype {self.dtype}, device {self.device}.')
        # verifications
        assert len(param['H'].shape) >= 3, \
            'H must be a nz x ny x nx tensor ' \
            'with nx=1 or ny=1 if H does not vary ' \
            f'in x or y direction, got shape {param["H"].shape}.'

        # grid
        self.nx = param['nx']
        self.ny = param['ny']
        self.nl = param['nl']
        self.dx = param['dx']
        self.dy = param['dy']
        self.H = param['H']
        print(f'  - nx, ny, nl =  {self.nx, self.ny, self.nl}')
        self.area = self.dx*self.dy
        self.slip_coef = param['slip_coef'] if 'slip_coef' in param.keys() else 1.

        # optional mask
        nx, ny = self.nx, self.ny
        if 'mask' in param.keys():
            mask = param['mask']
            shape = mask.shape[0], mask.shape[1]
            assert  shape == (nx, ny), f'Invalid mask shape {shape=}!=({nx},{ny})'
            vals = torch.unique(mask).tolist()
            assert  all([v in [0,1] for v in vals]) and vals != [0], \
                    f'Invalid mask with non-binary values : {vals}'
            print(f'  - {"non-" if len(vals)==2 else ""}trivial mask provided')

        else:
            print('  - no mask provided, domain assumed to be rectangular')
            mask = torch.ones(nx, ny, dtype=self.dtype, device=self.device)
        self.masks = Masks(mask)

        # boundary conditions
        assert self.slip_coef >= 0 and self.slip_coef <= 1, \
               f'slip coefficient must be in [0, 1], got {self.slip_coef}'
        cl_type = "free-" if self.slip_coef == 1 else \
                  ("no-" if self.slip_coef == 0 else "partial free-")
        print(f'  - {cl_type}slip boundary condition')

        # Coriolis parameter
        f = param['f']
        shape = f.shape[0], f.shape[1]
        assert  shape == (nx+1, ny+1), f'Invalid f shape {shape=}!=({nx},{ny})'
        self.f = f.unsqueeze(0)
        self.f0 = self.f.mean()
        self.f_ugrid = 0.5 * (self.f[:,:,1:] + self.f[:,:,:-1])
        self.f_vgrid = 0.5 * (self.f[:,1:,:] + self.f[:,:-1,:])
        self.f_hgrid = interp_TP(self.f)
        self.fstar_ugrid = self.f_ugrid * self.area
        self.fstar_vgrid = self.f_vgrid * self.area
        self.fstar_vgrid = self.f_vgrid * self.area
        self.fstar_hgrid = self.f_hgrid * self.area

        # gravity
        self.g_prime = param['g_prime']
        self.g = self.g_prime[0]

        # external top-layer forcing
        taux, tauy = param['taux'], param['tauy']
        self.set_wind_forcing(taux, tauy)
        self.bottom_drag_coef = param['bottom_drag_coef']

        # time
        self.dt = param['dt']
        print(f'  - integration time step {self.dt:.3e}')

        # ensemble
        self.n_ens = param['n_ens'] if 'n_ens' in param.keys() else 1

        # topography and ref values
        self.h_ref = self.H * self.area
        self.eta_ref = -self.H.sum(dim=-3) + reverse_cumsum(self.H, dim=-3)
        self.p_ref = torch.cumsum(self.g_prime * self.eta_ref, dim=-3)
        if self.h_ref.shape[-2] != 1 and self.h_ref.shape[-1] != 1:
            h_ref_ugrid = F.pad(self.h_ref, (0,0,1,1), mode='replicate')
            self.h_ref_ugrid = 0.5 * (h_ref_ugrid[...,1:,:] + h_ref_ugrid[...,:-1,:])
            h_ref_vgrid = F.pad(self.h_ref, (1,1), mode='replicate')
            self.h_ref_vgrid = 0.5 * (h_ref_vgrid[...,1:] + h_ref_vgrid[...,:-1])
            self.dx_p_ref = torch.diff(self.p_ref, dim=-2)
            self.dy_p_ref = torch.diff(self.p_ref, dim=-1)
        else:
            self.h_ref_ugrid = self.h_ref
            self.h_ref_vgrid = self.h_ref
            self.dx_p_ref = 0.
            self.dy_p_ref = 0.


        # initialize variables
        base_shape = (self.n_ens, self.nl,)
        self.h = torch.zeros(base_shape + (self.nx, self.ny), **self.arr_kwargs)
        self.u = torch.zeros(base_shape + (self.nx+1, self.ny), **self.arr_kwargs)
        self.v = torch.zeros(base_shape + (self.nx, self.ny+1), **self.arr_kwargs)
        self.comp_ke = comp_ke
        self.interp_TP = interp_TP
        self.compute_diagnostic_variables()

        # utils and flux computation functions
        self.comp_ke = comp_ke
        self.interp_TP = interp_TP
        self.h_flux_y = lambda h, v: flux(
                h, v,
                dim=-1,
                n_points=6,
                rec_func_2=linear2_centered,
                rec_func_4=wenoz4_left,
                rec_func_6=wenoz6_left,
                mask_2=self.masks.v_sten_hy_eq2[...,1:-1],
                mask_4=self.masks.v_sten_hy_eq4[...,1:-1],
                mask_6=self.masks.v_sten_hy_gt6[...,1:-1])
        self.h_flux_x = lambda h, u: flux(
                h, u,
                dim=-2,
                n_points=6,
                rec_func_2=linear2_centered,
                rec_func_4=wenoz4_left,
                rec_func_6=wenoz6_left,
                mask_2=self.masks.u_sten_hx_eq2[...,1:-1,:],
                mask_4=self.masks.u_sten_hx_eq4[...,1:-1,:],
                mask_6=self.masks.u_sten_hx_gt6[...,1:-1,:])

        self.w_flux_y = lambda w, v_ugrid: flux(
                w, v_ugrid,
                dim=-1,
                n_points=6,
                rec_func_2=linear2_centered,
                rec_func_4=wenoz4_left,
                rec_func_6=wenoz6_left,
                mask_2=self.masks.u_sten_wy_eq2[...,1:-1,:],
                mask_4=self.masks.u_sten_wy_eq4[...,1:-1,:],
                mask_6=self.masks.u_sten_wy_gt4[...,1:-1,:])
        self.w_flux_x = lambda w, u_vgrid: flux(
                w, u_vgrid,
                dim=-2,
                n_points=6,
                rec_func_2=linear2_centered,
                rec_func_4=wenoz4_left,
                rec_func_6=wenoz6_left,
                mask_2=self.masks.v_sten_wx_eq2[...,1:-1],
                mask_4=self.masks.v_sten_wx_eq4[...,1:-1],
                mask_6=self.masks.v_sten_wx_gt6[...,1:-1])

        # barotropic waves filtering for SW
        self.barotropic_filter = False
        if 'barotropic_filter' in param.keys() and param['barotropic_filter']:
            class_name = self.__class__.__name__
            if  class_name == 'SW':
                print('  - Using barotropic filter ', end="")
                self.barotropic_filter = param['barotropic_filter']
                self.tau = 2*self.dt
                if param['barotropic_filter_spectral']:
                    print('spectral approximation')
                    self.barotropic_filter_spectral = True
                    self.H_tot = self.H.sum(dim=-3, keepdim=True)
                    self.lambd = 1. / (self.g * self.dt * self.tau * self.H_tot)
                    self.helm_solver = HelmholtzNeumannSolver(
                            self.nx, self.ny, self.dx, self.dy, self.lambd,
                            self.dtype, self.device, mask=self.masks.h[0,0])
                else:
                    self.barotropic_filter_spectral = False
                    print('in exact form')
                    coef_ugrid = (self.h_tot_ugrid * self.masks.u)[0,0]
                    coef_vgrid = (self.h_tot_vgrid * self.masks.v)[0,0]
                    lambd = 1. / (self.g * self.dt * self.tau)
                    self.helm_solver = MG_Helmholtz(self.dx, self.dy,
                            self.nx, self.ny, coef_ugrid, coef_vgrid=coef_vgrid,
                            lambd=lambd, device=self.device, dtype=self.dtype,
                            mask=self.masks.h[0,0], niter_bottom=20,
                            use_compilation=False)
            else:
                print(f'  - class {class_name}!=SW, ignoring barotropic filter ')



        # precompile torch functions
        use_compilation =  param['compile'] if 'compile' in param.keys() else True
        if use_compilation:
            print(f'  - torch version {torch.__version__} ', end='')
            if torch.__version__[0] == '2':
                print('>= 2.0, using torch.compile for compilation.')
                self.comp_ke = torch.compile(self.comp_ke)
                self.interp_TP = torch.compile(self.interp_TP)
                self.h_flux_y = torch.compile(self.h_flux_y)
                self.h_flux_x = torch.compile(self.h_flux_x)
                self.w_flux_y = torch.compile(self.w_flux_y)
                self.w_flux_x = torch.compile(self.w_flux_x)
            else:
                print('< 2.0, using torch.jit.trace for compilation.')
                self.comp_ke = torch.jit.trace(
                        self.comp_ke, (self.u, self.U, self.v, self.V))
                self.interp_TP = torch.jit.trace(self.interp_TP, (self.U,))
                self.h_flux_y = torch.jit.trace(self.h_flux_y,
                        (self.h, self.V[...,1:-1]))
                self.h_flux_x = torch.jit.trace(self.h_flux_x,
                        (self.h, self.U[...,1:-1,:]))
                self.w_flux_y = torch.jit.trace(self.w_flux_y,
                        (self.omega[...,1:-1,:], self.V_m))
                self.w_flux_x = torch.jit.trace(self.w_flux_x,
                        (self.omega[...,1:-1], self.U_m))
        else:
            print('  - No compilation')


    def set_wind_forcing(self, taux, tauy):
        nx, ny = self.nx, self.ny
        assert type(taux) == float or taux.shape == (nx-1, ny), \
               f'taux must be a float or a {(nx-1, ny)} Tensor'
        assert type(tauy) == float or tauy.shape == (nx, ny-1), \
               f'taux must be a float or a {(nx-1, ny)} Tensor'
        self.taux = taux
        self.tauy = tauy


    def get_physical_uvh(self, numpy=False):
        """Get physical variables u_phys, v_phys, h_phys from state variables."""
        u_phys = (self.u / self.dx).cpu()
        v_phys = (self.v / self.dy).cpu()
        h_phys = (self.h / self.area).cpu()

        return (u_phys.numpy(), v_phys.numpy(), h_phys.numpy()) if numpy \
               else (u_phys, v_phys, h_phys)


    def set_physical_uvh(self, u_phys, v_phys, h_phys):
        """
        Set state variables with physical variables u_phys, v_phys, h_phys.
        """
        u_ = torch.from_numpy(u_phys) if isinstance(u_phys, np.ndarray) else u_phys
        v_ = torch.from_numpy(v_phys) if isinstance(v_phys, np.ndarray) else v_phys
        h_ = torch.from_numpy(h_phys) if isinstance(h_phys, np.ndarray) else h_phys
        u_  = u_.to(self.device)
        v_  = u_.to(self.device)
        h_  = u_.to(self.device)
        assert u_ * self.masks.u == u_, \
                'Input velocity u incoherent with domain mask, ' \
                'velocity must be zero out of domain.'
        assert v_ * self.masks.v == v_, \
                'Input velocity v incoherent with domain mask, ' \
                'velocity must be zero out of domain.'
        self.u = u_.type(self.dtype) * self.masks.u * self.dx
        self.v = v_.type(self.dtype) * self.masks.v * self.dy
        self.h = h_.type(self.dtype) * self.masks.h * self.area
        self.compute_diagnostic_variables()


    def get_print_info(self):
        """
        Returns a string with summary of current variables.
        """
        hl_mean = (self.h / self.area).mean((-1,-2)).squeeze().cpu().numpy()
        eta = (self.eta)
        u, v, h = self.u / self.dx, self.v / self.dy, self.h / self.area
        with np.printoptions(precision=2):
            return \
                f'u: {u.mean().cpu().item():+.5E}, ' \
                f'{u.abs().max().cpu().item():.5E}, ' \
                f'v: {v.mean().cpu().item():+.5E}, ' \
                f'{v.abs().max().cpu().item():.5E}, ' \
                f'hl_mean: {hl_mean}, ' \
                f'h min: {h.min().cpu().item():.5E}, ' \
                f'max: {h.max().cpu().item():.5E}, ' \
                f'eta_sur min: {eta[:,0].min().cpu().item():+.5f}, ' \
                f'max: {eta[:,0].max().cpu().item():.5f}'


    def advection_h(self):
        """
        Advection RHS for thickness perturbation h
        dt_h = - div(h_tot [u v]),  h_tot = h_ref + h
        """
        h_tot = self.h_ref + self.h
        h_tot_flux_y = self.h_flux_y(h_tot, self.V[...,1:-1])
        h_tot_flux_x = self.h_flux_x(h_tot, self.U[...,1:-1,:])
        return -div_nofluxbc(h_tot_flux_x, h_tot_flux_y) * self.masks.h


    def advection_momentum(self):
        """
        Advection RHS for momentum (u, v)
        """
        # Vortex-force + Coriolis
        omega_Vm = self.w_flux_y(self.omega[...,1:-1,:], self.V_m)
        omega_Um = self.w_flux_x(self.omega[...,1:-1], self.U_m)

        dt_u = omega_Vm + self.fstar_ugrid[...,1:-1,:] * self.V_m
        dt_v = -(omega_Um + self.fstar_vgrid[...,1:-1] * self.U_m)

        # grad pressure + k_energy
        ke_pressure = self.k_energy + self.p
        dt_u -= torch.diff(ke_pressure, dim=-2) + self.dx_p_ref
        dt_v -= torch.diff(ke_pressure, dim=-1) + self.dy_p_ref

        # wind forcing and bottom drag
        dt_u, dt_v = self.add_wind_forcing(dt_u, dt_v)
        dt_u, dt_v = self.add_bottom_drag(dt_u, dt_v)

        return F.pad(dt_u, (0,0,1,1))*self.masks.u, \
               F.pad(dt_v, (1,1,0,0))*self.masks.v


    def add_wind_forcing(self, du, dv):
        """
        Add wind forcing to the derivatives du, dv.
        """
        H_ugrid = (self.h_tot_ugrid) / self.area
        H_vgrid = (self.h_tot_vgrid) / self.area
        du[..., 0,:,:] +=  self.taux / H_ugrid[...,0,1:-1,:] * self.dx
        dv[..., 0,:,:] +=  self.tauy / H_vgrid[...,0,:,1:-1] * self.dy
        return du, dv


    def add_bottom_drag(self, du, dv):
        """
        Add bottom drag to the derivatives du, dv.
        """
        du[...,-1,:,:] += -self.bottom_drag_coef * self.u[...,-1,1:-1,:]
        dv[...,-1,:,:] += -self.bottom_drag_coef * self.v[...,-1,:,1:-1]
        return du, dv


    def compute_omega(self, u, v):
        """
        Pad u and v using boundary conditions (free-slip, partial free-slip,
        no-slip).
        """
        u_ = F.pad(u, (1,1,0,0))
        v_ = F.pad(v, (0,0,1,1))
        dx_v = torch.diff(v_, dim=-2)
        dy_u = torch.diff(u_, dim=-1)
        curl_uv = dx_v - dy_u
        alpha = 2 * (1 - self.slip_coef)
        omega = self.masks.w_valid * curl_uv \
              + self.masks.w_cornerout_bound * (1 - self.slip_coef) * curl_uv \
              + self.masks.w_vertical_bound * alpha * dx_v \
              - self.masks.w_horizontal_bound * alpha * dy_u
        return omega


    def compute_diagnostic_variables(self):
        """
        Compute the model's diagnostic variables given the prognostic
        variables self.u, self.v, self.h .
        """
        self.omega = self.compute_omega(self.u, self.v)
        self.eta = reverse_cumsum(self.h / self.area, dim=-3)
        self.p = torch.cumsum(self.g_prime * self.eta, dim=-3)
        self.U = self.u / self.dx**2
        self.V = self.v / self.dy**2
        self.U_m = self.interp_TP(self.U)
        self.V_m = self.interp_TP(self.V)
        self.k_energy = self.comp_ke(self.u, self.U, self.v, self.V) * self.masks.h
        # self.pv = (self.interp_TP(self.omega) + self.fstar_hgrid) \
                  # / (self.h_ref + self.h)

        h_ = replicate_pad(self.h, self.masks.h)
        self.h_ugrid = 0.5 * (h_[...,1:,1:-1] + h_[...,:-1,1:-1])
        self.h_vgrid = 0.5 * (h_[...,1:-1,1:] + h_[...,1:-1,:-1])
        self.h_tot_ugrid = self.h_ref_ugrid + self.h_ugrid
        self.h_tot_vgrid = self.h_ref_vgrid + self.h_vgrid


    def filter_barotropic_waves(self, dt_u, dt_v, dt_h):
        """
        Inspired from https://doi.org/10.1029/2000JC900089.
        """
        # compute RHS
        u_star = (self.u + self.dt*dt_u) / self.dx
        v_star = (self.v + self.dt*dt_v) / self.dy
        u_bar_star = (u_star * self.h_tot_ugrid).sum(dim=-3, keepdim=True) \
                     / self.h_tot_ugrid.sum(dim=-3, keepdim=True)
        v_bar_star = (v_star * self.h_tot_vgrid).sum(dim=-3, keepdim=True) \
                     / self.h_tot_vgrid.sum(dim=-3, keepdim=True)
        if self.barotropic_filter_spectral:
            rhs = 1. / (self.g * self.dt * self.tau) * (
                    torch.diff(u_bar_star, dim=-2) / self.dx \
                + torch.diff(v_bar_star, dim=-1) / self.dy)
            w_surf_imp = self.helm_solver.solve(rhs)
        else:
            rhs = 1. / (self.g * self.dt * self.tau) * (
                    torch.diff(self.h_tot_ugrid * u_bar_star, dim=-2) / self.dx \
                  + torch.diff(self.h_tot_vgrid * v_bar_star, dim=-1) / self.dy)
            coef_ugrid = (self.h_tot_ugrid * self.masks.u)[0,0]
            coef_vgrid = (self.h_tot_vgrid * self.masks.v)[0,0]
            w_surf_imp = self.helm_solver.solve(rhs, coef_ugrid, coef_vgrid)
            # WIP

        filt_u = F.pad(-self.g * self.tau * torch.diff(w_surf_imp, dim=-2), (0,0,1,1)) * self.masks.u
        filt_v = F.pad(-self.g * self.tau * torch.diff(w_surf_imp, dim=-1), (1,1)) * self.masks.v


        return dt_u + filt_u, \
               dt_v + filt_v, \
               dt_h


    def compute_time_derivatives(self):
        """
        Computes the state variables derivatives dt_u, dt_v, dt_h
        """
        self.compute_diagnostic_variables()
        dt_h = self.advection_h()
        dt_u, dt_v = self.advection_momentum()
        if self.barotropic_filter:
            dt_u, dt_v, dt_h = self.filter_barotropic_waves(dt_u, dt_v, dt_h)

        return dt_u, dt_v, dt_h


    def step(self):
        """
        Performs one step time-integration with RK3-SSP scheme.
        """
        dt0_u, dt0_v, dt0_h = self.compute_time_derivatives()
        self.u += self.dt * dt0_u
        self.v += self.dt * dt0_v
        self.h += self.dt * dt0_h

        dt1_u, dt1_v, dt1_h = self.compute_time_derivatives()
        self.u += (self.dt/4) * (dt1_u - 3*dt0_u)
        self.v += (self.dt/4) * (dt1_v - 3*dt0_v)
        self.h += (self.dt/4) * (dt1_h - 3*dt0_h)

        dt2_u, dt2_v, dt2_h = self.compute_time_derivatives()
        self.u += (self.dt/12) * (8*dt2_u - dt1_u - dt0_u)
        self.v += (self.dt/12) * (8*dt2_v - dt1_v - dt0_v)
        self.h += (self.dt/12) * (8*dt2_h - dt1_h - dt0_h)

