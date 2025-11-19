"""
Spectral 2D Helmholtz equation solver on rectangular and non-rectangular domain.
  - Colocated Dirichlet BC with DST-I  (type-I discrete sine transform)
  - Staggered Neumann   BC with DCT-II (type-II discrete consine transform)
  - Non-rectangular domains emmbedded in rectangular domains with a mask.
  - Capacitance matrix method for non-rectangular domains
Louis Thiry, 2023.
"""
import torch
import torch.nn.functional as F


def compute_laplace_dctII(nx, ny, dx, dy, arr_kwargs):
    """DCT-II of standard 5-points laplacian on uniform grid"""
    x, y = torch.meshgrid(torch.arange(nx, **arr_kwargs),
                          torch.arange(ny, **arr_kwargs),
                          indexing='ij')
    return 2*(torch.cos(torch.pi/nx*x) - 1)/dx**2 \
         + 2*(torch.cos(torch.pi/ny*y) - 1)/dy**2


def dctII(x, exp_vec):
    """
    1D forward type-II discrete cosine transform (DCT-II)
    using fft and precomputed auxillary vector exp_vec.
    """
    v = torch.cat([x[...,::2], torch.flip(x, dims=(-1,))[...,::2]], dim=-1)
    V = torch.fft.fft(v)
    return (V*exp_vec).real


def idctII(x, iexp_vec):
    """
    1D inverse type-II discrete cosine transform (DCT-II)
    using fft and precomputed auxillary vector iexp_vec.
    """
    N = x.shape[-1]
    x_rev = torch.flip(x, dims=(-1,))[...,:-1]
    v = torch.cat([x[...,0:1],
            iexp_vec[...,1:N]*(x[...,1:N]-1j*x_rev)], dim=-1) / 2
    V = torch.fft.ifft(v)
    y = torch.zeros_like(x)
    y[...,::2] = V[...,:N//2].real;
    y[...,1::2] = torch.flip(V, dims=(-1,))[...,:N//2].real
    return y


def dctII2D(x, exp_vec_x, exp_vec_y):
    """2D forward DCT-II."""
    return dctII(
            dctII(x, exp_vec_y).transpose(-1,-2),
            exp_vec_x).transpose(-1,-2)


def idctII2D(x, iexp_vec_x, iexp_vec_y):
    """2D inverse DCT-II."""
    return idctII(
            idctII(x, iexp_vec_y).transpose(-1,-2),
            iexp_vec_x).transpose(-1,-2)


def compute_dctII_exp_vecs(N, dtype, device):
    """Compute auxillary exp_vec and iexp_vec used in
    fast DCT-II computations with FFTs."""
    N_range = torch.arange(N, dtype=dtype, device=device)
    exp_vec = 2 * torch.exp(-1j*torch.pi*N_range/(2*N))
    iexp_vec = torch.exp(1j*torch.pi*N_range/(2*N))
    return exp_vec, iexp_vec


def solve_helmholtz_dctII(rhs, helmholtz_dctII,
        exp_vec_x, exp_vec_y,
        iexp_vec_x, iexp_vec_y):
    """Solves Helmholtz equation with DCT-II fast diagonalisation."""
    rhs_dctII = dctII2D(rhs.type(helmholtz_dctII.dtype), exp_vec_x, exp_vec_y)
    return idctII2D(rhs_dctII / helmholtz_dctII, iexp_vec_x, iexp_vec_y
            ).type(rhs.dtype)


def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform (DST-I), forward and inverse
    since DST-I is auto-inverse."""
    return torch.fft.irfft(
            -1j*F.pad(x, (1,1)), dim=-1, norm=norm)[...,1:x.shape[-1]+1]


def dstI2D(x, norm='ortho'):
    """2D DST-I."""
    return dstI1D(dstI1D(x, norm=norm).transpose(-1,-2),
                  norm=norm).transpose(-1,-2)


def compute_laplace_dstI(nx, ny, dx, dy, arr_kwargs):
    """Type-I discrete sine transform of the usual 5-points
    discrete laplacian operator on uniformly spaced grid."""
    x, y = torch.meshgrid(torch.arange(1,nx, **arr_kwargs),
                          torch.arange(1,ny, **arr_kwargs),
                          indexing='ij')
    return 2*(torch.cos(torch.pi/nx*x) - 1)/dx**2 \
         + 2*(torch.cos(torch.pi/ny*y) - 1)/dy**2


def solve_helmholtz_dstI(rhs, helmholtz_dstI):
    """Solves 2D Helmholtz equation with DST-I fast diagonalization."""
    return F.pad(dstI2D(dstI2D(rhs.type(helmholtz_dstI.dtype))/helmholtz_dstI),
                 (1,1,1,1)
                ).type(rhs.dtype)


def compute_capacitance_matrices(helmholtz_dstI, bound_xids, bound_yids):
    nl  = helmholtz_dstI.shape[-3]
    M = bound_xids.shape[0]

    # compute G matrices
    inv_cap_matrices = torch.zeros((nl, M, M), dtype=torch.float64, device='cpu')
    rhs = torch.zeros(helmholtz_dstI.shape[-3:], dtype=torch.float64,
                      device=helmholtz_dstI.device)
    for m in range(M):
        rhs.fill_(0)
        rhs[..., bound_xids[m], bound_yids[m]] = 1
        sol = dstI2D(dstI2D(rhs) / helmholtz_dstI.type(torch.float64))
        inv_cap_matrices[:,m] = sol[...,bound_xids, bound_yids].cpu()

    # invert G matrices to get capacitance matrices
    cap_matrices = torch.zeros_like(inv_cap_matrices)
    for l in range(nl):
        cap_matrices[l] = torch.linalg.inv(inv_cap_matrices[l])

    return cap_matrices.to(helmholtz_dstI.device)


def solve_helmholtz_dstI_cmm(rhs, helmholtz_dstI,
                            cap_matrices, bound_xids, bound_yids,
                            mask):
    sol_1 = dstI2D(
                dstI2D(rhs.type(helmholtz_dstI.dtype)) / helmholtz_dstI
            ).type(rhs.dtype)
    alphas = torch.einsum(
        '...ij,...j->...i', cap_matrices, -sol_1[..., bound_xids, bound_yids])

    rhs_2 = torch.zeros_like(rhs)
    rhs_2[..., bound_xids, bound_yids] = alphas

    return solve_helmholtz_dstI(rhs + rhs_2, helmholtz_dstI) * mask


class HelmholtzNeumannSolver:
    def __init__(self, nx, ny, dx, dy, lambd, dtype, device, mask=None):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.lambd = lambd
        self.device = device
        self.dtype = dtype

        # helmholtz dct-II
        self.helmholtz_dctII = compute_laplace_dctII(
                nx, ny, dx, dy, {'dtype':dtype, 'device': device}) \
                - lambd

        # auxillary vectors for DCT-II computations
        exp_vec_x, iexp_vec_x = compute_dctII_exp_vecs(nx, dtype, device)
        exp_vec_y, iexp_vec_y = compute_dctII_exp_vecs(ny, dtype, device)
        self.exp_vec_x = exp_vec_x.unsqueeze(0).unsqueeze(0)
        self.iexp_vec_x = iexp_vec_x.unsqueeze(0).unsqueeze(0)
        self.exp_vec_y = exp_vec_y.unsqueeze(0).unsqueeze(0)
        self.iexp_vec_y = iexp_vec_y.unsqueeze(0).unsqueeze(0)

        # mask
        if mask is not None:
            shape = mask.shape[0], mask.shape[1]
            assert shape == (nx, ny), f'Invalid mask {shape=} != nx, ny {nx, ny}'
            self.mask = mask.unsqueeze(0).type(dtype).to(device)
        else:
            self.mask = torch.ones(1, nx, ny, dtype=self.dtype, device=self.device)
        self.not_mask = 1 - self.mask

        # mask on u- and v-grid
        self.mask_u = (F.avg_pool2d(
                self.mask, (2,1), stride=(1,1),
                padding=(1,0), divisor_override=1) > 1.5
                ).type(self.dtype)
        self.mask_v = (F.avg_pool2d(
                self.mask, (1,2), stride=(1,1),
                padding=(0,1), divisor_override=1) > 1.5
                ).type(self.dtype)

        # irregular boundary indices
        mask_neighbor_x = self.mask * F.pad(
                F.avg_pool2d(
                    self.mask, (3,1), stride=(1,1),
                    divisor_override=1) < 2.5
                , (0,0,1,1))
        mask_neighbor_y = self.mask * F.pad(
                F.avg_pool2d(
                    self.mask, (1,3), stride=(1,1),
                    divisor_override=1) < 2.5
                , (1,1,0,0))
        self.mask_irrbound = torch.logical_or(
                mask_neighbor_x, mask_neighbor_y)
        self.irrbound_xids, self.irrbound_yids = torch.where(self.mask_irrbound[0])

        # compute capacitance matrix
        self.compute_capacitance_matrix()


    def helmholtz_reg_domain(self, f):
        f_ = F.pad(f, (1,1,1,1), mode='replicate')
        dxx_f = (f_[...,2:,1:-1] + f_[...,:-2,1:-1] - 2*f_[...,1:-1,1:-1]) \
                / self.dx**2
        dyy_f = (f_[...,1:-1,2:] + f_[...,1:-1,:-2] - 2*f_[...,1:-1,1:-1]) \
                / self.dy**2
        return dxx_f + dyy_f - self.lambd * f


    def helmholtz(self, f):
        if len(self.irrbound_xids) == 0:
            return self.helmholtz_reg_domain(f)

        f_ = F.pad(f, (1,1,1,1), mode='replicate')
        dx_f = torch.diff(f_[...,1:-1], dim=-2) / self.dx
        dy_f = torch.diff(f_[...,1:-1,:], dim=-1) / self.dy
        dxx_f = torch.diff(dx_f*self.mask_u, dim=-2) / self.dx
        dyy_f = torch.diff(dy_f*self.mask_v, dim=-1) / self.dy

        return (dxx_f + dyy_f - self.lambd * f) * self.mask


    def compute_capacitance_matrix(self):
        M = len(self.irrbound_xids)
        if M == 0:
            self.cap_matrix = None
            return

        # compute inverse capacitance matrice
        inv_cap_matrix = torch.zeros(
                (M, M), dtype=torch.float64, device=self.device)
        for m in range(M):
            v = torch.zeros(M, device=self.device, dtype=torch.float64)
            v[m] = 1
            inv_cap_matrix[:,m] = \
                (v - self.V_T(self.G(self.U(v))))

        # invert on cpu
        cap_matrix = torch.linalg.inv(inv_cap_matrix.cpu())

        # convert to dtype and transfer to device
        self.cap_matrix = cap_matrix.type(self.dtype).to(self.device)


    def U(self, v):
        Uv = torch.zeros_like(self.mask)
        Uv[...,self.irrbound_xids, self.irrbound_yids] = v
        return Uv


    def V_T(self, field):
        return (
            self.helmholtz_reg_domain(field)
          - self.helmholtz(field)
          )[..., self.irrbound_xids, self.irrbound_yids]


    def G(self, field):
        return solve_helmholtz_dctII(field, self.helmholtz_dctII,
                self.exp_vec_x, self.exp_vec_y, self.iexp_vec_x,
                self.iexp_vec_y)


    def solve(self, rhs):
        GF = self.G(rhs)
        if len(self.irrbound_xids) == 0:
            return GF
        V_TGF = self.V_T(GF)
        rho = torch.einsum(
            'ij,...j->...i', self.cap_matrix, V_TGF)
        return GF + self.G(self.U(rho))



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.rcParams.update({'font.size': 18})
    plt.ion()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    # grid
    N = 4
    nx, ny = 2*2**(N-1), 2*2**(N-1)
    shape = (nx, ny)
    L = 2000e3
    xc = torch.linspace(-L, L, nx+1, dtype=dtype, device=device)
    yc = torch.linspace(-L, L, ny+1, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(xc, yc, indexing='ij')
    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]

    # Helmholtz eq.
    lambd = 1e-2 * torch.ones(1,1,1).type(dtype).to(device)
    helmholtz_dirichletbc = lambda f, dx, dy, lambd: \
          (f[...,2:,1:-1] + f[...,:-2,1:-1] - 2*f[...,1:-1,1:-1])/dx**2 \
        + (f[...,1:-1,2:] + f[...,1:-1,:-2] - 2*f[...,1:-1,1:-1])/dy**2 \
        - lambd * f[...,1:-1,1:-1]
    helmholtz_dstI = compute_laplace_dstI(
            nx, ny, dx, dy, {'dtype':dtype, 'device': device}) \
            - lambd


    # Rectangular domain
    frect = torch.zeros(1, nx+1, ny+1, dtype=dtype, device=device)
    frect[...,1:-1,1:-1].normal_()
    Hfrect = helmholtz_dirichletbc(frect, dx, dy, lambd)
    frect_r = solve_helmholtz_dstI(Hfrect, helmholtz_dstI)
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].set_title('f')
    fig.colorbar(ax[0].imshow(frect[0].cpu().T, origin='lower'), ax=ax[0])
    ax[1].set_title('|f - f_r|')
    fig.colorbar(ax[1].imshow(torch.abs(frect - frect_r)[0].cpu().T, origin='lower'), ax=ax[1])
    fig.suptitle('Solving Dirichlet-BC Helmholtz equation on square domain with DSTI')
    fig.tight_layout()


    # Circular domain
    mask = (1 > ((xx/L)**2 + (yy/L)**2)).type(dtype)
    mask[[0,-1],:] = 0
    mask[[0,-1]] = 0
    domain_neighbor = \
        F.avg_pool2d(mask.reshape((1,1)+mask.shape), kernel_size=3, stride=1, padding=0)[0,0] > 0
    irrbound_xids, irrbound_yids = torch.where(
            torch.logical_and(mask[1:-1,1:-1] < 0.5, domain_neighbor))
    cap_matrices = compute_capacitance_matrices(
            helmholtz_dstI, irrbound_xids,
            irrbound_yids)

    fcirc = mask * torch.zeros_like(mask).normal_().unsqueeze(0)
    Hfcirc = helmholtz_dirichletbc(fcirc, dx, dy, lambd) * mask[1:-1,1:-1]
    fcirc_r = solve_helmholtz_dstI_cmm(Hfcirc, helmholtz_dstI,
                            cap_matrices, irrbound_xids,
                            irrbound_yids, mask)

    palette = plt.cm.bwr.with_extremes(bad='grey')
    fig, ax = plt.subplots(1,2, figsize=(18,9))
    ax[0].set_title('$f$')
    vM = fcirc[0].abs().max().cpu().item()
    # fcirc_ma = np.ma.masked_where((1-mask).cpu().numpy(), fcirc[0].cpu().numpy())
    fcirc_ma = fcirc[0].cpu().numpy()
    fig.colorbar(ax[0].imshow(fcirc_ma.T, vmin=-vM, vmax=vM, origin='lower', cmap=palette), ax=ax[0])
    ax[1].set_title('$f - f_{\\rm inv}$')
    diff =( fcirc - fcirc_r)[0].cpu().numpy()
    vM = np.abs(diff).max()
    # diff_ma = np.ma.masked_where((1-mask).cpu().numpy(), diff)
    diff_ma =diff
    fig.colorbar(ax[1].imshow(diff_ma.T, vmin=-vM, vmax=vM, origin='lower', cmap=palette), ax=ax[1])
    fig.suptitle('Solving Dirichlet-BC Helmholtz eq. $\\Delta f - f = r$ on circular domain with CMM and DST-I')
    ax[0].set_xticks([]), ax[1].set_xticks([]), ax[0].set_yticks([]), ax[1].set_yticks([])
    fig.tight_layout()


    ## Neumann BC
    dx, dy = 20000, 20000
    lambd = torch.ones(1,1,1).type(dtype).to(device) / dx * dy

    N = 32
    nx, ny = N//2, N
    f1 = torch.zeros(1, nx, ny, dtype=dtype, device=device).normal_()
    mask1 = torch.ones(nx, ny, dtype=dtype, device=device)

    solver1 = HelmholtzNeumannSolver(
            nx, ny, dx, dy, lambd, dtype, device, mask=mask1)

    H_f1 = solver1.helmholtz(f1)
    f1_r = solver1.solve(H_f1)

    vM = max(f1.abs().cpu().max().item(), f1_r.cpu().max().item())
    diff = (f1 - f1_r) * mask1
    vM2 = diff.abs().cpu().max().item()

    fig, ax = plt.subplots(1,3, figsize=(16,6))
    ax[0].set_title('f')
    fig.colorbar(ax[0].imshow(f1[0].cpu().T, vmin=-vM, vmax=vM, cmap='bwr', origin='lower'), ax=ax[0])
    ax[1].set_title('f_r')
    fig.colorbar(ax[1].imshow(f1_r[0].cpu().T, vmin=-vM, vmax=vM, cmap='bwr', origin='lower'), ax=ax[1])
    ax[2].set_title('|f - f_r|')
    fig.colorbar(ax[2].imshow(diff[0].cpu().T, vmin=-vM2, vmax=vM2, cmap='bwr', origin='lower'), ax=ax[2])
    fig.suptitle('Neumann-BC, solving Helmholtz eq. with DCT-II')
    ax[0].set_xticks([]), ax[1].set_xticks([]), ax[2].set_xticks([])
    ax[0].set_yticks([]), ax[1].set_yticks([]), ax[2].set_yticks([])
    fig.tight_layout()

    #
    nx, ny = N, N
    f2 = torch.zeros(1, nx, ny, dtype=dtype, device=device)
    f2[:,N//2:,:] = f1
    mask2 = torch.ones(nx, ny, dtype=dtype, device=device)
    mask2[:N//2,:] = 0

    solver2 = HelmholtzNeumannSolver(
            nx, ny, dx, dy, lambd, dtype, device, mask=mask2)

    H_f2 = solver2.helmholtz(f2)
    f2_r = solver2.solve(H_f2)

    vM = max(f2.abs().cpu().max().item(), f2_r.cpu().max().item())
    diff = (f2 - f2_r) * mask2
    vM2 = diff.abs().cpu().max().item()

    fig, ax = plt.subplots(1,3, figsize=(16,6))
    ax[0].set_title('f')
    fig.colorbar(ax[0].imshow(f2[0].cpu().T, vmin=-vM, vmax=vM, cmap='bwr', origin='lower'), ax=ax[0])
    ax[1].set_title('f_r')
    fig.colorbar(ax[1].imshow(f2_r[0].cpu().T, vmin=-vM, vmax=vM, cmap='bwr', origin='lower'), ax=ax[1])
    ax[2].set_title('|f - f_r|')
    fig.colorbar(ax[2].imshow(diff[0].cpu().T, vmin=-vM2, vmax=vM2, cmap='bwr', origin='lower'), ax=ax[2])
    fig.suptitle('Neumann-BC, solving Helmholtz eq. with DCT-II')
    ax[0].set_xticks([]), ax[1].set_xticks([]), ax[2].set_xticks([])
    ax[0].set_yticks([]), ax[1].set_yticks([]), ax[2].set_yticks([])
    fig.tight_layout()

