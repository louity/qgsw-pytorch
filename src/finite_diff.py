"""Finite difference operators in pytorch,
Louis Thiry, 6 march 2023."""

import torch
import torch.nn.functional as F


def interp_TP(f):
    return 0.25 *(f[...,1:,1:] + f[...,1:,:-1] + f[...,:-1,1:] + f[...,:-1,:-1])


def comp_ke(u, U, v, V):
    u_sq = u * U
    v_sq = v * V
    return 0.25*(u_sq[...,1:,:] + u_sq[...,:-1,:] + v_sq[...,1:] + v_sq[...,:-1])

def laplacian(f, dx, dy):
    return (f[...,2:,1:-1] - 2*f[...,1:-1,1:-1] + f[...,:-2,1:-1]) / dx**2 \
         + (f[...,1:-1,2:] - 2*f[...,1:-1,1:-1] + f[...,1:-1,:-2]) / dy**2


def grad_perp(f):
    """Orthogonal gradient"""
    return f[...,:-1] - f[...,1:], f[...,1:,:] - f[...,:-1,:]

def div_nofluxbc(flux_x, flux_y):
        return torch.diff(F.pad(flux_y, (1,1)), dim=-1) \
             + torch.diff(F.pad(flux_x, (0,0,1,1)), dim=-2)
