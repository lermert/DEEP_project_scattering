# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# pyforge, developed by The Swiss Seismological Service and Mondaic
# Copyright (C) 2022, The Swiss Seismological Service
# All rights reserved
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Implementation of a von karman random filter."""

import typing
from dataclasses import dataclass

import numpy as np

try:
    from scipy.fft import fftn, ifftn, fftshift, ifftshift
except AttributeError:
    from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
import xarray as xr
from numba import njit, objmode
from salvus.utils.type_validators import RequiredPrecondition
from scipy.special import gamma
from typing_extensions import Annotated

ValidVonKarmanCoord = RequiredPrecondition[
    typing.Union[typing.Tuple[float, float], np.ndarray]
](
    lambda x: len(x) == 2
    if isinstance(x, tuple)
    else isinstance(x, np.ndarray)
    and (np.diff(x) > 0).all()
    and np.isclose(np.diff(x), np.diff(x)[0]),
    "Please ensure to pass either a tuple of (kc_max, c_max) for the "
    "coordinate c or a numpy array of coordinate values that is "
    "uniformly increasing.",
)
"""Ensures that inputs passed as parameters to the Von Karman filters are valid."""


@dataclass(frozen=True)
class VonKarmanParams2D:
    """Parameters for a 2-D Von Karman filter."""

    x: Annotated[
        typing.Union[typing.Tuple[float, float], np.ndarray], ValidVonKarmanCoord
    ]
    y: Annotated[
        typing.Union[typing.Tuple[float, float], np.ndarray], ValidVonKarmanCoord
    ]

    ax: float = 200.0
    ay: float = 200.0
    nu: float = 0.3
    sigma: float = 0.05


@dataclass(frozen=True)
class VonKarmanParams3D:
    """Parameters for a 3-D Von Karman filter."""

    x: Annotated[
        typing.Union[typing.Tuple[float, float], np.ndarray], ValidVonKarmanCoord
    ]
    y: Annotated[
        typing.Union[typing.Tuple[float, float], np.ndarray], ValidVonKarmanCoord
    ]
    z: Annotated[
        typing.Union[typing.Tuple[float, float], np.ndarray], ValidVonKarmanCoord
    ]

    ax: float = 200.0
    ay: float = 200.0
    az: float = 200.0
    nu: float = 0.3
    sigma: float = 0.05


def _c_max(
    c: typing.Union[typing.Tuple[float, float], np.ndarray]
) -> typing.Tuple[float, float, float]:
    """
    Extract k[x] and max_[x] depending on how the VonKarman parameters were passed.

    Parameters
    ----------
    c
        Either a numpy array of positions or a tuple of (kx, x_max).

    Returns
    -------
        A tuple of (kx, x_min, x_max).
    """
    return (
        (c[0], 0, c[1])
        if isinstance(c, tuple)
        else (1 / (2 * (c[1] - c[0])), c.min(), np.ptp(c))
    )


@njit(parallel=True)
def _radial_psd_2d(kx, ky, ax, ay, nu):
    """2-D radial PSD. JIT compiled with Numba for performance."""
    k = np.sqrt(kx**2 * ax**2 + ky**2 * ay**2)
    return (4.0 * np.pi * nu * ax * ay) / (np.power(1 + k**2, nu + 1))


@njit(parallel=True)
def _radial_psd_3d(kx, ky, kz, ax, ay, az, nu):
    """3-D radial PSD. JIT compiled with Numba for performance."""
    k = np.sqrt(kx**2 * ax**2 + ky**2 * ay**2 + kz**2 * az**2)
    gf = gamma(nu + 1.5) / gamma(nu)
    return (
        gf
        * (8.0 * np.pi * np.sqrt(np.pi) * ax * ay * az)
        / np.power((1.0 + k**2), nu + 1.5)
    )


@njit(cache=True, parallel=True)
def _medium_2d(
    shape: typing.Tuple[int, int],
    psd_root: float,
    sigma: float,
    g: np.random.Generator,
) -> np.ndarray:
    """
    Compute the filter in the frequency domain.

    Unfortunately I cannot figure out how to make this generic over dimension as
    Numba needs a specific type annotation for the shape. I am sure its possible
    somehow though.

    Parameters
    ----------
    shape
        Shape of the array.
    psd_root
        PSD root.
    sigma
        Sigma.
    g
        Random number generator.

    Returns
    -------
        Filtered medium.
    """

    wn = g.random(shape)

    with objmode(ns="complex128[:,:]"):
        ns = fftn(wn, workers=-1)
        ns = fftshift(ns)

    ns /= np.abs(ns)
    c_psd = psd_root * ns

    with objmode(ns="complex128[:,:]"):
        m = ifftshift(c_psd)
        m = ifftn(m, workers=-1)

    m -= np.mean(m)
    m_std = np.std(m)

    return np.real(m) * sigma / m_std


@njit(cache=True, parallel=True)
def _medium_3d(
    shape: typing.Tuple[int, int, int],
    psd_root: float,
    sigma: float,
    g: np.random.Generator,
) -> np.ndarray:
    """
    Compute the filter in the frequency domain.

    Unfortunately I cannot figure out how to make this generic over dimension as
    Numba needs a specific type annotation for the shape. I am sure its possible
    somehow though.

    Parameters
    ----------
    shape
        Shape of the array.
    psd_root
        PSD root.
    sigma
        Sigma.
    g
        Random number generator.

    Returns
    -------
        Filtered medium.
    """

    wn = g.random(shape)

    with objmode(ns="complex128[:,:,:]"):
        ns = fftn(wn, workers=-1)
        ns = fftshift(ns)

    ns /= np.abs(ns)
    c_psd = psd_root * ns

    with objmode(m="complex128[:,:,:]"):
        m = ifftshift(c_psd)
        m = ifftn(m, workers=-1)

    m -= np.mean(m)
    m_std = np.std(m)

    return np.real(m) * sigma / m_std


def von_karman_2d(p: VonKarmanParams2D, g: np.random.Generator) -> xr.DataArray:
    """
    Compute the von_karman medium.

    Parameters
    ----------
    p
        Filter parameters.
    g
        Random number generator.

    Returns
    -------
        The Von Karman medium as an xarray DataArray.
    """

    kx_max, x_min, x_max = _c_max(p.x)
    ky_max, y_min, y_max = _c_max(p.y)
    dkx, dky = 1 / x_max, 1 / y_max
    kx = np.arange(-kx_max, kx_max + dkx, dkx)
    ky = np.arange(-ky_max, ky_max + dky, dky)

    dx, dy = 1 / (2 * kx_max), 1 / (2 * ky_max)
    x = np.arange(0, x_max, dx)
    y = np.arange(0, y_max, dy)
    len_x = min(len(x), len(kx))
    len_y = min(len(y), len(ky))

    kx, x = kx[:len_x], x[:len_x]
    ky, y = ky[:len_y], y[:len_y]

    kxv, kyv = np.meshgrid(kx, ky, indexing="ij")

    psd = _radial_psd_2d(kxv, kyv, p.ax, p.ay, p.nu)
    psd_root = np.sqrt(psd)

    return xr.DataArray(
        _medium_2d(kxv.shape, psd_root, p.sigma, g),
        [("x", x + x_min), ("y", y + y_min)],
        attrs={
            "kx_max": kx_max,
            "ky_max": ky_max,
            "a_x": p.ax,
            "a_y": p.ay,
            "Hurst nu": p.nu,
            "unit_xy": "m",
        },
    )


def von_karman_3d(p: VonKarmanParams3D, g: np.random.Generator) -> xr.DataArray:
    """
    Compute the von_karman medium.

    Parameters
    ----------
    p
        Filter parameters.
    g
        Random number generator.

    Returns
    -------
        The Von Karman medium as an xarray DataArray.
    """

    kx_max, x_min, x_max = _c_max(p.x)
    ky_max, y_min, y_max = _c_max(p.y)
    kz_max, z_min, z_max = _c_max(p.z)
    dkx, dky, dkz = 1 / x_max, 1 / y_max, 1 / z_max
    kx = np.arange(-kx_max, kx_max + dkx, dkx)
    ky = np.arange(-ky_max, ky_max + dky, dky)
    kz = np.arange(-kz_max, kz_max + dkz, dkz)

    dx, dy, dz = 1 / (2 * kx_max), 1 / (2 * ky_max), 1 / (2 * kz_max)
    print("dx, dy, dz: ", dx, dy, dz)
    print("dkx, dky, dkz: ", dkx, dky, dkz)
    x = np.arange(0, x_max, dx)
    y = np.arange(0, y_max, dy)
    z = np.arange(0, z_max, dz)
    len_x = min(len(x), len(kx))
    len_y = min(len(y), len(ky))
    len_z = min(len(z), len(kz))

    kx, x = kx[:len_x], x[:len_x]
    ky, y = ky[:len_y], y[:len_y]
    kz, z = kz[:len_z], z[:len_z]

    kxv, kyv, kzv = np.meshgrid(kx, ky, kz, indexing="ij")

    psd = _radial_psd_3d(kxv, kyv, kzv, p.ax, p.ay, p.az, p.nu)
    psd_root = np.sqrt(psd)

    return xr.DataArray(
        _medium_3d(kxv.shape, psd_root, p.sigma, g),
        [("x", x + x_min), ("y", y + y_min), ("z", z + z_min)],
        attrs={
            "kx_max": kx_max,
            "ky_max": ky_max,
            "kz_max": kz_max,
            "a_x": p.ax,
            "a_y": p.ay,
            "a_z": p.az,
            "Hurst nu": p.nu,
            "unit_xy": "m",
        },
    )


def von_karman(
    p: typing.Union[VonKarmanParams2D, VonKarmanParams3D],
    r: np.random.Generator = np.random.default_rng(0),
):
    """
    Compute the von_karman medium.

    Copied from the SED reference implementation with some refactoring and JIT
    compiling for performance. Performance heavy parts should now run in
    parallel on as many cores as you have. Likely slower for very small inputs
    as the JIT stage isn't free, but should be substantially faster (and more
    memory efficient) for production situations.

    Parameters
    ----------
    p
        Filter parameters. Implementation will be dispatched based on whether
        these are 2- or 3-D.
    r
        A random seed that can be passed to ensure deterministic outcomes.

    Returns
    -------
        The Von Karman medium as a data array.
    """

    return (
        von_karman_2d(p, r) if isinstance(p, VonKarmanParams2D) else von_karman_3d(p, r)
    )
