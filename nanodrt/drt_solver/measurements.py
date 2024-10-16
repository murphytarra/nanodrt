"Module containing the battery Measurement object for the simulation and fitting modules"

import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True) # enable 64 bit precision (default is 32)


class ImpedanceMeasurement(eqx.Module):
    """
    Class defining the battery measurement
    """

    # Real and imaginary Impedance arrays
    Z_re: jnp.ndarray
    Z_im: jnp.ndarray

    # Frequencies corresponding to Impedance Arrays
    f: jnp.ndarray

    # time constant array used for simulations
    tau: jnp.ndarray = dataclasses.field(default=None)  # type: ignore

    # define init
    def __init__(self, Z_re, Z_im, f) -> None:
        """
        Class defining the battery measurement

        Args:
            Z_re (array-like): Array containing the real part of impedance.
            Z_im (array-like): Array containing the imaginary part of impedance.
            f (array-like): Frequencies corresponding to the impedance.
        """

        # Convert to jnp.ndarray if not already
        self.Z_re = jnp.asarray(Z_re) if not isinstance(Z_re, jnp.ndarray) else Z_re
        self.Z_im = jnp.asarray(Z_im) if not isinstance(Z_im, jnp.ndarray) else Z_im
        self.f = jnp.asarray(f) if not isinstance(f, jnp.ndarray) else f

        # corresponding time constants in seconds
        self.tau = 1.0 / (2 * jnp.pi * self.f)

        self.__validate_init__()

    def __repr__(self) -> str:
        """
        Provide a representation that shows the ranges of Z_re, Z_im, and f.
        """

        Z_re_range = (
            f"min={self.Z_re.min()}, max={self.Z_re.max()}"
            if self.Z_re.size > 0
            else "empty"
        )

        Z_im_range = (
            f"min={self.Z_im.min()}, max={self.Z_im.max()}"
            if self.Z_im.size > 0
            else "empty"
        )

        f_range = (
            f"min={self.f.min()}, max={self.f.max()}" if self.f.size > 0 else "empty"
        )

        return (
            f"{self.__class__.__name__}(shape={self.f.shape}, "
            f"(Z_re=Range({Z_re_range}), Z_im=Range({Z_im_range}), f=Range({f_range}))"
        )

    def __validate_init__(self) -> None:
        """
        Validates the initial state of the Measurement instance.
        """

        # Check for correct types
        if not isinstance(self.Z_re, jnp.ndarray):
            raise TypeError("Z_re must be a jnp.ndarray")
        if not isinstance(self.Z_im, jnp.ndarray):
            raise TypeError("Z_im must be a jnp.ndarray")
        if not isinstance(self.f, jnp.ndarray):
            raise TypeError("f must be a jnp.ndarray")

        # Check for equal lengths
        if len(self.Z_re) != len(self.Z_im) or len(self.Z_re) != len(self.f):
            raise ValueError("Z_re, Z_im, and f must all be of the same length")

        # Check for non-negative frequencies
        if jnp.any(self.f <= 0):
            raise ValueError("Frequencies must be non-negative and non-zero")

        # Check for non-empty arrays
        if self.Z_re.size == 0 or self.Z_im.size == 0 or self.f.size == 0:
            raise ValueError("Z_re, Z_im, and f must not be empty")
