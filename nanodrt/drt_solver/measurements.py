"Module containing the battery Measurement object for the simulation and fitting modules"

import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


class ImpedenceMeasurement(eqx.Module):
    """
    Class defining the battery measurement
    """

    # Real and imaginary Impedance arrays
    Z_re: jnp.ndarray
    Z_im: jnp.ndarray

    # Frequencies corresponding to Impedance Arrays
    f: jnp.array

    # time constant array used for simulations
    tau: jnp.ndarray = dataclasses.field(default=None)  # type: ignore

    # define init
    def __init__(self, Z_re=jnp.ndarray, Z_im=jnp.ndarray, f=jnp.ndarray) -> None:
        """
        Class defining the battery measurement

        Args:
            Z_re (jnp.ndarray): Array containing the real part of impedance. Defaults to jnp.ndarray.
            Z_im (jnp.ndarray):  Array containing the imaginary part of impedance. Defaults to jnp.ndarray.
            f (jnp.ndarray): Frequencies corresponding to the impedance. Defaults to jnp.ndarray.
        """

        # Impedences in Ohms
        self.Z_re = Z_re
        self.Z_im = Z_im

        # frequencies in hertz
        self.f = f

        # corresponding time constants in seconds
        self.tau = 1.0 / (2 * jnp.pi * f)

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
            f"{self.__class__.__name__}(Z_re=Range({Z_re_range}), "
            f"Z_im=Range({Z_im_range}), f=Range({f_range}))"
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
        if jnp.any(self.f < 0):
            raise ValueError("Frequencies must be non-negative")

        # Check for non-empty arrays
        if self.Z_re.size == 0 or self.Z_im.size == 0 or self.f.size == 0:
            raise ValueError("Z_re, Z_im, and f must not be empty")
