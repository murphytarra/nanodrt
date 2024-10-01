"Module containing the drt spectrum for the simulation and fitting modules"

import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


class DRT(eqx.Module):
    """
    Class defining the DRT spectrum of the battery
    """

    # Resistance and Inductance of a battery device
    R_inf: float
    L_0: float

    # DRT spectrum for battery device
    x: jnp.ndarray

    # Corresponding time constants with DRT spectrum
    tau: jnp.array

    def __init__(self, R_inf=float, L_0=float, x=jnp.ndarray, tau=jnp.ndarray) -> None:
        """
        Class defining the DRT spectrum of the battery

        Args:
            R_inf (_type_, optional): _description_. Defaults to float.
            L_0 (_type_, optional): _description_. Defaults to float.
            x (_type_, optional): _description_. Defaults to jnp.ndarray.
            tau (_type_, optional): _description_. Defaults to jnp.ndarray.
        """

        # Resistance and inductance of battery device
        self.R_inf = R_inf
        self.L_0 = L_0

        # drt spectrum of device
        self.x = jnp.abs(x)

        # time constants of device
        self.tau = tau

        # self.__validate__init__()

    def __repr__(self) -> str:
        """
        Provide a representation that shows the main properties of the DRT instance,
        including the range of the DRT spectrum and time constants.
        """

        # Format the range of x values if the array is not empty
        x_range = (
            f"min={self.x.min()}, max={self.x.max()}" if self.x.size > 0 else "empty"
        )

        # Format the range of tau values if the array is not empty
        tau_range = (
            f"min={self.tau.min()}, max={self.tau.max()}"
            if self.tau.size > 0
            else "empty"
        )

        return (
            f"{self.__class__.__name__}(R_inf={self.R_inf}, L_0={self.L_0}, "
            f"DRT_spectrum=Range({x_range}), time_constants=Range({tau_range}))"
        )

    def __validate__init__(self) -> None:
        """
        Validates the initial state of the DRT instance.
        """
        # Check for correct types
        if not isinstance(self.R_inf, (int, float)):
            raise TypeError("R_inf must be a number")
        if not isinstance(self.L_0, (int, float)):
            raise TypeError("L_0 must be a number")
        if not isinstance(self.x, jnp.ndarray):
            raise TypeError("x must be a jnp.ndarray")
        if not isinstance(self.tau, jnp.ndarray):
            raise TypeError("tau must be a jnp.ndarray")

        # Check for non-negative resistance and inductance
        if self.R_inf < 0:
            raise ValueError("R_inf must be non-negative")
        if self.L_0 < 0:
            raise ValueError("L_0 must be non-negative")

        # Check for equal lengths of x and tau
        if len(self.x) != len(self.tau):
            raise ValueError("x and tau must be of the same length")

        # Check for non-empty arrays
        if self.x.size == 0 or self.tau.size == 0:
            raise ValueError("x and tau must not be empty")
