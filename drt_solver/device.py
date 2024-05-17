"Module containing the battery Measurement object for the simulation and fitting modules"

import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


class Measurement(eqx.Module):
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


class DRT(eqx.Module):
    """
    Class defining the DRT spectrum of the battery
    """

    # Resistance and Inductance of a battery device
    R_inf: float
    L_0: float

    # DRT spectrum for battery device
    gamma: jnp.ndarray

    # Corresponding time constants with DRT spectrum
    tau: jnp.array

    def __init__(
        self, R_inf=float, L_0=float, gamma=jnp.ndarray, tau=jnp.ndarray
    ) -> None:
        """
        Class defining the DRT spectrum of the battery

        Args:
            R_inf (_type_, optional): _description_. Defaults to float.
            L_0 (_type_, optional): _description_. Defaults to float.
            gamma (_type_, optional): _description_. Defaults to jnp.ndarray.
            tau (_type_, optional): _description_. Defaults to jnp.ndarray.
        """

        # Resistance and inductance of battery device
        self.R_inf = R_inf
        self.L_0 = L_0

        # drt spectrum of device
        self.gamma = jnp.abs(gamma)

        # time constants of device
        self.tau = tau

        # self.__validate__init__()

    def __repr__(self) -> str:
        """
        Provide a representation that shows the main properties of the DRT instance,
        including the range of the DRT spectrum and time constants.
        """

        # Format the range of gamma values if the array is not empty
        gamma_range = (
            f"min={self.gamma.min()}, max={self.gamma.max()}"
            if self.gamma.size > 0
            else "empty"
        )

        # Format the range of tau values if the array is not empty
        tau_range = (
            f"min={self.tau.min()}, max={self.tau.max()}"
            if self.tau.size > 0
            else "empty"
        )

        return (
            f"{self.__class__.__name__}(R_inf={self.R_inf}, L_0={self.L_0}, "
            f"DRT_spectrum=Range({gamma_range}), time_constants=Range({tau_range}))"
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
        if not isinstance(self.gamma, jnp.ndarray):
            raise TypeError("gamma must be a jnp.ndarray")
        if not isinstance(self.tau, jnp.ndarray):
            raise TypeError("tau must be a jnp.ndarray")

        # Check for non-negative resistance and inductance
        if self.R_inf < 0:
            raise ValueError("R_inf must be non-negative")
        if self.L_0 < 0:
            raise ValueError("L_0 must be non-negative")

        # Check for equal lengths of gamma and tau
        if len(self.gamma) != len(self.tau):
            raise ValueError("gamma and tau must be of the same length")

        # Check for non-empty arrays
        if self.gamma.size == 0 or self.tau.size == 0:
            raise ValueError("gamma and tau must not be empty")


class FittedSpectrum(eqx.Module):
    """
    Dataclass which contains fitted parameters and loss function values
    """

    # Parameters fitted during optimisation process
    params: jnp.ndarray

    # state information of optimisation process
    state: jnp.ndarray

    # time constants used in optimisation process
    tau: jnp.ndarray

    # Label parameters that have been fitted
    R_inf: float = dataclasses.field(default=None)  # type: ignore
    L_0: float = dataclasses.field(default=None)  # type: ignore
    drt: jnp.ndarray = dataclasses.field(default=None)  # type: ignore

    # Final value of loss function
    value: float = dataclasses.field(default=None)  # type: ignore

    def __init__(
        self, params: jnp.ndarray, state: jnp.ndarray, tau: jnp.ndarray
    ) -> None:

        self.params = params
        self.state = state
        self.tau = tau

        self.value = self.state.value

        self.R_inf = self.params[0]
        self.L_0 = self.params[1]
        self.drt = jnp.abs(self.params[2:])

    def __repr__(self) -> str:
        return (
            f"FittedSpectrum(params={self.params}, state={self.state}, tau={self.tau}, "
            f"R_inf={self.R_inf}, L_0={self.L_0}, drt={self.drt}, value={self.value})"
        )
