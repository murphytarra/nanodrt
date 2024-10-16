"Module containing the DRT object for the simulation and fitting modules"


import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

# Ask about typechecking again? Look up...
from nanodrt.drt_solver.drt import DRT
from nanodrt.drt_solver.solvers import RBFSolver


config.update("jax_enable_x64", True)


class Simulation(eqx.Module):
    """
    Class setting up and running the simulation
    """

    # dataclass containing DRT spectrum
    drt: DRT

    # frequencies to simulate the DRT spectrum with
    f_vec: jnp.ndarray

    # Method of integration used to simulate DRT spectrum - at the moment this is either 'trapezoid' or 'rbf'
    integration_method: str = dataclasses.field(default="trapezoid")  # type: ignore

    # Logarithm of tau vector
    log_t_vec: jnp.array = dataclasses.field(default=None)  # type: ignore

    # RBF Function to use
    rbf_function: jnp.array = dataclasses.field(default=None)  # type: ignore

    def __repr__(self) -> str:
        return (
            f"Simulation(drt={self.drt}, f_vec={self.f_vec}, "
            f"integration_method={self.integration_method})"
        )

    def __init__(
        self, drt: DRT, f_vec: jnp.ndarray, integration_method: str = "rbf"
    ) -> None:
        """Class for simulation of DRT spectrum

        Args:
             drt (DRT): DRT object containing drt spectrum and time constants
            integration_method (str, optional): Method used to determine integral throughout simulation. Defaults to "trapezoid".
            f_vec (jnp.ndarray): frequencies used in Measurement object to determine the impedance.

        """

        # DRT object containing the DRT spectrum and time constants
        self.drt = drt

        # Frequency array used for simulation
        self.f_vec = f_vec

        # Method of integration used when calculating DRT spectrum
        self.integration_method = integration_method

        # Logarithm of time constants
        self.log_t_vec = jnp.log(self.drt.tau)

        self.__validate_init__()

    def __validate_init__(self) -> None:
        """Validate the initialization parameters."""
        if not isinstance(self.drt, DRT):
            raise TypeError(
                f"Expected drt to be an instance of DRT, got {type(self.drt)}"
            )
        if not isinstance(self.f_vec, jnp.ndarray):
            raise TypeError(
                f"Expected f_vec to be a jnp.ndarray, got {type(self.f_vec)}"
            )
        if self.f_vec.ndim != 1:
            raise ValueError(
                f"Expected f_vec to be a 1-dimensional array, got {self.f_vec.ndim} dimensions"
            )
        if not isinstance(self.integration_method, str):
            raise TypeError(
                f"Expected integration_method to be a string, got {type(self.integration_method)}"
            )
        # if self.integration_method != "trapezoid":
        #     raise ValueError(
        #         f"Unsupported integration method: {self.integration_method}"
        # )

    @eqx.filter_jit
    def run(
        self,
    ):
        """Function which selects and runs the solver to determine the Impedance

        Returns:
            jnp.ndarray: Real and Imaginary Impedance values
        """

        if self.integration_method == "rbf":
            integrals = RBFSolver(
                drt=self.drt,
                f_vec=self.f_vec,
                log_t_vec=self.log_t_vec,
            )
            integration = integrals()
            Z_re = self.drt.R_inf + integration[0] @ self.drt.x
            Z_im = 2 * jnp.pi * self.f_vec * self.drt.L_0 + integration[1] @ self.drt.x
        return Z_re, Z_im
