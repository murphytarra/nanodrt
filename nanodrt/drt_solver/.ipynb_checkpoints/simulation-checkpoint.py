"Module containing the DRT object for the simulation and fitting modules"

import equinox as eqx
import dataclasses
from typing import Optional

import jax.numpy as jnp
from jax import config

# Ask about typechecking again? Look up...
from nanodrt.drt_solver.drt import DRT
from nanodrt.drt_solver.solvers import TrapezoidalSolver, RBFSolver

config.update("jax_enable_x64", True)


class Simulation(eqx.Module):
    """
    Class setting up and running the simulation
    """
    
    # # Variable Declaration (Type annotations)
    # frequencies to simulate the DRT spectrum with
    f_vec: jnp.ndarray
    
    # log of tau values making up the DRT fit
    log_tau_vec: jnp.ndarray
    
    # R_inf and L_0
    R_inf: float
    L_0: float

    # Method of integration used to simulate DRT spectrum - at the moment this is either 'trapezoid' or 'rbf'
    integration_method: str = dataclasses.field(default="rbf")  # type: ignore

    # Optional attributes
    # # if integration method is rbf
    x: Optional[jnp.ndarray] = dataclasses.field(default=None)  # type: ignore
    A_matrices: Optional[jnp.ndarray] = dataclasses.field(default=None)  # type: ignore
    # # if integration method is trapezoidal
    gamma: Optional[jnp.ndarray] = dataclasses.field(default=None)  # type: ignore

    def __repr__(self) -> str:
        return (
            f"Simulation(log_tau_vec={self.log_tau_vec}, f_vec={self.f_vec}, "
            f"integration_method={self.integration_method})"
        )

    def __init__(
        self, f_vec: jnp.ndarray, log_tau_vec: jnp.ndarray, R_inf: float, L_0: float, integration_method: str = "rbf", x: Optional[jnp.ndarray] = None, A_matrices: Optional[jnp.ndarray] = None, gamma: Optional[jnp.ndarray] = None
    ) -> None:
        """Class for simulation of DRT spectrum

        Args:
            integration_method (str, optional): Method used to determine integral throughout simulation. Defaults to "trapezoid".
            f_vec (jnp.ndarray): frequencies used in Measurmment object to determine the impedence.

        """

        # Frequency array used for simulation
        self.f_vec = f_vec

        # Logarthm of time constants
        self.log_tau_vec = log_tau_vec
        
        # Series R and L
        self.R_inf = R_inf
        self.L_0 = L_0
        
        # Method of integration used when calculating DRT spectum
        self.integration_method = integration_method

        # Construct rbf or trapezoidal fit
        if self.integration_method == "rbf":
            if x is None or A_matrices is None:
                raise ValueError("x and A_matrices are required for integration_method='rbf'")
            self.x = x
            self.A_matrices = A_matrices
        elif self.integration_method == "trapezoid":
            if gamma is None:
                raise ValueError("gamma is required for integration_method='trapezoid'")
            self.gamma = gamma
        else:
            raise ValueError("Unsupported integration_method. Choose 'rbf' or 'trapezoid'.")

        #self.__validate_init__()

    # def __validate_init__(self) -> None:
    #     """Validate the initialization parameters."""
    #     if not isinstance(self.drt, DRT):
    #         raise TypeError(
    #             f"Expected drt to be an instance of DRT, got {type(self.drt)}"
    #         )
    #     if not isinstance(self.f_vec, jnp.ndarray):
    #         raise TypeError(
    #             f"Expected f_vec to be a jnp.ndarray, got {type(self.f_vec)}"
    #         )
    #     if self.f_vec.ndim != 1:
    #         raise ValueError(
    #             f"Expected f_vec to be a 1-dimensional array, got {self.f_vec.ndim} dimensions"
    #         )
    #     if not isinstance(self.integration_method, str):
    #         raise TypeError(
    #             f"Expected integration_method to be a string, got {type(self.integration_method)}"
    #         )
        # if self.integration_method != "trapezoid":
        #     raise ValueError(
        #         f"Unsupported integration method: {self.integration_method}"
        # )

    @eqx.filter_jit
    def run(
        self,
    ):
        """Function which selects and runs the solver to determine the impedance

        Returns:
            jnp.ndarray: Real and Imaginary impedance values
        """
        if self.integration_method == "rbf":
            integrals = RBFSolver(
                x = self.x,
                A_matrices = self.A_matrices
            )
            integration = integrals()
            Z_re = self.R_inf + integration[0]
            Z_im = (
                2 * jnp.pi * self.f_vec * self.L_0 + integration[1]
            )
        

        if self.integration_method == "trapezoid":
            integrals = TrapezoidalSolver(
                f_vec=self.f_vec, 
                log_tau_vec=self.log_tau_vec,
                gamma = self.gamma
            )
            integration = integrals()
            Z_re = self.R_inf + integration[0]
            Z_im = 2 * jnp.pi * self.f_vec * self.L_0 + integration[1]

        return Z_re, Z_im
