" Module containing the dataclasses for fitted parameters"

import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

from nanodrt.drt_solver.solvers import TrapezoidalSolver, RBFSolver, x_to_gamma, A_Matrices_Calculator
from nanodrt.drt_solver.simulation import Simulation

from nanodrt.drt_solver.drt import DRT

config.update("jax_enable_x64", True)


class FittedSpectrum(eqx.Module):
    """
    Dataclass which contains fitted parameters and loss function values
    """

    # Parameters fitted during optimisation process
    params: jnp.ndarray

    # state information of optimisation process
    state: jnp.ndarray

    # time constants used in optimisation process
    tau_vec: jnp.ndarray

    # Log of time constants.
    log_tau_vec: jnp.ndarray

    # Frequencies of which were fitted to
    f_vec: jnp.ndarray

    # Label parameters that have been fitted
    R_inf: float = dataclasses.field(default=None)  # type: ignore
    L_0: float = dataclasses.field(default=None)  # type: ignore
    gamma: jnp.ndarray = dataclasses.field(default=None)  # type: ignore

    # Final value of loss function
    value: float = dataclasses.field(default=None)  # type: ignore

    # Integration method used
    integration_method: str = dataclasses.field(default=None)  # type: ignore

    # Radial basis function used (if applicable)
    rbf_function: str = dataclasses.field(default=None)  # type: ignore
    
    # DRT object used in impedance simulation
    drt: DRT = dataclasses.field(default=None) # type: ignore
    
    x: jnp.ndarray = dataclasses.field(default=None)
    A_matrices: jnp.ndarray

    def __init__(
        self,
        params: jnp.ndarray,
        state: jnp.ndarray,
        tau_vec: jnp.ndarray,
        f_vec: jnp.ndarray,
        integration_method: str,
        rbf_function: str,
    ) -> None:
        """Dataclass for the fitted spectrum obtained in optimisation process

        Args:
            params (jnp.ndarray): final parameters obtained in optimisation process
            state (dict): state obtained in optimisation process
            tau (jnp.ndarray): time constants used in optimisation process
            f_vec (jnp.ndarray): frequencies used in optimisation process for the impedence measurement.
            integration_method (str): integration method used in optimisation process.
        """
        
        # Type of integration method used in optimisation process
        self.integration_method = integration_method
        # rbf_function used throughout simulation (only used if integration method was rbf)
        self.rbf_function = rbf_function

        # Optimised parameters for the DRT spectrum
        self.params = params
        self.state = state

        # Time constants used in optimisation process
        self.tau_vec = tau_vec
        self.log_tau_vec = jnp.log(tau_vec)

        # Final value of residuals in optimisation process
        self.value = self.state.value

        # Extract optimised values
        self.R_inf = jnp.abs(self.params[0])
        self.L_0 = jnp.abs(self.params[1])
        if self.integration_method == "trapezoid":
            # enforce non-negativity as was done in loss function
            self.gamma = jnp.abs(self.params[2:])
        elif self.integration_method == "rbf":
            # enforce non-negativity as was done in loss function
            self.x = jnp.abs(self.params[2:])
            out_tau_vec, self.gamma = x_to_gamma(x_vec=self.x, tau_map_vec = self.tau_vec, tau_vec = self.tau_vec, mu =5, rbf_function = self.rbf_function)

        # Frequencies used in optimisation process
        self.f_vec = f_vec
        
        self.drt = DRT(self.R_inf, self.L_0, self.gamma, self.tau_vec)
        
        self.A_matrices = A_Matrices_Calculator(f_vec=self.f_vec, log_tau_vec=self.log_tau_vec, rbf_function=self.rbf_function).A_matrices()

    def __repr__(self) -> str:
        return (
            f"FittedSpectrum(params={self.params}, state={self.state}, tau={self.tau_vec}, "
            f"R_inf={self.R_inf}, L_0={self.L_0}, drt={self.drt}, value={self.value})"
        )

    def simulate(self) -> jnp.ndarray:
        """Simulate the impedence from the optimised values.

        Returns:
            jnp.ndarray: Real and Imaginary impedences.
        """
        if self.integration_method == "trapezoid":
            simulation = Simulation(f_vec=self.f_vec, log_tau_vec=self.log_tau_vec, R_inf=self.R_inf, L_0=self.L_0, integration_method=self.integration_method, gamma=self.gamma)
        elif self.integration_method == "rbf":
            simulation = Simulation(f_vec=self.f_vec, log_tau_vec=self.log_tau_vec, R_inf=self.R_inf, L_0=self.L_0, integration_method=self.integration_method, x=self.x, A_matrices=self.A_matrices)
        Z_re, Z_im = simulation.run()
        return Z_re, Z_im
        
