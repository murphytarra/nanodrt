" Module containing the dataclasses for fitted parameters"

import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

from nanodrt2.drt_solver.solvers import RBFSolver

from nanodrt2.drt_solver.drt import DRT

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
    tau: jnp.ndarray

    # Log of time constants.
    log_t_vec: jnp.ndarray

    # Frequencies of which were fitted to
    f_vec: jnp.ndarray

    # Label parameters that have been fitted
    R_inf: float = dataclasses.field(default=None)  # type: ignore
    L_0: float = dataclasses.field(default=None)  # type: ignore
    x: jnp.ndarray = dataclasses.field(default=None)  # type: ignore

    # Final value of loss function
    value: float = dataclasses.field(default=None)  # type: ignore

    # Integration method used
    integration_method: str = dataclasses.field(default=None)  # type: ignore

    rbf_function: str = dataclasses.field(default=None)  # type: ignore
    mu: float = dataclasses.field(default=None)  # type: ignore

    def __init__(
        self,
        params: jnp.ndarray,
        state: jnp.ndarray,
        tau: jnp.ndarray,
        f_vec: jnp.ndarray,
        integration_method: str,
        rbf_function: str,
        mu: float,
    ) -> None:
        """Dataclass for the fitted spectrum obtained in optimisation process

        Args:
            params (jnp.ndarray): final parameters obtained in optimisation process
            state (dict): state obtained in optimisation process
            tau (jnp.ndarray): time constants used in optimisation process
            f_vec (jnp.ndarray): frequencies used in optimisation process for the Impedance measurement.
            integration_method (str): integration method used in optimisation process.
        """

        # Optimised parameters for the DRT spectrum
        self.params = params
        self.state = state

        # Time constants used in optimisation process
        self.tau = tau
        self.log_t_vec = jnp.log(tau)

        # Final value of residuals in optimisation process
        self.value = self.state.value

        # Extract optimised values
        self.R_inf = self.params[0]
        self.L_0 = self.params[1]
        self.x = jnp.abs(self.params[2:])

        # Frequencies used in optimisation process
        self.f_vec = f_vec

        # Type of integration method used in optimisation process
        self.integration_method = integration_method

        # rbf_function used throughout simulation
        self.rbf_function = rbf_function
        self.mu = mu

    def __repr__(self) -> str:
        return (
            f"FittedSpectrum(params={self.params}, state={self.state}, tau={self.tau}, "
            f"R_inf={self.R_inf}, L_0={self.L_0}, value={self.value})"
        )

    def simulate(self) -> jnp.ndarray:
        """Simulate the Impedance from the optimised values.

        Returns:
            jnp.ndarray: Real and Imaginary Impedances.
        """
        drt = DRT(self.R_inf, self.L_0, self.x, self.tau)

        if self.integration_method == "rbf":
            integrals = RBFSolver(
                drt=drt,
                f_vec=self.f_vec,
                log_t_vec=self.log_t_vec,
            )
            integration = integrals()
            Z_re = self.R_inf + integration[0] @ self.x
            Z_im = 2 * jnp.pi * self.f_vec * self.L_0 + integration[1] @ self.x
        return Z_re, Z_im
