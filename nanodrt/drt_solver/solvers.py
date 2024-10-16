import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

from jax import jit, vmap
from quadax import trapezoid

from nanodrt.drt_solver.drt import DRT

config.update("jax_enable_x64", True)


class RBFSolver(eqx.Module):
    """
    Class to calculate the integral of the DRT spectrum using the RBF Discretisation method. Do not use on its own - accessible through Simulation.
    """

    # DRT Object used to determine integrand
    drt: DRT

    # Range of frequencies used to determine integrand values
    f_vec: jnp.ndarray

    # logarithm of time constants used for integral
    log_t_vec: jnp.ndarray = dataclasses.field(default=None)  # type: ignore

    rbf_function: str = dataclasses.field(default="gaussian")  # type: ignore

    mu: jnp.ndarray = dataclasses.field(default=1.0)  # type: ignore

    @eqx.filter_jit
    def __call__(self) -> jnp.ndarray:

        # First we create A Matrix
        A_mat = self.A_matrix()

        # Obtain the real and imaginary Impedance integrals
        integral_re = A_mat[0]
        integral_im = A_mat[1]

        return jnp.array([integral_re, integral_im])

    @eqx.filter_jit
    def gaussian(self, y: float, mu: float) -> float:
        """
        Gaussian Kernal used in RBF discretisation

        Args:
            log_tau_m (jnp.ndarray): time constant for RBF to be evaluated at
            mu (float): constant used for gaussian filter - determines FWHM

        Returns:
            float: RBF kernal value
        """
        return jnp.exp(-((y) ** 2) / mu)

    @eqx.filter_jit
    def A_element_re(self, f_m: float, log_tau_n: float) -> float:
        """
        Calculate the A matrix real component for the RBF method for a given frequency and time constant.
        This corresponds to one element in the A matrix in total.

        Args:
            f_m (float): Frequency to calculate the A matrix
            log_tau_n (float): Time constant to calculate the A matrix

        Returns:
            float: Value of element of A matrix for specific frequency and time constant.
        """

        # Define y
        y = jnp.linspace(-50, 50, 500)

        # Checking which radial basis function to use
        if self.rbf_function == "gaussian":
            phi = self.gaussian(y, mu=self.mu) 

        exponent = y + jnp.log(f_m) + log_tau_n

        factor = 1.0 / (1 + (4.0 * (jnp.pi**2) * jnp.exp(2 * exponent)))
        d_y = jnp.abs(y[1] - y[0]) # width of small interval in discretized version of continuous integral
        return (phi * factor * d_y).sum(axis=-1)  # discretisation? careful with tau and ranges

    @eqx.filter_jit
    def A_element_im(self, f_m: float, log_tau_n: float) -> float:
        """
        Calculate the A matrix imaginary component for the RBF method for a given frequency and time constant.
        This corresponds to one element in the A matrix in total.

        Args:
            f_m (float): Frequency to calculate the A matrix
            log_tau_n (float): Time constant to calculate the A matrix

        Returns:
            float: Value of element of A matrix for specific frequency and time constant.
        """

        # Define y
        y = jnp.linspace(-50, 50, 500)

        # Checking which radial basis function to use
        if self.rbf_function == "gaussian":
            phi = self.gaussian(y, mu=self.mu) 

        exponent = y + jnp.log(f_m) + log_tau_n
        factor = (
            2.0
            * jnp.pi
            * jnp.exp(exponent)
            / (1 + (4.0 * (jnp.pi**2) * jnp.exp(2 * exponent)))
        )  # size (n, )
        d_y = jnp.abs(y[1] - y[0])

        return -(phi * factor * d_y).sum(axis=-1)

    @eqx.filter_jit
    def A_vector_re(self, f_m: float) -> jnp.ndarray:
        """
        Calculate row of A matrix for specific frequency

        Args:
            f_m (_type_): Frequency to calculate the A matrix row
        Returns:
            jnp.ndarray: Array which corresponds to the row of the A matrix for a specific frequency
        """
        A_vec = vmap(self.A_element_re, in_axes=(None, 0))(f_m, self.log_t_vec)
        return A_vec

    @eqx.filter_jit
    def A_vector_im(self, f_m: float) -> jnp.ndarray:
        """
        Calculate row of A matrix for specific frequency

        Args:
            f_m (_type_): Frequency to calculate the A matrix row
        Returns:
            jnp.ndarray: Array which corresponds to the row of the A matrix for a specific frequency
        """
        A_vec = vmap(self.A_element_im, in_axes=(None, 0))(f_m, self.log_t_vec)
        return A_vec

    @eqx.filter_jit
    def A_matrix(
        self,
    ) -> jnp.ndarray:
        """
        Function to calculate both real and imaginary A matrices.

        Returns:
            jnp.ndarray: Real and Imaginary A Matrices.
        """
        A_mat_re = vmap(self.A_vector_re, in_axes=(0))(self.f_vec)
        A_mat_im = vmap(self.A_vector_im, in_axes=(0))(self.f_vec)
        return jnp.array([A_mat_re, A_mat_im])

    def M_matrix():
        pass
