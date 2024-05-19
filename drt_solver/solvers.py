import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

from jax import jit, vmap
from quadax import trapezoid, simpson

# Ask about typechecking again? Look up...
from drt_solver.device import Measurement, DRT

config.update("jax_enable_x64", True)


class TrapezoidalSolver(eqx.Module):
    """
    Class to calculate the integral of the DRT spectrum using the Trapezoidal Rule. Do not use on its own - accessible through Simulation.
    """

    # DRT Object used to determine integrand
    drt: DRT

    # Range of frequencies used to determine integrand values
    f_vec: jnp.ndarray

    # logarithm of time constants used for integral
    log_t_vec: jnp.ndarray = dataclasses.field(default=None)  # type: ignore

    @eqx.filter_jit
    def __call__(self) -> jnp.ndarray:

        integral_re = vmap(self.integrand_re, in_axes=(0))(self.f_vec)
        total_integral_re = trapezoid(self.log_t_vec, integral_re)

        integral_im = vmap(self.integrand_im, in_axes=(0))(self.f_vec)
        total_integral_im = trapezoid(self.log_t_vec, integral_im)

        return jnp.array([total_integral_re, total_integral_im])

    @eqx.filter_jit
    def integrand_re(self, f) -> jnp.ndarray:
        """
        Real iintegrand calculation with respect to log_tau_vec

        Args:
            f (_type_): frequency at which to calculate the integrand with
        """

        omega = 2.0 * jnp.pi * f
        integrand = self.drt.gamma / (1.0 + (omega * jnp.exp(self.log_t_vec)) ** 2)
        return integrand

    @eqx.filter_jit
    def integrand_im(self, f) -> jnp.ndarray:
        """
        Imaginary iintegrand calculation with respect to log_tau_vec

        Args:
            f (_type_): frequency at which to calculate the integrand with
        """

        omega = 2.0 * jnp.pi * f
        integrand = -(self.drt.gamma * omega * jnp.exp(self.log_t_vec)) / (
            1 + (omega * jnp.exp(self.log_t_vec)) ** 2
        )
        return integrand


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

    @eqx.filter_jit
    def __call__(self) -> jnp.ndarray:

        # First we create A Matrix
        A_mat = self.A_matrix()

        integral_re = A_mat[0] @ self.drt.gamma
        integral_im = A_mat[1] @ self.drt.gamma

        return jnp.array([integral_re, integral_im])

    @eqx.filter_jit
    def gaussian(self, log_tau_m, mu) -> float:
        return jnp.exp(-((mu * (log_tau_m - self.log_t_vec)) ** 2))

    @eqx.filter_jit
    def A_element_re(self, f_m: float, log_tau_n: float) -> float:
        phi = self.gaussian(log_tau_n, mu=5.0)  # size (n, )
        factor = 1.0 / (
            1 + (2.0 * jnp.pi * jnp.exp(self.log_t_vec) * f_m) ** 2
        )  # size (n, )
        d_log_t = jnp.abs(self.log_t_vec[1] - self.log_t_vec[0])
        return (phi * factor * d_log_t).sum(axis=-1)

    @eqx.filter_jit
    def A_element_im(self, f_m: float, log_tau_n: float) -> float:
        phi = self.gaussian(log_tau_n, mu=5.0)  # size (n, )
        factor = (
            2.0
            * jnp.pi
            * jnp.exp(self.log_t_vec)
            * f_m
            / (1 + (2.0 * jnp.pi * jnp.exp(self.log_t_vec) * f_m) ** 2)
        )  # size (n, )
        d_log_t = jnp.abs(self.log_t_vec[1] - self.log_t_vec[0])
        return -(phi * factor * d_log_t).sum(axis=-1)

    @eqx.filter_jit
    def A_vector_re(self, f_m) -> jnp.ndarray:
        A_vec = vmap(self.A_element_re, in_axes=(None, 0))(f_m, self.log_t_vec)
        return A_vec

    @eqx.filter_jit
    def A_vector_im(self, f_m) -> jnp.ndarray:
        A_vec = vmap(self.A_element_im, in_axes=(None, 0))(f_m, self.log_t_vec)
        return A_vec

    @eqx.filter_jit
    def A_matrix(
        self,
    ) -> jnp.ndarray:
        A_mat_re = vmap(self.A_vector_re, in_axes=(0))(self.f_vec)
        A_mat_im = vmap(self.A_vector_im, in_axes=(0))(self.f_vec)
        return jnp.array([A_mat_re, A_mat_im])
