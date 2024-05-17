import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

from jax import jit, vmap
from quadax import trapezoid

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
