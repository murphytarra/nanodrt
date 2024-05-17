import equinox as eqx
import jax.numpy as jnp
from jax import jit, vmap
from quadax import trapezoid
import jax


class Characterise(eqx.Module):
    """
    Class used to measure battery device
    """

    f_vec: jax.Array
    t_vec: jax.Array
    gamma: jax.Array
    log_t_vec: jax.Array

    def __init__(self, f_vec: jnp.ndarray, t_vec: jnp.ndarray, gamma: jnp.ndarray):
        """
        Class used to measure battery device for given frequency and gamma

        Args:
            f_vec (jnp.ndarray): Frequency in Hertz
            t_vec (jnp.ndarray): Relaxation Time
            gamma (jnp.ndarray): Distribution of Relaxation Times
        """
        self.f_vec = f_vec
        self.t_vec = t_vec
        self.gamma = gamma
        self.log_t_vec = jnp.log(t_vec)

    @eqx.filter_jit
    def integrand_re(self, f):
        omega = 2.0 * jnp.pi * f
        integrand = self.gamma / (1.0 + (omega * jnp.exp(self.log_t_vec)) ** 2)
        return integrand

    @eqx.filter_jit
    def integrand_im(self, f):
        omega = 2.0 * jnp.pi * f
        integrand = -(self.gamma * omega * jnp.exp(self.log_t_vec)) / (
            1 + (omega * jnp.exp(self.log_t_vec)) ** 2
        )
        return integrand

    @eqx.filter_jit
    def obtain_Z(self, device):
        integral_re = vmap(self.integrand_re, in_axes=(0))(self.f_vec)
        Z_re = trapezoid(self.log_t_vec, integral_re) + device.R_inf

        integral_im = vmap(self.integrand_im, in_axes=(0))(self.f_vec)
        Z_im = (
            trapezoid(self.log_t_vec, integral_im)
            + +2 * jnp.pi * self.f_vec * device.L_0
        )
        return Z_re, Z_im


class Battery(eqx.Module):
    """
    Class used to measure battery device
    """

    R_inf: float
    L_0: float
    f_vec: jax.Array
    t_vec: jax.Array
    gamma: jax.Array
    log_t_vec: jax.Array

    def __init__(
        self,
        f_vec: jnp.ndarray,
        t_vec: jnp.ndarray,
        gamma: jnp.ndarray,
        R_inf: float,
        L_0: float,
    ):
        """
        Class used to measure battery device for given frequency and gamma

        Args:
            f_vec (jnp.ndarray): Frequency in Hertz
            t_vec (jnp.ndarray): Relaxation Time
            gamma (jnp.ndarray): Distribution of Relaxation Times
        """
        self.f_vec = f_vec
        self.t_vec = t_vec
        self.gamma = gamma
        self.log_t_vec = jnp.log(t_vec)
        self.R_inf = R_inf
        self.L_0 = L_0

    @eqx.filter_jit
    def integrand_re(self, f):
        omega = 2.0 * jnp.pi * f
        integrand = self.gamma / (1.0 + (omega * jnp.exp(self.log_t_vec)) ** 2)
        return integrand

    @eqx.filter_jit
    def integrand_im(self, f):
        omega = 2.0 * jnp.pi * f
        integrand = -(self.gamma * omega * jnp.exp(self.log_t_vec)) / (
            1 + (omega * jnp.exp(self.log_t_vec)) ** 2
        )
        return integrand

    @eqx.filter_jit
    def obtain_Z(self):
        integral_re = vmap(self.integrand_re, in_axes=(0))(self.f_vec)
        Z_re = trapezoid(self.log_t_vec, integral_re) + self.R_inf

        integral_im = vmap(self.integrand_im, in_axes=(0))(self.f_vec)
        Z_im = (
            trapezoid(self.log_t_vec, integral_im) + 2 * jnp.pi * self.f_vec * self.L_0
        )
        return Z_re, Z_im
