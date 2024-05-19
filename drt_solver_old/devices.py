import equinox as eqx
import jax.numpy as jnp
from jax import jit, vmap
from quadax import trapezoid
import jax


class Device(eqx.Module):
    """
    Class defining battery device
    """

    R_inf: float
    R_ct: float
    tau_0: float
    L_0: float
    phi: float
    f_vec: jax.Array
    t_vec: jax.Array
    gamma: jax.Array
    log_t_vec: jax.Array

    def __init__(
        self,
        R_inf: float,
        R_ct: float,
        tau_0: float,
        L_0: float,
        phi: float,
        f_vec: jnp.ndarray,
        t_vec: jnp.ndarray,
        gamma: jnp.ndarray,
    ) -> None:
        """
        _summary_

        Args:
            R_inf (float): _description_
            R_ct (float): _description_
            tau_0 (float): _description_
            L_0 (float): _description_
            phi (float): _description_
            f_vec (jnp.ndarray): _description_
            t_vec (jnp.ndarray): _description_
            gamma (jnp.ndarray): _description_
        """

        self.R_ct = R_ct
        self.R_inf = R_inf
        self.tau_0 = tau_0
        self.L_0 = L_0
        self.phi = phi
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
    def obtain_Z(
        self,
    ):
        integral_re = vmap(self.integrand_re, in_axes=(0))(self.f_vec)
        Z_re = trapezoid(self.log_t_vec, integral_re) + self.R_inf

        integral_im = vmap(self.integrand_im, in_axes=(0))(self.f_vec)
        Z_im = (
            trapezoid(self.log_t_vec, integral_im) + 2 * jnp.pi * self.f_vec * self.L_0
        )
        return Z_re, Z_im
