import equinox as eqx
import jax.numpy as jnp
from jax import jit, vmap
from quadax import trapezoid
import jax


class Measurement(eqx.Module):
    tau_vec: jax.Array
    Z_re: jax.Array
    Z_im: jax.Array

    def __init__(self, tau, Z_re, Z_im):
        self.tau_vec = tau
        self.Z_re = Z_re
        self.Z_im = Z_im
