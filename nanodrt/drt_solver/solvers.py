import equinox as eqx
import dataclasses

import jax.numpy as jnp
from jax import config

from jax import jit, vmap
from quadax import trapezoid

from nanodrt.drt_solver.drt import DRT

config.update("jax_enable_x64", True)


class TrapezoidalSolver(eqx.Module):
    """
    Class to calculate the integral of the DRT spectrum using the Trapezoidal Rule. Do not use on its own - accessible through Simulation.
    """

    # Range of frequencies used to determine integrand values
    f_vec: jnp.ndarray

    # logarithm of time constants used for integral
    log_tau_vec: jnp.ndarray
    
    # gamma used to determine integrand
    gamma: jnp.ndarray

    @eqx.filter_jit
    def __call__(self) -> jnp.ndarray:

        # Determine the real integral using the trapezoidal rule
        integral_re = vmap(self.integrand_re, in_axes=(0))(self.f_vec)
        total_integral_re = trapezoid(self.log_tau_vec, integral_re)

        # Determine the imaginary integral using the trapezoidal rule
        integral_im = vmap(self.integrand_im, in_axes=(0))(self.f_vec)
        total_integral_im = trapezoid(self.log_tau_vec, integral_im)

        return jnp.array([total_integral_re, total_integral_im])

    @eqx.filter_jit
    def integrand_re(self, f) -> jnp.ndarray:
        """
        Real integrand calculation with respect to log_tau_vec

        Args:
            f (_type_): frequency at which to calculate the integrand with
        """

        omega = 2.0 * jnp.pi * f
        integrand = self.gamma / (1.0 + (omega * jnp.exp(self.log_tau_vec)) ** 2)
        return integrand

    @eqx.filter_jit
    def integrand_im(self, f) -> jnp.ndarray:
        """
        Imaginary integrand calculation with respect to log_tau_vec

        Args:
            f (_type_): frequency at which to calculate the integrand with
        """

        omega = 2.0 * jnp.pi * f
        integrand = -(self.gamma * omega * jnp.exp(self.log_tau_vec)) / (
            1 + (omega * jnp.exp(self.log_tau_vec)) ** 2
        )
        return integrand

class RBF(eqx.Module):
    """
    Class to act as the radial basis function for a given discretisation.
    """
    
    # shape factor of rbf function
    mu: float 
    
    # rbf function used for discretisation
    rbf_function: str = dataclasses.field(default="gaussian") # type: ignore
    
    @eqx.filter_jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mu = self.mu
        rbf_function = self.rbf_function
        if rbf_function == "gaussian":
            return jnp.exp(-(mu*x)**2)
        elif rbf_function == "C2":
            return jnp.exp(-jnp.abs(mu * x)) * (1 + jnp.abs(mu * x))
        elif rbf_function == "C4":
            return jnp.exp(-jnp.abs(mu * x)) * (
            1 + jnp.abs(mu * x) + (1.0 / 3.0) * jnp.abs(mu * x) ** 2)
        elif rbf_function == "C6":
            return jnp.exp(-jnp.abs(mu * x)) * (
            1 + jnp.abs(mu * x)
            + (2.0 / 5.0) * jnp.abs(mu * x) ** 2
            + (1.0 / 15.0) * jnp.abs(mu * x) ** 3)
        else:
            raise ValueError("Unsupported rbf_function")
    
class A_Matrices_Calculator(eqx.Module):
    """
    Class used to calculate the A matrices when rbf integration is used.
    """
    # frequencies of data fitting to
    f_vec: jnp.ndarray
    
    # log of time constants used in discretisation
    log_tau_vec: jnp.ndarray
    
    # function used for discretisation
    rbf_function: str = dataclasses.field(default="gaussian") # type: ignore
    
#     def __init__(self, f_vec, log_tau_vec, rbf_function):
#         """
#         created whilst removing eqx code whilst testing if A works
#         """
#         self.f_vec = f_vec
#         self.log_tau_vec = log_tau_vec
#         self.rbf_function = rbf_function
    
    @eqx.filter_jit
    def A_matrices(self) -> jnp.ndarray:
        """
        Function to calculate both real and imaginary A matrices.

        Returns:
            jnp.ndarray: Real and Imaginary A Matrices.
        """
        print("A matrices called")
        # implement Toeplitz later
        
        # Brute force
        A_mat_re = vmap(self.A_vector_re, in_axes=(0))(self.f_vec)
        A_mat_im = vmap(self.A_vector_im, in_axes=(0))(self.f_vec)
        print("A matrices complete")
        return jnp.array([A_mat_re, A_mat_im])
    
    @eqx.filter_jit
    def A_vector_re(self, f_n: float) -> jnp.ndarray:
        """
        Calculate row of A matrix for specific frequency

        Args:
            f_n (_type_): Frequency to calculate the A matrix row
        Returns:
            jnp.ndarray: Array which corresponds to the row of the A matrix for a specific frequency
        """
        A_vec = vmap(self.A_element_re, in_axes=(None, 0))(f_n, self.log_tau_vec)
        return A_vec

    @eqx.filter_jit
    def A_vector_im(self, f_n: float) -> jnp.ndarray:
        """
        Calculate row of A matrix for specific frequency

        Args:
            f_n (_type_): Frequency to calculate the A matrix row
        Returns:
            jnp.ndarray: Array which corresponds to the row of the A matrix for a specific frequency
        """
        A_vec = vmap(self.A_element_im, in_axes=(None, 0))(f_n, self.log_tau_vec)
        return A_vec

    @eqx.filter_jit
    def A_element_re(self, f_n: float, log_tau_m: float) -> float:
        """
        Calculate the A matrix real component for the RBF method for a given frequency and time constant.
        This corresponds to one element in the A matrix in total.

        Args:
            f_n (float): Frequency to calculate the A matrix
            log_tau_m (float): Time constant to calculate the A matrix

        Returns:
            float: Value of element of A matrix for specific frequency and time constant.
        """
        
        phi = RBF(mu=5, rbf_function = self.rbf_function)
        phi_vec = phi(log_tau_m-self.log_tau_vec) # careful here, according to Tara code, want to pass log_tau_m - log_tau_vec. if pass log_tau_vec to log_tau_m might just get a big array of zeros? Don't want element wise here not sure if autos to that.
        # or have I got this all wrong, she sort of did the same thing. Need to think about the dimensions. 

        factor = 1.0 / (
            1 + (2.0 * jnp.pi * jnp.exp(self.log_tau_vec) * f_n) ** 2
        )  # size (n, )
        d_log_t = jnp.abs(self.log_tau_vec[1] - self.log_tau_vec[0])
        return (phi_vec * factor * d_log_t).sum(axis=-1)

    #@eqx.filter_jit
    def A_element_im(self, f_n: float, log_tau_m: float) -> float:
        """
        Calculate the A matrix imaginary component for the RBF method for a given frequency and time constant.
        This corresponds to one element in the A matrix in total.

        Args:
            f_n (float): Frequency to calculate the A matrix
            log_tau_m (float): Time constant to calculate the A matrix

        Returns:
            float: Value of element of A matrix for specific frequency and time constant.
        """

        phi = RBF(mu=5, rbf_function = self.rbf_function)
        phi_vec = phi(log_tau_m-self.log_tau_vec) # careful here, according to Tara code, want to pass log_tau_m - log_tau_vec. if pass log_tau_vec to log_tau_m might just get a big array of zeros? Don't want element wise here not sure if autos to that.
        # or have I got this all wrong, she sort of did the same thing. Need to think about the dimensions. 

        factor = (
            2.0
            * jnp.pi
            * jnp.exp(self.log_tau_vec)
            * f_n
            / (1 + (2.0 * jnp.pi * jnp.exp(self.log_tau_vec) * f_n) ** 2)
        )  # size (n, )
        d_log_t = jnp.abs(self.log_tau_vec[1] - self.log_tau_vec[0])
        return -(phi_vec * factor * d_log_t).sum(axis=-1)


class RBFSolver(eqx.Module):
    """
    Class to calculate the integral of the DRT spectrum using the RBF Discretisation method. Do not use on its own - accessible through Simulation.
    """
    
    x: jnp.ndarray
    
    A_matrices: jnp.ndarray

    @eqx.filter_jit
    def __call__(self) -> jnp.ndarray:
        # Determine the real integral 
        total_integral_re = self.A_matrices[0] @ self.x

        # Determine the imaginary integral
        total_integral_im = self.A_matrices[1] @ self.x

        return jnp.array([total_integral_re, total_integral_im])

def x_to_gamma(
    x_vec: jnp.ndarray, 
    tau_map_vec: jnp.ndarray, 
    tau_vec: jnp.ndarray, 
    mu: float, 
    rbf_function: str = dataclasses.field(default="gaussian") # type: ignore
    ) -> jnp.ndarray: 
    
    """  
       This function maps the x vector to the gamma_vec
       Inputs:
            x_vec : the DRT vector obtained by mapping gamma_vec to x
            tau_map_vec : vector mapping x_vec to gamma_vec
            tau_vec : DRT vector
            mu: shape factor 
            rbf_type: selected RBF type
       Outputs: 
            tau_vec and gamma_vec obtained by mapping x to gamma
    """

    phi = RBF(mu=mu, rbf_function = rbf_function)
    N_taus = tau_vec.size
    N_tau_map = tau_map_vec.size
    gamma_vec = jnp.zeros([N_tau_map, 1])

    B = jnp.zeros([N_tau_map, N_taus])

    for p in range(0, N_tau_map):
        for q in range(0, N_taus):
            delta_log_tau = jnp.log(tau_map_vec[p])-jnp.log(tau_vec[q])
            #B[p,q] = phi(delta_log_tau) # apparently jax doesn't like this
            B = B.at[p, q].set(phi(delta_log_tau))

    gamma_vec = B@x_vec
    out_tau_vec = tau_map_vec 
        
    return out_tau_vec, gamma_vec



