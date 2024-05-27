"Module containing different methods of extracting the regularisation parameters "

import equinox as eqx
import jax.numpy as jnp
import jaxopt
import dataclasses

# Ask about typechecking again? Look up...
from nanodrt.drt_solver.drt import DRT
from nanodrt.drt_solver.measurements import ImpedenceMeasurement
from nanodrt.fitting.fits import FittedSpectrum
from nanodrt.drt_solver.simulaton import Simulation
from nanodrt.drt_solver.solvers import RBFSolver

class GCV(eqx.Module): 
    """ 
    Object which computes the regularisation parameter by minimizing the GCV function
    """
    measurement: ImpedenceMeasurement

    A_re: jnp.ndarray 

    A_im: jnp.ndarray

    M: jnp.ndarray  = dataclasses.field(default=None)  # type: ignore

    init_lbd: float  = dataclasses.field(default=None)  # type: ignore
    
    @eqx.filter_jit
    def __call__(self) -> float:

        solver = jaxopt.LBFGS(fun=self.compute_GCV, maxiter=50000)
        res = solver.run(init_params=jnp.array(self.init_lbd), A_re=self.A_re, A_im=self.A_im, measurement=self.measurement,  M=jnp.eye(self.A_re.shape[1] + 2))
        param, state = res
        return param

    @eqx.filter_jit
    def compute_GCV(self, init_params, A_re, A_im, measurement, M):
    
        """
        This function computes the score for the generalized cross-validation (GCV) approach.
        Reference: G. Wahba, A comparison of GCV and GML for choosing the smoothing parameter in the generalized spline smoothing problem, Ann. Statist. 13 (1985) 1378–1402.
        Inputs: 
            log_lambda: regularization parameter
            A_re: discretization matrix for the real part of the impedance
            A_im: discretization matrix for the real part of the impedance
            Z_re: vector of the real parts of the impedance
            Z_im: vector of the imaginary parts of the impedance
            M: differentiation matrix 
        Output:
            GCV score
        """
        
        lambda_value = jnp.exp(init_params)
        Z = jnp.hstack(( measurement.Z_re,measurement.Z_im)) # stacked impedance

        n_cv = Z.shape[0] # n_cv = 2*N_freqs with N_freqs the number of EIS frequencies
        n_t = A_re.shape[1]

        n_cv2 = int(n_cv/2)
        ones = jnp.ones(n_cv2)
        zeros = jnp.zeros(n_cv2)

        A_top = jnp.vstack((zeros, ones, A_re.T))
        A_bottom = jnp.vstack((2.*jnp.pi*measurement.f, zeros, A_im.T))
        A = jnp.hstack((A_top, A_bottom))
        #print(A.shape) # size (N + 2, 2M)
            
        #A = jnp.concatenate((A_re, A_im), axis = 0) # matrix A with A_re and A_im ; see (5) in [4]
    
        n_cv = Z.shape[0] # n_cv = 2*N_freqs with N_freqs the number of EIS frequencies
        
        A_agm = A@A.T + lambda_value*M # see (13) in [4]
        
        #print(A_agm.shape)
        L_agm = jnp.linalg.cholesky(A_agm) # Cholesky transform to inverse A_agm
        inv_L_agm = jnp.linalg.inv(L_agm)
        inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
        A_GCV = A.T@inv_A_agm@A  # see (13) in [4]
        # GCV score; see (13) in [4]
        GCV_num = 1/n_cv*jnp.linalg.norm((jnp.eye(n_cv)-A_GCV)@Z)**2 # numerator
        GCV_dom = (1/n_cv*jnp.trace(jnp.eye(n_cv)-A_GCV))**2 # denominator
        
        GCV_score = GCV_num/GCV_dom
        return GCV_score


