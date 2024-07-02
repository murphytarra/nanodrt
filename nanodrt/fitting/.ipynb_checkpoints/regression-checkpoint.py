"Module containing the Regression object which is used in the Optimisation Process "

import equinox as eqx
import jax # using for printing
import jax.numpy as jnp
import jaxopt
import dataclasses

# Ask about typechecking again? Look up...
from nanodrt.drt_solver.drt import DRT
from nanodrt.drt_solver.measurements import ImpedenceMeasurement
from nanodrt.fitting.fits import FittedSpectrum
from nanodrt.drt_solver.simulation import Simulation
from nanodrt.drt_solver.solvers import A_Matrices_Calculator, x_to_gamma
from nanodrt.fitting.regularization import GCV


class Regression(eqx.Module):
    """
    Class to perform Regression to determine DRT Spectrum
    """

    # Measurement object containing experimental data
    measurement: ImpedenceMeasurement

    # DRT Object containing spectrum and time constants intial guessses
    drt: DRT

    # Dictionary containing hyperparameters for solver
    solver_dict: dict[str, float]

    # Method of Integration used in Regression
    integration_method: str = dataclasses.field(default="rbf")  # type: ignore

    rbf_function: str = dataclasses.field(default="gaussian")  # type: ignore
    
    log_tau_vec: jnp.ndarray
    A_matrices: jnp.ndarray
    
    def __init__(self, measurement, drt, solver_dict, integration_method="rbf", rbf_function="gaussian"):
        
        self.measurement = measurement
        self.drt=drt
        self.solver_dict = solver_dict
        self.integration_method = integration_method
        self.rbf_function = rbf_function
        self.log_tau_vec = jnp.log(self.drt.tau)
        # Calculate A Matrices and save as attribute
        self.A_matrices = A_Matrices_Calculator(f_vec=self.measurement.f, log_tau_vec=self.log_tau_vec, rbf_function=self.rbf_function).A_matrices()
        print("initialisation of regression complete")
        print(self.A_matrices)
        #print(jax.debug.callback
        #print(jax.device_get(self.A_matrices))
        #jax.debug.print("A_matrices: {}", self.A_matrices)

    @eqx.filter_jit()
    def fit(
        self,
    ):
        """
        Calls Regression Operation

        Returns:
            Simulation: Simulation Object containing the optimized parameters
        """

        # # Calculate A matrices and save as self.A_matrix                          # I guess is not needed if trapezoid chosen? will be default rbf which is gaussian otherwise if set as none presumably some error thrown by rbf solver
        # integrals = RBFSolver(    
        #     drt=self.drt,
        #     f_vec=self.measurement.f,
        #     log_t_vec=jnp.log(self.drt.tau),
        #     rbf_function=self.rbf_function,
        # )
        # A_matrices = integrals()
        

        

        
        # Calculate M1 matrix - only 1st derivative implemented for now
        
        # Choose regression parameter
        init_lbd = self.solver_dict["init_lbd"]                                  # how does this work if trapezoidal chosen?
        # if self.solver_dict["lbd_selection"] == "GCV":
        #     # get the GCV computed value
        #     M = jnp.eye(A_matrices[0].shape[1] + 2)
        #     gcv = GCV(
        #         self.measurement,
        #         A_matrices[0],
        #         A_matrices[1],
        #         M,
        #         init_lbd,
        #     )
        #     init_lbd = jnp.exp(gcv())

        # Extract number of steps
        maxiter = self.solver_dict["maxiter"]

        if self.integration_method == "trapezoid":
            
            # Initial Parameters to be fitted in optimization process
            init_params = jnp.hstack((self.drt.R_inf, self.drt.L_0, self.drt.gamma))

        if self.integration_method == "rbf":
            
            x_initial = self.drt.gamma # use initial gammas as intial guess for x
            
            # Initial Parameters to be fitted in optimization process
            init_params = jnp.hstack((self.drt.R_inf, self.drt.L_0, x_initial))
        
        # define solver for new loss function
        solver = jaxopt.LBFGS(fun=self.loss_function, maxiter=maxiter)

        # Run solver
        res = solver.run(
            init_params=init_params,
            lbd=init_lbd,
        )

        # Extract parameters
        params, state = res

        # Create fitted spectrum object
        fit = FittedSpectrum(
            params,
            state,
            self.drt.tau,
            self.measurement.f,
            self.integration_method,
            self.rbf_function,
        )

        return fit

    @eqx.filter_jit()
    def loss_function(
        self,
        init_params: jnp.ndarray,
        lbd: float,
    ) -> float:
        """Loss Function minimized in regularisation process. Default is Tikhonov Regularisation

        Args:
            init_params (jnp.ndarray): Initial parameters used when fitting
            tau (jnp.ndarray): Time constants used in integration method
            lbd (float): regularisation parameter

        Returns:
            float: Return residuals of loss function calculated.
        """

        # Extract parameters
        R_inf = jnp.abs(init_params[0])
        L_0 = jnp.abs(init_params[1])
        
        if self.integration_method == "rbf":
            # Here we take the absolute values of x to ensure non negativity for the resulting gamma
            x= jnp.abs(init_params[2:])
            simulation = Simulation(f_vec=self.measurement.f, log_tau_vec=self.log_tau_vec, R_inf=R_inf, L_0=L_0, integration_method=self.integration_method, x=x, A_matrices=self.A_matrices)
            # calculate gamma for basic regularisation method - can change later
            out_tau_vec, gamma = x_to_gamma(x_vec=x, tau_map_vec = self.drt.tau, tau_vec = self.drt.tau, mu =5, rbf_function = self.rbf_function)
    
        elif self.integration_method == "trapezoid":
            # Here we take the absolute values of gamma to ensure non negativity
            gamma = jnp.abs(init_params[2:])
            simulation = Simulation(f_vec=self.measurement.f, log_tau_vec=self.log_tau_vec, R_inf=R_inf, L_0=L_0, integration_method=self.integration_method, gamma=gamma)
            
        # Obtain impedance values from simulation
        Z_re, Z_im = simulation.run()
        
        # calculate residuals with regularisation
        residuals = jnp.sum(
            (Z_re - self.measurement.Z_re) ** 2 + (Z_im - self.measurement.Z_im) ** 2
        ) + (lbd**2) * jnp.sum(gamma**2)

        return residuals