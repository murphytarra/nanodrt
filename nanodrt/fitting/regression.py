"Module containing the Regression object which is used in the Optimisation Process "

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
from nanodrt.fitting.regularization import GCV


class Regression(eqx.Module):
    """
    Class to perform Regression to determine DRT Spectrum
    """

    # Measurement object containing experimental data
    measurement: ImpedenceMeasurement

    # DRT Object containing spectrum and time constants
    drt: DRT

    # Dictionary containing hyperparameters for solver
    solver_dict: dict[str, float]

    # Method of Integration used in Regression
    integration_method: str = dataclasses.field(default="trapezoid")  # type: ignore

    rbf_function: str = dataclasses.field(default=None)  # type: ignore

    @eqx.filter_jit()
    def __call__(
        self,
    ):
        """
        Calls Regression Operation

        Returns:
            Simulation: Simulation Object containing the optimized parameters
        """

        # Calculate A matrices and save as self.A_matrix
        integrals = RBFSolver(
            drt=self.drt,
            f_vec=self.measurement.f,
            log_t_vec=jnp.log(self.drt.tau),
            rbf_function=self.rbf_function,
        )
        A_matrices = integrals()

        init_lbd = self.solver_dict["init_lbd"]
        if self.solver_dict["lbd_selection"] == "GCV":
            # get the GCV computed value
            M = jnp.eye(A_matrices[0].shape[1] + 2)
            gcv = GCV(
                self.measurement,
                A_matrices[0],
                A_matrices[1],
                M,
                init_lbd,
            )
            init_lbd = jnp.exp(gcv())

        # Extract number of steps
        maxiter = self.solver_dict["maxiter"]

        # Initial Parameters to be fitted in optimization process
        init_params = jnp.hstack((self.drt.R_inf, self.drt.L_0, self.drt.gamma))

        if self.integration_method == "trapezoid":
            # Perform solver LBFGS - future give user ability to adjust?
            solver = jaxopt.LBFGS(fun=self.loss_function, maxiter=maxiter)

            # Run solver
            res = solver.run(
                init_params=init_params,
                tau=self.drt.tau,
                measurement=self.measurement,
                lbd=init_lbd,
            )

        if self.integration_method == "rbf":
            # define solver for new loss function
            solver = jaxopt.LBFGS(fun=self.loss_function_rbf, maxiter=maxiter)

            # Run solver
            res = solver.run(
                init_params=init_params,
                A_matrices=A_matrices,
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
        tau: jnp.ndarray,
        measurement: ImpedenceMeasurement,
        lbd: float,
    ) -> float:
        """Loss Function minimized in regularisation process. Default is Tikhonov Regularisation

        Args:
            init_params (jnp.ndarray): Initial parameters used when fitting
            tau (jnp.ndarray): Time constants used in integration method
            measurement (ImpedenceMeasurement):  Measurement Object containing experimental data
            lbd (float): regularisation parameter

        Returns:
            float: Return residuals of loss function calculated.
        """

        # Extract parameters
        R_inf = init_params[0]
        L_0 = init_params[1]

        # Here we take the absolute value to ensure non negativity
        gamma = jnp.abs(init_params[2:])

        # Create DRT object with parameters
        drt = DRT(R_inf, L_0, gamma, tau)

        # Create simulation object with parameters
        simulation = Simulation(drt, measurement.f)

        # Obtain impedance values
        Z_re, Z_im = simulation.run()

        # calculate residuals
        residuals = jnp.sum(
            (Z_re - measurement.Z_re) ** 2 + (Z_im - measurement.Z_im) ** 2
        ) + (lbd**2) * jnp.sum(drt.gamma**2)

        return residuals

    def loss_function_rbf(
        self,
        init_params: jnp.ndarray,
        A_matrices: jnp.ndarray,
        lbd: float,
    ) -> float:
        """Loss Function minimized in regularisation process. Default is Tikhonov Regularisation. RBF discretisation is used when calculating the impedence.

        Args:
            init_params (jnp.ndarray): Initial parameters which are optimised throughout the regression
            A_matrices (jnp.ndarray): Matrices calculated for the RBF function
            lbd (float): regularisation parameter

        Returns:
            float: Return residuals of loss function calculated.
        """
        # Extract parameters
        R_inf = init_params[0]
        L_0 = init_params[1]

        # Here we take the absolute value to ensure non negativity
        gamma = jnp.abs(init_params[2:])

        # Calculate the real and imaginary part of the impedence
        Z_re = R_inf + A_matrices[0] @ gamma
        Z_im = 2 * jnp.pi * self.measurement.f * L_0 + A_matrices[1] @ gamma

        # calculate residuals
        residuals = jnp.sum(
            (Z_re - self.measurement.Z_re) ** 2 + (Z_im - self.measurement.Z_im) ** 2
        ) + (lbd**2) * jnp.sum(gamma**2)

        return residuals
