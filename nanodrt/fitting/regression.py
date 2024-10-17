"Module containing the Regression object which is used in the Optimization Process "
import equinox as eqx
import jax.numpy as jnp
import jaxopt
import dataclasses

# Ask about typechecking again? Look up...
from nanodrt.drt_solver.drt import DRT
from nanodrt.drt_solver.measurements import ImpedanceMeasurement
from nanodrt.fitting.fits import FittedSpectrum
from nanodrt.drt_solver.simulation import Simulation
from nanodrt.drt_solver.solvers import RBFSolver


class Regression(eqx.Module):
    """
    Class to perform Regression to determine DRT Spectrum
    """

    # Measurement object containing experimental data
    measurement: ImpedanceMeasurement

    # DRT Object containing spectrum and time constants
    drt: DRT

    # Dictionary containing hyperparameters for solver
    solver_dict: dict[str, float]

    # Method of Integration used in Regression
    integration_method: str = dataclasses.field(default="rbf")  # type: ignore

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

        integrals1 = RBFSolver(
            drt=self.drt,
            f_vec=self.measurement.f,
            log_t_vec=jnp.log(self.drt.tau),
            mu=self.solver_dict["mu"],
        )
        A_matrices = integrals1()

        init_lbd = self.solver_dict["init_lbd"]

        # Extract number of steps
        maxiter = self.solver_dict["maxiter"]

        # Initial Parameters to be fitted in optimization process
        init_params = jnp.hstack((self.drt.R_0, self.drt.L_0, self.drt.gamma)) # use gamma values passed as initial guess

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
            self.solver_dict["mu"],
        )

        return fit

    def loss_function_rbf(
        self,
        init_params: jnp.ndarray,
        A_matrices: jnp.ndarray,
        lbd: float,
    ) -> float:
        """Loss Function minimized in regularisation process. Default is Tikhonov Regularisation. RBF discretisation is used when calculating the Impedance.

        Args:
            init_params (jnp.ndarray): Initial parameters which are optimised throughout the regression
            A_matrices (jnp.ndarray): Matrices calculated for the RBF function
            lbd (float): regularisation parameter

        Returns:
            float: Return residuals of loss function calculated.
        """
        # Extract parameters
        # Here we take the absolute value to ensure non negativity
        R_0 = jnp.abs(init_params[0])
        L_0 = jnp.abs(init_params[1])
        x = jnp.abs(init_params[2:])

        # Calculate the real and imaginary part of the Impedance
        Z_re = R_0 + A_matrices[0] @ x
        Z_im = 2 * jnp.pi * self.measurement.f * L_0 + A_matrices[1] @ x

        # calculate residuals
        residuals = jnp.sum(
            (Z_re - self.measurement.Z_re) ** 2 + (Z_im - self.measurement.Z_im) ** 2
        ) + (lbd**2) * jnp.sum(x**2)

        return residuals
