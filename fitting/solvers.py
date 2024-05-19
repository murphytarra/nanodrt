"Module containing the solvers used for the optimisation process "

import equinox as eqx
import jax.numpy as jnp
import jaxopt
import dataclasses

# Ask about typechecking again? Look up...
from drt_solver.device import DRT, Measurement, FittedSpectrum
from drt_solver.simulaton import Simulation
from drt_solver.solvers import RBFSolver


class Regression(eqx.Module):
    """
    Class to perform Regression to determine DRT Spectrum
    """

    # Measurement object containing experimental data
    measurement: Measurement

    # DRT Object containing spectrum and time constants
    drt: DRT

    # Dictionary containing hyperparameters for solver
    solver_dict: dict[str, float]

    integration_method: str = dataclasses.field(default="trapezoid")  # type: ignore

    A_matrices: jnp.ndarray = dataclasses.field(default="None")  # type: ignore

    @eqx.filter_jit()
    def __call__(
        self,
    ):
        """
        Calls Regression Operation

        Returns:
            Simulation: Simulation Object containing the optimized parameters
        """

        # Extract regularisation parameter
        lbd = self.solver_dict["lambda"]

        # Extract number of steps
        maxiter = self.solver_dict["maxiter"]

        # build simulator initial here

        if self.integration_method == "trapezoid":
            # calculate A matrices by building simulator

            # How to switch the loss function?

            # Perform solver LBFGS - future give user ability to adjust?
            solver = jaxopt.LBFGS(fun=self.loss_function, maxiter=maxiter)

        if self.integration_method == "rbf":
            # Calculate A matrices and save as self.A_matrix
            integrals = RBFSolver(
                drt=self.drt, f_vec=self.measurement.f, log_t_vec=jnp.log(self.drt.tau)
            )
            integ = integrals()
            self.A_matrices = integ.A_matrix()

            # define solver for new loss function
            solver = jaxopt.LBFGS(fun=self.loss_function_rbf, maxiter=maxiter)

        # Initial Parameters to be fitted in optimization process
        init_params = jnp.hstack((self.drt.R_inf, self.drt.L_0, self.drt.gamma))

        # Run solver
        res = solver.run(
            init_params=init_params,
            tau=self.drt.tau,
            measurement=self.measurement,
            lbd=lbd,
        )

        # Extract parameters
        params, state = res

        # Create fitted spectrum object
        fit = FittedSpectrum(params, state, self.drt.tau)

        return fit

    @eqx.filter_jit()
    def loss_function(
        self,
        init_params: jnp.ndarray,
        tau: jnp.ndarray,
        measurement: Measurement,
        lbd: float,
    ) -> float:
        """Loss Function minimized in regularisation process. Default is Tikhonov Regularisation

        Args:
            params (jnp.ndarray): Initial parameters used when fitting
            measurement (Measurement): Measurement Object containing experimental data

        Returns:
            float: Value of loss function
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
