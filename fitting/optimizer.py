"Module containing the Optimizer Class to determine the DRT Spectrum "

import equinox as eqx
import dataclasses
import jax.numpy as jnp
from fitting.solvers import Regression

# Ask about typechecking again? Look up...
from drt_solver.device import DRT, Measurement


class Optimizer(eqx.Module):
    """
    Class used to determine the DRT Spectrum
    """

    # Measurment object containing experimental data
    measurement: Measurement

    # DRT containing intitial guess for parameters and spectrum to be fitted
    drt: DRT

    # solver used to fit the data
    solver: str = dataclasses.field(default="regression")  # type: ignore

    # Dictionary of solvers used for fitting
    solver_dict: dict[str:float] = dataclasses.field(default=None)  #  type: ignore

    def __init__(
        self,
        measurement: Measurement,
        drt: DRT,
        solver: str = "regression",
        solver_dict: dict[str, float] = None,
    ) -> None:
        """
        Class for setting up and running optimization process

        Args:
            measurement (Measurement): Measurement Object containing experimental data
            solver (str, optional): Method of solving used. Defaults to regression.
            params (jnp.ndarray, optional): Initial guess for parameters used. Defaults to None.
            solver_dict (dict): Dictionary of values used for optimisation process. Dictionary types depend on process used.

        """

        # Measurement object
        self.measurement = measurement

        # DRT object to be measured
        self.drt = drt

        # Type of solver used
        self.solver = solver.lower()

        # Hyperparameters for the solving method
        self.solver_dict = solver_dict

        self.__validate_init__()

    def __repr__(self) -> str:
        return (
            f"Optimizer(measurement={self.measurement}, drt={self.drt}, "
            f"solver={self.solver}, solver_dict={self.solver_dict})"
        )

    def __validate_init__(self) -> None:
        """Validate the initialization parameters."""
        if not isinstance(self.measurement, Measurement):
            raise TypeError(
                f"Expected measurement to be an instance of Measurement, got {type(self.measurement)}"
            )
        if not isinstance(self.drt, DRT):
            raise TypeError(
                f"Expected drt to be an instance of DRT, got {type(self.drt)}"
            )
        if not isinstance(self.solver, str):
            raise TypeError(f"Expected solver to be a string, got {type(self.solver)}")
        if self.solver not in ["regression"]:
            raise ValueError(f"Unsupported solver method: {self.solver}")
        if self.solver_dict is not None and not isinstance(self.solver_dict, dict):
            raise TypeError(
                f"Expected solver_dict to be a dictionary, got {type(self.solver_dict)}"
            )

    @eqx.filter_jit()
    def run(
        self,
    ):  # what does this return? simulate object?
        """
        Runs the simulation based on the type of solver selected

        Returns:
            Simulation: Simulation object with optimized values
        """

        if self.solver == "regression":
            regression = Regression(
                measurement=self.measurement,
                drt=self.drt,
                solver_dict=self.solver_dict,
            )
            fit = regression()
            return fit
