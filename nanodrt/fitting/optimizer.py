"Module containing the Optimizer Class to determine the DRT Spectrum "

import equinox as eqx
import dataclasses
import jax.numpy as jnp
from nanodrt.fitting.regression import Regression

# Ask about typechecking again? Look up...
from nanodrt.drt_solver.drt import DRT
from nanodrt.drt_solver.measurements import ImpedenceMeasurement


class Optimizer(eqx.Module):
    """
    Class used to determine the DRT Spectrum
    """

    # Measurment object containing experimental data
    measurement: ImpedenceMeasurement

    # DRT containing intitial guess for parameters and spectrum to be fitted
    drt: DRT

    # solver used to fit the data
    solver: str = dataclasses.field(default="regression")  # type: ignore

    # Integration Method used in the simulation
    integration_method: str = dataclasses.field(default="trapezoid")  # type: ignore

    # Dictionary of solvers used for fitting
    solver_dict: dict[str:float] = dataclasses.field(default=None)  #  type: ignore

    def __init__(
        self,
        measurement: ImpedenceMeasurement,
        drt: DRT,
        solver: str = "regression",
        integration_method: str = "trapezoid",
        solver_dict: dict[str, float] = None,
    ) -> None:
        """
        Class for setting up and running optimization process

        Args:
            measurement (Measurement): Measurement Object containing experimental data
            drt (DRT): DRT object used for the simulation process.
            solver (str, optional): Method of solving used. Defaults to regression.
            integration_method: Method used in simulation process to determine the impedence.
            solver_dict (dict): Dictionary of values used for optimisation process. Dictionary types depend on process used.

        """

        # Measurement object
        self.measurement = measurement

        # DRT object to be measured
        self.drt = drt

        # Type of solver used
        self.solver = solver.lower()

        # Integration Method used in simulation process
        self.integration_method = integration_method.lower()

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
        if not isinstance(self.measurement, ImpedenceMeasurement):
            raise TypeError(
                f"Expected measurement to be an instance of ImpedenceMeasurement, got {type(self.measurement)}"
            )
        if not isinstance(self.drt, DRT):
            raise TypeError(
                f"Expected drt to be an instance of DRT, got {type(self.drt)}"
            )
        if not isinstance(self.solver, str):
            raise TypeError(f"Expected solver to be a string, got {type(self.solver)}")
        if self.solver not in ["regression"]:
            raise ValueError(f"Unsupported solver method: {self.solver}")
        if not isinstance(self.integration_method, str):
            raise TypeError(
                f"Expected integration_method to be a string, got {type(self.integration_method)}"
            )
        if self.integration_method not in ["trapezoid", "rbf"]:
            raise ValueError(
                f"Unsupported integration method: {self.integration_method}"
            )
        if self.solver_dict is not None and not isinstance(self.solver_dict, dict):
            raise TypeError(
                f"Expected solver_dict to be a dictionary, got {type(self.solver_dict)}"
            )
        if self.solver_dict is not None:
            for key, value in self.solver_dict.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Keys in solver_dict must be strings, got {type(key)}"
                    )
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"Values in solver_dict must be int or float, got {type(value)}"
                    )

    @eqx.filter_jit()
    def run(
        self,
    ):
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
                integration_method=self.integration_method,
            )
            fit = regression()
            return fit
