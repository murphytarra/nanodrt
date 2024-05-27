"Module containing different methods of extracting the regularisation parameters "

import equinox as eqx
import jax.numpy as jnp
import jaxopt
import dataclasses

# Ask about typechecking again? Look up...
from nanodrt.drt_solver.measurements import ImpedenceMeasurement


class GCV(eqx.Module):
    """
    Object which computes the regularisation parameter by minimizing the GCV function
    """

    # Object containing Measurement information
    measurement: ImpedenceMeasurement

    # Real component of A matrices
    A_re: jnp.ndarray

    # Imaginary component of A matrices
    A_im: jnp.ndarray

    # Matrix in normalisation
    M: jnp.ndarray = dataclasses.field(default=None)  # type: ignore

    # Initial value of lbd used in the optimisation process
    init_lbd: float = dataclasses.field(default=None)  # type: ignore

    def __post_init__(self):
        self.validate_init()

    @eqx.filter_jit
    def __call__(self) -> float:

        solver = jaxopt.LBFGS(fun=self.compute_GCV, maxiter=50000)
        res = solver.run(
            init_params=jnp.array(self.init_lbd),
            A_re=self.A_re,
            A_im=self.A_im,
            measurement=self.measurement,
            M=jnp.eye(self.A_re.shape[1] + 2),
        )
        param, state = res
        return param

    @eqx.filter_jit
    def compute_GCV(self, init_params, A_re, A_im, measurement, M) -> float:
        """
        Compute the Generalized Cross Validation Metric for a given lambda value.

        Args:
            init_params (float): Value of lambda used
            A_re (jnp.ndarray): Real component of A matrix used in DRT calculation
            A_im (jnp.ndarray): Imaginary Component of A matrix used in DRT Calculation
            measurement (ImpedanceMeasurement): Object containing measurement information
            M (Matrix): Used in Regularisation

        Returns:
            float: Value of GCV
        """

        # Ensure that the lambda value is positive
        lambda_value = jnp.exp(init_params)

        # Create the vector of measured impedences
        Z = jnp.hstack((measurement.Z_re, measurement.Z_im))  # stacked impedance

        # Shape of impedance
        n_cv = Z.shape[0]  # n_cv = 2*N_freqs with N_freqs the number of EIS frequencies

        # Create arrays of zeros and ones
        n_cv2 = int(n_cv / 2)
        ones = jnp.ones(n_cv2)
        zeros = jnp.zeros(n_cv2)

        # Create total A matrix
        A_top = jnp.vstack((zeros, ones, A_re.T))
        A_bottom = jnp.vstack((2.0 * jnp.pi * measurement.f, zeros, A_im.T))
        A = jnp.hstack((A_top, A_bottom))

        # Determine inverse calculation used in GCV Calculation
        A_agm = A @ A.T + lambda_value * M  # see (13) in [4]
        L_agm = jnp.linalg.cholesky(A_agm)  # Cholesky transform to inverse A_agm
        inv_L_agm = jnp.linalg.inv(L_agm)
        inv_A_agm = inv_L_agm.T @ inv_L_agm  # inverse of A_agm
        A_GCV = A.T @ inv_A_agm @ A  # see (13) in [4]

        # GCV score
        GCV_num = (
            1 / n_cv * jnp.linalg.norm((jnp.eye(n_cv) - A_GCV) @ Z) ** 2
        )  # numerator
        GCV_dom = (1 / n_cv * jnp.trace(jnp.eye(n_cv) - A_GCV)) ** 2  # denominator

        GCV_score = GCV_num / GCV_dom
        return GCV_score

    def validate_init(self):
        """
        Validate the initialization parameters to ensure they meet the necessary criteria.
        """
        if self.A_re.ndim != 2:
            raise ValueError(
                f"A_re must be a 2-dimensional array, but got {self.A_re.ndim} dimensions"
            )
        if self.A_im.ndim != 2:
            raise ValueError(
                f"A_im must be a 2-dimensional array, but got {self.A_im.ndim} dimensions"
            )
        if self.M is not None and self.M.shape != (
            self.A_re.shape[1] + 2,
            self.A_re.shape[1] + 2,
        ):
            raise ValueError(
                f"M must be of shape {(self.A_re.shape[1] + 2, self.A_re.shape[1] + 2)}, but got {self.M.shape}"
            )

    def __repr__(self):
        return (
            f"GCV(measurement={self.measurement}, "
            f"A_re=Array of shape {self.A_re.shape}, "
            f"A_im=Array of shape {self.A_im.shape}, "
            f"M=Array of shape {None if self.M is None else self.M.shape}, "
            f"init_lbd={self.init_lbd})"
        )
