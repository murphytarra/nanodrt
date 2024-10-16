import jax.numpy as jnp


def ZARC(f, R_0, R_ct, tau_0, phi):
    """
    Calculate the impedance based on the given parameters.

    Args:
    f (float or array-like): Frequency of the input signal (in Hz).
    R_0 (float): Infinite frequency resistance (ohmic resistance).
    R_ct (float): Charge-transfer resistance.
    tau_0 (float): Characteristic time constant.
    phi (float): Dispersion coefficient, related to the roughness of the impedance response.

    Returns:
    complex or ndarray of complex: Computed impedance for each frequency.
    """

    # Calculate the complex part of the impedance
    omega_tau = 2 * jnp.pi * f * tau_0
    exponent = 1j * omega_tau
    Z = R_0 + R_ct / (1 + exponent**phi)

    return Z


def gamma_ZARC(tau, R_ct, tau_0, phi):
    """
    Calculate the gamma function, which is a component in certain electrochemical impedance spectroscopy models.

    Args:
    tau (float or array-like): Time constants where the function is evaluated.
    R_ct (float): Charge-transfer resistance, a parameter of the impedance model.
    tau_0 (float): Reference time constant for normalization.
    phi (float): Dispersion coefficient, which characterizes the distribution of relaxation times.

    Returns:
    ndarray or float: Computed value of the gamma function for each time constant in tau.
    """

    num = R_ct * jnp.sin((1 - phi) * jnp.pi)
    den = (
        2
        * jnp.pi
        * (jnp.cosh(phi * jnp.log(tau / tau_0)) - jnp.cos(jnp.pi * (1 - phi)))
    )

    return num / den
