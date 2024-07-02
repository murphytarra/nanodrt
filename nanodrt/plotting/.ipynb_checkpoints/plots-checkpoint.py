"Module that contains objects to plot the Impedance and the DRT following fitting"

import equinox as eqx
import jax.numpy as jnp
from nanodrt.fitting.fits import FittedSpectrum
from nanodrt.drt_solver.measurements import ImpedenceMeasurement
import dataclasses
import seaborn as sns
from jax import vmap
import matplotlib.pyplot as plt
from typing import List, Union
import numpy as np


class Plot(eqx.Module):

    final_sim: FittedSpectrum

    measurement: ImpedenceMeasurement

    def __init__(self, final_sim, measurement) -> None:

        self.final_sim = final_sim
        self.measurement = measurement

    def gaussian(self, log_tau_m: float, log_tau_vec: jnp.array, mu: float) -> float:
        """
        Guassian Kernal used in RBF discretisation

        Args:
            log_tau_m (jnp.ndarray): time constant for RBF to be evaluated at
            mu (float): constant used for guassian filter - determines FWHM

        Returns:
            float: RBF kernal value
        """
        return jnp.exp(-((mu * (log_tau_m - log_tau_vec)) ** 2))

    def show_gamma(
        self,
    ) -> None:

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set(style="whitegrid", font="Times New Roman", font_scale=1.5)

        if self.final_sim.integration_method == "rbf":
            phi = vmap(self.gaussian, in_axes=(0, None, None))(
                self.final_sim.log_t_vec, self.final_sim.log_t_vec, 5.0
            )  # size (n, )
            print("dimensions")
            print(phi.shape)
            print(self.final_sim.gamma.shape)

            # Plotting the data
            ax.plot(
                self.final_sim.log_t_vec,
                (self.final_sim.gamma * phi).sum(axis=1),
                "-o",
                color="blue",
                linewidth=2,
            )

        elif self.final_sim.integration_method == "trapezoid":

            # Plotting the data
            ax.plot(
                self.final_sim.log_t_vec,
                self.final_sim.gamma,
                "-o",
                color="blue",
                linewidth=2,
            )

        # Set labels
        ax.set_xlabel(r"$\log(\tau)$", fontsize=16)
        ax.set_ylabel(r"$\gamma(\log(\tau))$", fontsize=16)

        # Set the title
        ax.set_title("DRT Spectrum", fontsize=20)

        # Remove top and right spines for a cleaner look
        sns.despine()

        # Adjust the figure layout for better spacing
        fig.tight_layout()

        # Display the plot
        plt.show()

    def show_Z(self) -> None:

        # Set the Seaborn style to "whitegrid" for a clean look
        sns.set(style="whitegrid", font="Times New Roman", font_scale=1.5)

        # Create a new figure with specific size
        fig, ax = plt.subplots(figsize=(8, 6))

        Z_fit_re, Z_fit_im = self.final_sim.simulate()

        # Plotting the simulated data
        ax.plot(Z_fit_re, -Z_fit_im, label="Fitted Data", color="blue", linewidth=2)

        # Plotting the exact data with markers
        ax.plot(
            self.measurement.Z_re,
            -self.measurement.Z_im,
            "o",
            label="Exact Data",
            color="red",
            markersize=6,
        )

        # Set labels
        ax.set_xlabel(r"$Z_{re}$ ($\Omega$)", fontsize=16)
        ax.set_ylabel(r"$-Z_{im}$ ($\Omega$)", fontsize=16)

        # Set the title
        ax.set_title("Fitted and Extracted Impedance Values", fontsize=20)

        # Add a legend
        ax.legend(fontsize=16, frameon=False, loc="best")

        # Remove top and right spines for a cleaner look
        sns.despine()

        # Adjust the figure layout for better spacing
        fig.tight_layout()

    def show(self) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        sns.set(style="whitegrid", font="Times New Roman", font_scale=1.5)

        # Plot gamma on the first subplot
        if self.final_sim.integration_method == "rbf":
            phi = vmap(self.gaussian, in_axes=(0, None, None))(
                self.final_sim.log_t_vec, self.final_sim.log_t_vec, 5.0
            )  # size (n, )

            # Plotting the data
            axs[0].plot(
                jnp.exp(self.final_sim.log_t_vec),
                (self.final_sim.gamma * phi).sum(axis=1),
                "-o",
                color="blue",
                linewidth=2,
            )

        elif self.final_sim.integration_method == "trapezoid":
            # Plotting the data
            axs[0].plot(
                jnp.exp(self.final_sim.log_t_vec),
                self.final_sim.gamma,
                "-o",
                color="blue",
                linewidth=2,
            )

        # Set labels and title for the first subplot
        axs[0].set_xlabel(r"$\log(\tau)$", fontsize=16)
        axs[0].set_ylabel(r"$\gamma(\log(\tau))$", fontsize=16)
        axs[0].set_title("DRT Spectrum", fontsize=20)

        # Set x-axis to logarithmic scale
        axs[0].set_xscale("log")

        # Plot DRT on the second subplot
        Z_fit_re, Z_fit_im = self.final_sim.simulate()

        # Plotting the simulated data
        axs[1].plot(Z_fit_re, -Z_fit_im, label="Fitted Data", color="blue", linewidth=2)

        # Plotting the exact data with markers
        axs[1].plot(
            self.measurement.Z_re,
            -self.measurement.Z_im,
            "o",
            label="Exact Data",
            color="red",
            markersize=6,
        )

        # Set labels and title for the second subplot
        axs[1].set_xlabel(r"$Z_{re}$ ($\Omega$)", fontsize=16)
        axs[1].set_ylabel(r"$-Z_{im}$ ($\Omega$)", fontsize=16)
        axs[1].set_title("Fitted and Extracted Impedance Values", fontsize=20)

        # Add a legend to the second subplot
        axs[1].legend(fontsize=16, frameon=False, loc="best")

        # Remove top and right spines for a cleaner look
        sns.despine()

        # Adjust the figure layout for better spacing
        fig.tight_layout()

        # Save the plot with high resolution
        plt.savefig("nature_journal_combined_plot.png", dpi=300)

        # Display the plot
        plt.show()
        
def plot_fit_comparison(
    f: Union[np.ndarray, jnp.ndarray], 
    Z_measured_re: Union[np.ndarray, jnp.ndarray], 
    Z_measured_im: Union[np.ndarray, jnp.ndarray], 
    tau_list: Union[Union[np.ndarray, jnp.ndarray], List[Union[np.ndarray, jnp.ndarray]]], 
    gamma_list: Union[Union[np.ndarray, jnp.ndarray], List[Union[np.ndarray, jnp.ndarray]]], 
    labels: Union[str, List[str]] = None, 
    Z_fit_re: Union[Union[np.ndarray, jnp.ndarray], List[Union[np.ndarray, jnp.ndarray]]] = None, 
    Z_fit_im: Union[Union[np.ndarray, jnp.ndarray], List[Union[np.ndarray, jnp.ndarray]]] = None
) -> None:
    """
    Plot DRT Spectrum and Nyquist plot.

    Args:
        f (Union[np.ndarray, jnp.ndarray]): Range of frequencies used to determine integrand values.
        Z_measured_re (Union[np.ndarray, jnp.ndarray]): Real part of measured impedance.
        Z_measured_im (Union[np.ndarray, jnp.ndarray]): Imaginary part of measured impedance.
        tau_list (Union[Union[np.ndarray, jnp.ndarray], List[Union[np.ndarray, jnp.ndarray]]]): List of tau arrays for multiple data sets or a single tau array.
        gamma_list (Union[Union[np.ndarray, jnp.ndarray], List[Union[np.ndarray, jnp.ndarray]]]): List of gamma arrays for multiple data sets or a single gamma array.
        labels (Union[str, List[str]], optional): List of labels for each tau and gamma set or a single label. Defaults to None.
        Z_fit_re (Union[Union[np.ndarray, jnp.ndarray], List[Union[np.ndarray, jnp.ndarray]]], optional): List of real parts of fitted impedance or a single array. Defaults to None.
        Z_fit_im (Union[Union[np.ndarray, jnp.ndarray], List[Union[np.ndarray, jnp.ndarray]]], optional): List of imaginary parts of fitted impedance or a single array. Defaults to None.

    Returns:
        None
    """

    # Ensure tau_list and gamma_list are lists of arrays
    if isinstance(tau_list, (np.ndarray, jnp.ndarray)) and tau_list.ndim == 1:
        tau_list = [tau_list]
    if isinstance(gamma_list, (np.ndarray, jnp.ndarray)) and gamma_list.ndim == 1:
        gamma_list = [gamma_list]

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(tau_list))]
    elif isinstance(labels, str):
        labels = [labels]

    if Z_fit_re is None:
        Z_fit_re = []
    elif isinstance(Z_fit_re, (np.ndarray, jnp.ndarray)) and Z_fit_re.ndim == 1:
        Z_fit_re = [Z_fit_re]
    if Z_fit_im is None:
        Z_fit_im = []
    elif isinstance(Z_fit_im, (np.ndarray, jnp.ndarray)) and Z_fit_im.ndim == 1:
        Z_fit_im = [Z_fit_im]

    # Set the Seaborn style to "whitegrid" for a clean look
    sns.set(style="whitegrid", font="Times New Roman", font_scale=1.5)
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Iterate over the tau and gamma lists to plot multiple data sets
    for tau, gamma, label in zip(tau_list, gamma_list, labels):
        axs[0].plot(
            tau,
            gamma,
            "-o",
            linewidth=2,
            label=label
        )
    
    # Set labels and title for the first subplot
    axs[0].set_xlabel(r"$\tau \ (s)$", fontsize=16)
    axs[0].set_ylabel(r"$\gamma(\ln\tau) \ (\Omega)$", fontsize=16)
    axs[0].set_title("DRT Spectrum", fontsize=20)
    axs[0].legend(fontsize=16, frameon=False, loc="best")  # Add legend to the first subplot

    # Set x-axis to logarithmic scale
    axs[0].set_xscale("log")

    # Plotting the simulated data
    for Z_re, Z_im, label in zip(Z_fit_re, Z_fit_im, labels):
        axs[1].plot(Z_re, -Z_im, label=label, linewidth=2, marker="x")

    # Plotting the exact data with markers
    axs[1].plot(
        Z_measured_re,
        -Z_measured_im,
        "o",
        label="Exact Data",
        color="red",
        markersize=6,
    )

    # Set labels and title for the second subplot
    axs[1].set_aspect('equal', adjustable='datalim')  # Ensure aspect ratio is equal
    axs[1].set_xlabel(r"$Z_{re}$ ($\Omega$)", fontsize=16)
    axs[1].set_ylabel(r"$-Z_{im}$ ($\Omega$)", fontsize=16)
    axs[1].set_title("Fitted and Extracted Impedance Values", fontsize=20)

    # Add a legend to the second subplot
    axs[1].legend(fontsize=16, frameon=False, loc="best")

    # Remove top and right spines for a cleaner look
    sns.despine()

    # Adjust the figure layout for better spacing
    fig.tight_layout()

    # Save the plot with high resolution
    # plt.savefig("nature_journal_combined_plot.png", dpi=300)

    # Display the plot
    plt.show()
