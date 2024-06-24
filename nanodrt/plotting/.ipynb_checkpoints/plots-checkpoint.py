"Module that contains objects to plot the Impedance and the DRT following fitting"

import equinox as eqx
import jax.numpy as jnp
from nanodrt.fitting.fits import FittedSpectrum
from nanodrt.drt_solver.measurements import ImpedenceMeasurement
import dataclasses
import seaborn as sns
from jax import vmap
import matplotlib.pyplot as plt


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
