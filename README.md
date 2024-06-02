# NanoDRT: Fitting Library for EIS DRT Spectroscopy

NanoDRT is a small python package written in JAX to determine the DRT of EIS Spectroscopy Data. 

## Installation 

To install NanoDRT, run the following code in your terminal. 

This will create a conda environment with the NanoDRT installed. To use NanoDRT, ensure you are using this environment. 

```
conda create -n nanodrt python=3.8 -y 
conda activate myenv 
git clone git@github.com:murphytarra/nanodrt.git
pip install -e nanodrt
```

For more information on conda environments see [here](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html). 

## Quick Example 

Let's first provide a quick example to how we can use NanoDRT to fit to EIS data. 

```
import jax.numpy as jnp 
import pandas as pd 

# How to make this all one import?
from nanodrt.drt_solver.drt import DRT 
from nanodrt.fitting.optimizer import Optimizer 
from nanodrt.drt_solver.measurements import ImpedenceMeasurement 
from nanodrt.drt_solver.utils import (ZARC, 
                                 gamma_ZARC) 

columns = ["f", "Z_re", "Z_im"]
df = pd.read_csv("data/single_ZARC.csv", header=None, names=columns)

f = df["f"]
tau = 1./(2*jnp.pi*f)

measurement = ImpedenceMeasurement(df["Z_re"], df["Z_im"], df["f"])

#Create our DRT guess
gamma_guess = gamma_ZARC(tau, R_ct=33, tau_0=.2, phi= .7)
drt = DRT(R_inf= 10, L_0=0, gamma=gamma_guess, tau=tau)

# Fit 
solver_dict = {"init_lbd": 0.05, "lbd_selection": "GCV", 'maxiter': 5e3}
optim = Optimizer(drt=drt, measurement=measurement, solver="regression", integration_method="rbf", solver_dict = solver_dict, rbf_function="guassian")
final_sim = optim.run()
Z_re_fitted, Z_im_fitted = final_sim.simulate()
```

## Introduction to DRT and EIS Data
Electrochemical Impedance Spectroscopy (EIS) is a powerful technique used to study the electrical properties of materials and systems, often in the context of batteries, fuel cells, and corrosion studies. The Distribution of Relaxation Times (DRT) provides a way to interpret EIS data by separating the contributions of different electrochemical processes based on their characteristic time constants.

### What is EIS?
EIS measures the impedance of a system over a range of frequencies, providing insights into various processes such as charge transfer, diffusion, and double-layer capacitance. The resulting data is often represented as a Nyquist plot (imaginary vs. real part of impedance) or a Bode plot (impedance magnitude and phase vs. frequency).

### What is DRT?
DRT is a method to deconvolute the EIS data into contributions from different relaxation processes. Each process has a characteristic time constant, and DRT helps to identify these time constants and their respective strengths, providing a more detailed understanding of the underlying mechanisms.

## Formulas for DRT to Impedance

The relationship between the DRT spectrum and the impedance can be described using the following formulas:

### Impedance from DRT

The impedance $ Z(f)$ as a function of frequency $ f $ can be expressed using the Distribution of Relaxation Times (DRT) $ g(\log \tau) $ as follows:

$$Z(f) = i2\pi f L_0 + R_{\infty} + \int_{-\infty}^{\infty} \frac{g(\log \tau)}{1 + i2\pi f \tau} \, d\log \tau $$

Where:
- \( L_0 \) is the inductance.
- \( R_{\infty} \) is the high-frequency resistance.
- \( \tau \) is the relaxation time.

### Real and Imaginary Parts of Impedance

The real part $ Z' $ and imaginary part $Z'' $ of the impedance are given by:

$$Z'(f) = R_{\infty} + \int_{-\infty}^{\infty} \frac{g(\log \tau)}{1 + (2\pi f \tau)^2} \, d\log \tau $$

$$Z''(f) = 2\pi f L_0 - \int_{-\infty}^{\infty} \frac{2\pi f \tau \cdot g(\log \tau)}{1 + (2\pi f \tau)^2} \, d\log \tau $$

These formulas allow the computation of the impedance spectrum from the DRT, providing insights into the contributions of different electrochemical processes.


### Why is DRT Useful?
By applying DRT to EIS data, researchers can:

- Distinguish between overlapping processes.
- Quantify the contribution of different mechanisms.
- Improve the interpretation and modeling of electrochemical systems.

## Methods of Integration Available

When fitting to EIS data - several numerical methods of integration are available to the user. 

The simpliest example is using `trapezoidal`, which uses the trapezoidal method to determine the integral for the DRT calculation. 

To use the trapezoidal method, one must simply input `integration_method=trapezoidal` into the optimizer, as shown below; 

```
optim = Optimizer(drt=drt, measurement=measurement, solver="regression", integration_method="trapezoidal", solver_dict = solver_dict)
``` 

However, as seen in CITE, different methods of calculating this integral have proven more accurate - one such being the 'Radial Basis Function Method'. For more information on this method of calculating integrals see HERE. The current radial basis functions available to the user are given in the table below. 

| RBF Function | Formula                             |
|--------------|-------------------------------------|
| Gaussian     | $\phi(r) = e^{-(\epsilon r)^2}$ |
| C2           | $ \phi(r) = (1 + \epsilon r) e^{-\epsilon r} $ |
| C4           | $ \phi(r) = (1 + \epsilon r + (\epsilon r)^2 / 2) e^{-\epsilon r} $ |
| C6           | $ \phi(r) = (1 + \epsilon r + (\epsilon r)^2 / 2 + (\epsilon r)^3 / 6) e^{-\epsilon r} $ |

Where:
- $ r $ is the radial distance.
- $ \epsilon $ is a shape parameter.

To use a radial basis function in the optimisation process, one need simply input in the function and method of integration, as given below; 


```
optim = Optimizer(drt=drt, measurement=measurement, solver="regression", integration_method="rbf", solver_dict = solver_dict, rbf_function="guassian")
``` 

The above example uses the `guassian` rbf function. However, one can also use INSERT. 


## Regularisation Terms 
As outlined in INSERT, regularisation is of huge importance when fitting to the desired DRT spectrum. To do so, we typically add a regularisation function to the loss function, given below as; 

ISNERT

However, determining the magnitude of the regularisation parameter can be difficult, and several methods of determining it's value has been outlined HERE. 

In NanoDRT, we allow the user to either input in a predetermined value for the regularisation parameter by inputting it in the `solver_dict` of the optimizer; 

`solver_dict = {"init_lbd": 0.05, 'maxiter': 5e3}`

However, one can also use the GCV, which determines the value of the regularisation parameter for you, given an initial guess;

`solver_dict = {"init_lbd": 0.05, "lbd_selection": "GCV", 'maxiter': 5e3}`

In both examples, `maxiter` is the maximum number of iterations the optimizer can do, before converging. 

Note: the value of the regularisation parameter chosen will greatly effect your overall fitting and must be chosen carefully. 

## Examples 

### Double Zarc Model 

### Ed's Weird Data