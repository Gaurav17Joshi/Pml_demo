import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import time
import numpy.random as npr

import jax
import jax.numpy as jnp
from jax import lax
from tensorflow_probability.substrates import jax as tfp
dist = tfp.distributions

import seaborn as sns
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings("ignore")

def run_hmc(dataset, nofchains, nofsamples):
    npr.seed(0)
    start_time_hmc = time.time()

    with pm.Model() as model:
        # Prior: Beta(2, 2)
        theta = pm.Beta("theta", alpha=2, beta=2)
        
        # Likelihood: Bernoulli likelihood
        y = pm.Bernoulli("y", p=theta, observed=dataset)
        # print(nofchains)
        # Using HMC to sample from the posterior
        # Here 2000 is the number of samples and 1000 is the number of tuning steps
        trace = pm.sample(nofsamples, tune=1000, cores=nofchains)

    end_time_hmc = time.time()

    return trace, end_time_hmc - start_time_hmc

def run_advi(dataset, nofiterations, nofsamples):
    start_time_advi = time.time()
    with pm.Model() as advi_model:
        # Prior: Beta(2, 2)
        theta = pm.Beta("theta", 2, 2)

        # Model likelihood: Bernoulli
        y = pm.Bernoulli("y", p=theta, observed=dataset)  # Bernoulli

        # Creating an ADVI instance
        advi = pm.ADVI()

        # Tracks the mean and std during training
        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,  
            std=advi.approx.std.eval,  
        )

        # Performing the ADVI optimization
        # Here approx is an instance of Approximation class, n is the number of iterations
        approx = advi.fit(callbacks=[tracker], n=nofiterations)

    # Draw samples from the approximated posterior
    trace_approx = approx.sample(nofsamples)

    end_time_advi = time.time()
    return trace_approx, advi, end_time_advi - start_time_advi, tracker

def run_hmc2(X1, X2, Y, nofchains, nofsamples):
    npr.seed(0)
    start_time_hmc = time.time()
    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Expected value of outcome
        mu = alpha + beta[0] * X1 + beta[1] * X2

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

        trace = pm.sample(nofsamples, tune=1000, cores=nofchains)

    end_time_hmc = time.time()

    return trace, end_time_hmc - start_time_hmc

def run_advi2(X1, X2, Y, nofiterations, nofsamples):
    start_time_advi = time.time()
    with pm.Model() as advi_model:
        # Priors for unknown model parameters
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Expected value of outcome
        mu = alpha + beta[0] * X1 + beta[1] * X2

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

        # trace = pm.sample(2000, tune=1000, cores=4)

        # Creating an ADVI instance
        advi = pm.ADVI()

        # Tracks the mean and std during training
        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,  
            std=advi.approx.std.eval,  
        )

        # Performing the ADVI optimization
        # Here approx is an instance of Approximation class, n is the number of iterations
        approx = advi.fit(callbacks=[tracker], n=nofiterations)

    # Draw samples from the approximated posterior
    trace_approx = approx.sample(nofsamples)

    end_time_advi = time.time()

    return trace_approx, advi, end_time_advi - start_time_advi, tracker