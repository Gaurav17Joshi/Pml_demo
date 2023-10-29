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

# Title of Streamlit app
st.title("ADVI vs. HMC")


from functions import run_hmc, run_advi, run_hmc2, run_advi2


st.set_option('deprecation.showPyplotGlobalUse', False)

"""
## Bayesian Inference Comparison of ADVI and HMC
"""

"""
Hamiltonian Monte Carlo (HMC) is a MCMC method that uses Hamiltonian Dynamics to efficiently sample from
high dimensional and complicated Distributions.

To understand more, one can refer to it original [paper](https://arxiv.org/pdf/1206.1901.pdf), or 
checkout this [Blog Post](https://bjlkeng.io/posts/hamiltonian-monte-carlo/#id4)

Automatic Differentiation Variational Inference (ADVI) is a fast approximate Bayesian inference method that employs variational 
inference techniques to estimate posterior distributions by optimizing a surrogate probability distribution. 
It's computationally efficient for large datasets and complex models.

To understand more, one can refer to it original [paper](https://arxiv.org/pdf/1603.00788.pdf).

In this demo, we will try and use ADVI and HMC on various datasets and models and 
compare their performances. We will be the PyMC3 library, with a tensorflow backend for this.

"""

model_choice = st.radio("DATA Models:", ["Beta Bernoulli Model", "Linear Regression Model"])

button1 = st.button("Generate Data")

if button1:
    if model_choice == "Beta Bernoulli Model":

        """
        THE MODEL 1

        In this setup, we have taken a coin flipping experiment, where we flip a coin 8 times and record the number of heads.

        This is Beta-Bernoulli distribution, where the prior is Beta(2, 2) and the likelihood is Bernoulli.

        Here, we have plotted the prior, likelihood and the true posterior.
        """

        key = jax.random.PRNGKey(127)
        dataset = np.zeros(8)
        n_samples = len(dataset)
        print(f"Dataset: {dataset}")
        n_heads = dataset.sum()
        n_tails = n_samples - n_heads

        def prior_dist():
            return dist.Beta(concentration1=2.0, concentration0=2.0)

        def likelihood_dist(theta):
            return dist.Bernoulli(probs=theta)
        
        # closed form of beta posterior
        a = prior_dist().concentration1
        b = prior_dist().concentration0

        exact_posterior = dist.Beta(concentration1=a + n_heads, concentration0=b + n_tails)

        theta_range = jnp.linspace(0.01, 0.99, 100)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(theta_range, exact_posterior.prob(theta_range), "g--", label="True Posterior")
        ax.plot(theta_range, prior_dist().prob(theta_range), label="Prior")

        likelihood = jax.vmap(lambda x: jnp.prod(likelihood_dist(x).prob(dataset)))(theta_range)
        likelihood = likelihood / (likelihood.sum()*(theta_range[1]-theta_range[0]))
        ax.plot(theta_range, likelihood, "r-.", label="Likelihood")

        ax.set_title("Prior, Likelihood and True Posterior")
        ax.set_xlabel("theta")
        ax.set_ylabel("Likelihood")
        ax.set_ylabel("Probability Densities")
        ax.legend()
        st.pyplot(fig)
    
    elif model_choice == "Linear Regression Model":

        """
        THE MODEL 2

        In this setup, we have taken a linear regression model in 2 dimensions

        Y = alpha + beta1 * X1 + beta2 * X2 + epsilon
            
        where epsilon ~ N(0, sigma)
        
        The dataset consists of 100 datapoints from the above model, with alpha = 1, beta1 = 1, beta2 = 2.5, sigma = 1.
        """

        RANDOM_SEED = 8927
        rng = np.random.default_rng(RANDOM_SEED)

        alpha, sigma = 1, 1
        beta = [1, 2.5]

        # Size of dataset
        size = 100

        # Predictor variable
        X1 = np.random.randn(size)
        X2 = np.random.randn(size) * 0.2

        # Simulate outcome variable
        Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
        axes[0].scatter(X1, Y, alpha=0.6)
        axes[1].scatter(X2, Y, alpha=0.6)
        axes[0].set_ylabel("Y")
        axes[0].set_xlabel("X1")
        axes[1].set_xlabel("X2")
        st.pyplot(fig)


button = st.button("Run Inference")

if button:
    if model_choice == "Beta Bernoulli Model":
        # Example data generation for Model 1

        """
        THE MODEL 1

        In this setup, we have taken a coin flipping experiment, where we flip a coin 8 times and record the number of heads.

        This is Beta-Bernoulli distribution, where the prior is Beta(2, 2) and the likelihood is Bernoulli.

        Here, we have plotted the prior, likelihood and the true posterior.
        """

        key = jax.random.PRNGKey(127)
        dataset = np.zeros(8)
        n_samples = len(dataset)
        print(f"Dataset: {dataset}")
        n_heads = dataset.sum()
        n_tails = n_samples - n_heads

        def prior_dist():
            return dist.Beta(concentration1=2.0, concentration0=2.0)

        def likelihood_dist(theta):
            return dist.Bernoulli(probs=theta)
        
        # closed form of beta posterior
        a = prior_dist().concentration1
        b = prior_dist().concentration0

        exact_posterior = dist.Beta(concentration1=a + n_heads, concentration0=b + n_tails)
        theta_range = jnp.linspace(0.01, 0.99, 100)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(theta_range, exact_posterior.prob(theta_range), "g--", label="True Posterior")
        ax.plot(theta_range, prior_dist().prob(theta_range), label="Prior")

        likelihood = jax.vmap(lambda x: jnp.prod(likelihood_dist(x).prob(dataset)))(theta_range)
        likelihood = likelihood / (likelihood.sum()*(theta_range[1]-theta_range[0]))
        ax.plot(theta_range, likelihood, "r-.", label="Likelihood")

        ax.set_title("Prior, Likelihood and True Posterior")
        ax.set_xlabel("theta")
        ax.set_ylabel("Likelihood")
        ax.set_ylabel("Probability Densities")
        ax.legend()
        st.pyplot(fig)

        """
        -----
        HMC
        """

        trace, time_hmc = run_hmc(dataset)
        st.write(f"Time taken to do HMC sampling: {time_hmc:.3f} seconds")
        thetas = jnp.array(trace.posterior["theta"])

        """
        We run 4 chains of HMC with 2000 samples, and 1000 dropoff samples.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("HMC posterior sample distribution for 4 chains")
        ax.plot(theta_range, exact_posterior.prob(theta_range), "b--", label="$p(x)$: True Posterior")
        for i in range(1, 5):
            sns.kdeplot(thetas[i-1, :], label=f"HMC: Chain {i}", clip=(0.0, 1.0), alpha = 0.5)
        ax.set_xlabel("theta")
        ax.legend()
        st.pyplot(fig)

        """
        We can visulaise the trace of the HMC samples for each chain used.
        """

        fig, ax = plt.subplots(4,1, figsize=(10, 8))
        fig.suptitle("HMC posterior sample distribution for 4 chains", fontsize=16)
        for i in range(4):
            ax[i].plot(thetas[i-1, :], label = f"Chain {i+1}", alpha = 0.7)
            ax[i].set_xlabel("Samples")
            ax[i].set_ylabel("theta")
            ax[i].legend()
        st.pyplot(fig)

        """
        -----
        ADVI
        """

        trace_approx, advi, time_advi, tracker = run_advi(dataset)
        st.write(f"Time taken to fit and sample from ADVI: {time_advi:.3f} seconds")

        """
        We can see the -ELBO decreasing as the ADVI optimization progresses.
        """

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(advi.hist, label="-ELBO")
        ax.set_title("Decrease in -ELBO during optimization")

        ax.hlines(y=advi.hist[-1], xmin=0, xmax=len(advi.hist), color = "r", linestyles= "--", 
                label=f"-ELBO = {advi.hist[-1]:.3f}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("-ELBO")
        ax.legend()
        st.pyplot(fig)

        """
        ADVI posterior distribution
        """

        thetas_advi = jnp.array(trace_approx.posterior["theta"]).flatten()

        fig, ax = plt.subplots()
        ax.plot(theta_range, exact_posterior.prob(theta_range), "b--", label="$p(x)$: True Posterior")
        sns.kdeplot(thetas_advi, label="ADVI posterior", clip=(0.0, 1.0)) #, bw_adjust=1)
        ax.set_xlabel("theta")
        ax.legend()
        st.pyplot(fig)

        """
        Checking the convergence of the ADVI posterior parameters, mean and sigma
        """
        fig, ax = plt.subplots(2,1)
        ax[0].set_title("ADVI posterior parameters")
        ax[0].plot(tracker["mean"], label="$\mu$, of variational posterior")
        ax[0].legend()
        ax[1].plot(tracker["std"], label="$\sigma$ of variational posterior")
        ax[1].set_xlabel("Iterations")
        ax[1].legend()
        st.pyplot(fig)

        """
        -----

        Comparing the ADVI and HMC posterior distributions
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("HMC vs ADVI posterior samples")
        ax.plot(theta_range, exact_posterior.prob(theta_range), "b--", label="$p(x)$: True Posterior")
        sns.kdeplot(thetas_advi, label="ADVI posterior", clip=(0.0, 1.0))
        for i in range(1, 5):
            sns.kdeplot(thetas[i-1, :], label=f"Chain {i}", clip=(0.0, 1.0), alpha = 0.5)
        ax.set_xlabel("theta")
        ax.legend()
        st.pyplot(fig)


    elif model_choice == "Linear Regression Model":
        """
        THE MODEL 2

        In this setup, we have taken a linear regression model in 2 dimensions

        Y = alpha + beta1 * X1 + beta2 * X2 + epsilon
            
        where epsilon ~ N(0, sigma)
        
        The dataset consists of 100 datapoints from the above model, with alpha = 1, beta1 = 1, beta2 = 2.5, sigma = 1.
        """

        RANDOM_SEED = 8927
        rng = np.random.default_rng(RANDOM_SEED)

        alpha, sigma = 1, 1
        beta = [1, 2.5]

        # Size of dataset
        size = 100

        # Predictor variable
        X1 = np.random.randn(size)
        X2 = np.random.randn(size) * 0.2

        # Simulate outcome variable
        Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
        axes[0].scatter(X1, Y, alpha=0.6)
        axes[1].scatter(X2, Y, alpha=0.6)
        axes[0].set_ylabel("Y")
        axes[0].set_xlabel("X1")
        axes[1].set_xlabel("X2")
        st.pyplot(fig)

        """
        ---
        HMC

        Here we will first use Hamiltonian Monte carlo to sample from the posterior parameters of the model given
        the data. We will using 4 chains of HMC with 2000 samples, and 1000 dropoff samples.

        We will plot the posterior distributions of the parameters alpha, beta1, beta2 and sigma, for all the 4 chains.
        """

        trace, time_hmc = run_hmc2(X1, X2, Y)
        st.write(f"Time taken to do HMC sampling: {time_hmc:.3f} seconds")

        alphas = jnp.array(trace.posterior["alpha"])
        betas = jnp.array(trace.posterior["beta"])
        sigmas = jnp.array(trace.posterior["sigma"])

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("HMC posterior sample distribution of alpha")
        for i in range(1, 5):
            sns.kdeplot(alphas[i-1, :], label=f"alpha  Chain {i}", alpha = 0.5) #, bw_adjust=1)
        plt.xlabel("alpha")
        plt.legend()

        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("HMC posterior sample distribution of beta$")
        for i in range(1, 5):
            sns.kdeplot(betas[i-1, :, 0], label=f"beta_1: Chain {i}", alpha = 0.5) #, bw_adjust=1)

        ax.vlines(x=np.mean(betas[:,:,0]), ymin=0, ymax=4, color="red", linestyle="--", label = f"Mean1: {np.mean(betas[:,:,0]):.3f}")
        for i in range(1, 5):
            sns.kdeplot(betas[i-1, :, 1], label=f"beta_2: Chain {i}", alpha = 0.5) #, bw_adjust=1)
        ax.vlines(x=np.mean(betas[:,:,1]), ymin=0, ymax=1, color="red", linestyle="--", label = f"Mean2: {np.mean(betas[:,:,1]):.3f}")
        ax.set_xlabel("beta")
        ax.legend()

        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("HMC posterior sample distribution of sigma")
        for i in range(1, 5):
            sns.kdeplot(sigmas[i-1, :], label=f"sigma Chain {i}", alpha = 0.5) #, bw_adjust=1)
        ax.set_xlabel("sigma")
        ax.legend()

        st.pyplot(fig)

        """
        --- 
        ADVI

        Now, we will try and use ADVI to approximate the posterior distribution of the parameters.
        The, we will sample parameters from the posterior.

        """                 

        trace_approx, advi, time_advi, tracker = run_advi2(X1, X2, Y)

        st.write(f"Time taken to execute the ADVI fitting and sampling: {time_advi:.3f} seconds")

        st.write("We can see how the -ve ELBO decreases to a minimum")

        alphas2 = jnp.array(trace_approx.posterior["alpha"]).flatten()
        betas2 = jnp.array(trace_approx.posterior["beta"])
        sigmas2 = jnp.array(trace_approx.posterior["sigma"]).flatten()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(advi.hist, label="-ELBO")
        ax.set_title("Decrease in -ELBO during optimization")

        ax.hlines(y=advi.hist[-1], xmin=0, xmax=len(advi.hist), color = "r", linestyles= "--", 
                label=f"-ELBO = {advi.hist[-1]:.3f}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("-ELBO")
        ax.legend()
        
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("ADVI posterior samples of alpha")
        sns.kdeplot(alphas2, label="ADVI posterior") #, bw_adjust=1)
        ax.vlines(x=np.mean(alphas2), ymin=0, ymax=3.5, color="red", linestyle="--",
                     label = f"Mean: {np.mean(alphas2):.3f}")
        ax.set_xlabel("alpha")
        ax.legend()

        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("ADVI posterior samples of beta")
        sns.kdeplot(betas2[0,:, 0], label="beta_1 ADVI posterior")
        ax.vlines(x=np.mean(betas2[0,:,0]), ymin=0, ymax=4, color="red", linestyle="--", 
                   label = f"Mean1: {np.mean(betas2[0,:,0]):.3f}")
        sns.kdeplot(betas2[0,:, 1], label="beta_2 ADVI posterior")
        ax.vlines(x=np.mean(betas2[0,:,1]), ymin=0, ymax=1, color="red", linestyle="--", 
                   label = f"Mean2: {np.mean(betas2[0,:,1]):.3f}")
        ax.set_xlabel("Beta")
        ax.legend()

        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("ADVI posterior samples of sigma")
        sns.kdeplot(sigmas2, label="ADVI posterior") #, bw_adjust=1)
        ax.set_xlabel("Sigma")
        ax.legend()

        st.pyplot(fig)


        """
        -----
        Comparing the ADVI and HMC posterior distributions
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("HMC vs ADVI posterior samples")
        sns.kdeplot(alphas2, label="alpha ADVI posterior")
        for i in range(1, 5):
            sns.kdeplot(alphas[i-1, :], label=f"alpha HMC Chain {i}", alpha = 0.5, linestyle="--")
            ax.set_xlabel("alpha")
            ax.legend()
        
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("HMC vs ADVI posterior samples")
        sns.kdeplot(betas2[0,:, 0], label="beta_1 ADVI posterior")
        for i in range(1, 5):
            sns.kdeplot(betas[i-1, :, 0], label=f"beta_1 HMC Chain {i}", alpha = 0.5, linestyle="--")
            ax.set_xlabel("beta1")
            ax.legend()
        sns.kdeplot(betas2[0,:, 1], label="beta_2 ADVI posterior")
        for i in range(1, 5):
            sns.kdeplot(betas[i-1, :, 1], label=f"beta_2 HMC Chain {i}", alpha = 0.5, linestyle="--")
            ax.set_xlabel("Beta")
            ax.legend()
        ax.legend()

        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("HMC vs ADVI posterior samples")
        sns.kdeplot(sigmas2, label="sigma ADVI posterior")
        for i in range(1, 5):
            sns.kdeplot(sigmas[i-1, :], label=f"sigma HMC Chain {i}", alpha = 0.5, linestyle="--")
            ax.set_xlabel("sigma")
            ax.legend()
        ax.legend()

        st.pyplot(fig)



    # Display results
    st.subheader("Results")

    st.write(f" - ADVI took {time_advi:.3f} seconds to fit and sample from the posterior, while HMC took {time_hmc:.3f} seconds.")

    """
    Hense we can see that ADVI is much faster than HMC. 
    
    ADVI has to fit its variational posterior, and then sample from it, while HMC samples directly from the posterior.

    These datasets are simple, so both the methods perform very well on them. 

    To do a more detailed analysis, we would have to use models which are in higher dimenstions, which have 
    multimodal posterior distributions. (ADVI has a simple variational posterior, so it cannot approximate, 
    mulimodal posteriors well.) 
    """


