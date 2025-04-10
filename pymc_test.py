import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import openturns as ot
import openturns.viewer as viewer
from sklearn.gaussian_process import GaussianProcessRegressor
from SALib.sample import saltelli
from SALib.analyze import sobol

# Set seed
np.random.seed(42)

# Simulation parameters
N_AUDIENCES = 10  # Number of unique audience members
OBS_PER_AUDIENCE = 8  # Observations per audience member

# True hyperparameters
hyper_mu_intercept = 2.0    # Population-level intercept
hyper_sigma_intercept = 0.5 # Between-audience intercept variability
hyper_mu_slope = -0.03      # Population-level cost slope 
hyper_sigma_slope = 0.01    # Between-audience slope variability

# Simulate audience-specific parameters
audience_ids = np.repeat(np.arange(N_AUDIENCES), OBS_PER_AUDIENCE)
true_intercepts = np.random.normal(hyper_mu_intercept, hyper_sigma_intercept, N_AUDIENCES)
true_slopes = np.random.normal(hyper_mu_slope, hyper_sigma_slope, N_AUDIENCES)

# Simulate cost and clicks
cost = np.random.gamma(shape=5, scale=1, size=N_AUDIENCES*OBS_PER_AUDIENCE)
eta = true_intercepts[audience_ids] + true_slopes[audience_ids] * cost
clicks = np.random.poisson(lam=np.exp(eta))

# Create DataFrame
data = pd.DataFrame({
    "audience_id": audience_ids,
    "cost": cost,
    "clicks": clicks
})

# Visualize relationships for 6 random audiences
sample_audiences = np.random.choice(N_AUDIENCES, 6, replace=False)

with pm.Model() as abc_model:
    # Hyperpriors for population-level parameters
    mu_intercept = pm.Normal("mu_intercept", mu=0, sigma=2)
    sigma_intercept = pm.HalfNormal("sigma_intercept", sigma=0.5)
    
    mu_slope = pm.Normal("mu_slope", mu=0, sigma=0.1)
    sigma_slope = pm.HalfNormal("sigma_slope", sigma=0.05)
    
    alpha_z = pm.Normal("alpha_z", 0, 1, shape=N_AUDIENCES)
    alpha = pm.Deterministic("alpha", mu_intercept + alpha_z * sigma_intercept) #varying intercept
    
    beta_z = pm.Normal("beta_z", 0, 1, shape=N_AUDIENCES)
    beta = pm.Deterministic("beta", mu_slope + beta_z * sigma_slope) #varying slope for clicks
    
    def simulator_model(rng, alpha, beta, size = None):
        audience_ids = data.audience_id
        cost = data.cost
        eta = alpha[audience_ids] + beta[audience_ids]*cost
        lam = np.exp(eta)
        clicks = np.random.poisson(lam=abs(eta))
        return clicks

    pm.Simulator(
        "Y_obs",
        simulator_model,
        params = (alpha, beta),
        distance = "gaussian",
        sum_stat = "sort",
        epsilon = 10,
        observed = clicks,
    )

with abc_model:
    trace = pm.sample_smc(
        draws = 100, 
        chains = 4,
        cores = 4,
        compute_convergence_checks = True,
        return_inferencedata = True,
        progressbar = True
    )

    textsize = 7
    for plot in ["trace"]:
        az.plot_trace(trace, kind = plot, plot_kwargs = {"textsize": textsize})
        plt.show()

    df = trace.to_dataframe(include_coords=False, groups="posterior")
    X = df[["mu_intercept", "mu_slope", "sigma_intercept", "sigma_slope"]]
    Y = df["chain"].astype("float")
    
    reg = GaussianProcessRegressor()
    reg.fit(X, Y)

    X_cols = list(X.columns)
    bounds = [
        [X[col].min(), X[col].max()]
        for col in X_cols
    ]

    problem = {
        'num_vars': len(X_cols),
        'names': X_cols,
        'bounds': bounds
    }
    
    param_values = saltelli.sample(problem, 64, calc_second_order=True)
    preds = reg.predict(param_values)
    sobol_indices = sobol.analyze(problem, preds, print_to_console=True)
    plt.bar(problem['names'], sobol_indices['S1'], yerr=sobol_indices['S1_conf'])
    plt.title("First-order Sobol Sensitivity Indices")
    plt.ylabel("S1 index")
    plt.show()