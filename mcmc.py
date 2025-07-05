import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, beta, gamma, multivariate_normal

# --- Helper Functions for MCMC Algorithms ---

def metropolis_hastings(target_pdf, proposal_std, initial_val, num_samples):
    """
    Performs Metropolis-Hastings sampling for a 1D target distribution.

    Args:
        target_pdf (function): The probability density function of the target distribution.
        proposal_std (float): The standard deviation of the normal proposal distribution.
        initial_val (float): The starting value for the chain.
        num_samples (int): The number of samples to generate.

    Returns:
        np.ndarray: An array of generated samples.
    """
    samples = np.zeros(num_samples)
    current_val = initial_val
    
    # Calculate the PDF of the current value, handle potential zero PDF at start
    current_pdf = target_pdf(current_val)
    if current_pdf == 0:
        # Avoid division by zero. This can happen if the initial value is outside the support of the distribution (e.g., negative for Gamma).
        # A small perturbation or a better initial value is needed. Here we add a small value to avoid error.
        current_pdf = 1e-10


    for i in range(num_samples):
        # Propose a new value from a normal distribution centered at the current value
        proposed_val = np.random.normal(current_val, proposal_std)
        proposed_pdf = target_pdf(proposed_val)

        # Calculate the acceptance ratio
        if current_pdf > 0:
            acceptance_ratio = proposed_pdf / current_pdf
        else:
            acceptance_ratio = 1.0 # Always accept if current_pdf was 0

        # Accept or reject the new value
        if np.random.uniform(0, 1) < acceptance_ratio:
            current_val = proposed_val
            current_pdf = proposed_pdf if proposed_pdf > 0 else 1e-10
        
        samples[i] = current_val
        
    return samples

def gibbs_sampler(mu, cov, initial_vals, num_samples):
    """
    Performs Gibbs sampling for a 2D Bivariate Normal distribution.

    Args:
        mu (list or np.ndarray): The mean vector [mu_x, mu_y].
        cov (list or np.ndarray): The covariance matrix [[var_x, cov_xy], [cov_xy, var_y]].
        initial_vals (list): The starting values [x0, y0].
        num_samples (int): The number of samples to generate for each variable.

    Returns:
        np.ndarray: A (num_samples x 2) array of generated [x, y] samples.
    """
    samples = np.zeros((num_samples, 2))
    current_vals = np.array(initial_vals, dtype=float)
    
    # Pre-calculate conditional distribution parameters
    mu_x, mu_y = mu[0], mu[1]
    var_x, var_y = cov[0][0], cov[1][1]
    cov_xy = cov[0][1]
    
    # Conditional std deviations
    cond_std_x = np.sqrt(var_x - (cov_xy**2 / var_y))
    cond_std_y = np.sqrt(var_y - (cov_xy**2 / var_x))

    for i in range(num_samples):
        # Sample x from P(x | y)
        cond_mean_x = mu_x + (cov_xy / var_y) * (current_vals[1] - mu_y)
        current_vals[0] = np.random.normal(cond_mean_x, cond_std_x)

        # Sample y from P(y | x)
        cond_mean_y = mu_y + (cov_xy / var_x) * (current_vals[0] - mu_x)
        current_vals[1] = np.random.normal(cond_mean_y, cond_std_y)
        
        samples[i, :] = current_vals
        
    return samples


# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("📊 Markov Chain Monte Carlo (MCMC) Data Generator")

st.markdown("""
This application allows you to generate data from various probability distributions using MCMC methods. 
You can choose between the **Metropolis-Hastings** algorithm for univariate distributions and **Gibbs Sampling** for a bivariate normal distribution.
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ MCMC Parameters")

    mcmc_algorithm = st.selectbox(
        "Select MCMC Algorithm",
        ("Metropolis-Hastings", "Gibbs Sampling"),
        help="Choose the sampling method. Gibbs sampling is configured for a Bivariate Normal target."
    )

    st.markdown("---")
    
    if mcmc_algorithm == "Metropolis-Hastings":
        st.subheader("Target Distribution (1D)")
        distribution_name = st.selectbox(
            "Select Distribution",
            ("Normal", "Beta", "Gamma")
        )
        
        # Parameters for 1D distributions
        if distribution_name == "Normal":
            target_mu = st.slider("Mean (μ)", -10.0, 10.0, 0.0, 0.1)
            target_sigma = st.slider("Standard Deviation (σ)", 0.1, 10.0, 1.0, 0.1)
            target_pdf = lambda x: norm.pdf(x, loc=target_mu, scale=target_sigma)
            true_params = {'Mean': target_mu, 'Std Dev': target_sigma, 'Variance': target_sigma**2}
        
        elif distribution_name == "Beta":
            target_alpha = st.slider("Shape (α)", 0.1, 10.0, 2.0, 0.1)
            target_beta_param = st.slider("Shape (β)", 0.1, 10.0, 5.0, 0.1)
            target_pdf = lambda x: beta.pdf(x, a=target_alpha, b=target_beta_param)
            mean = target_alpha / (target_alpha + target_beta_param)
            variance = (target_alpha * target_beta_param) / ((target_alpha + target_beta_param)**2 * (target_alpha + target_beta_param + 1))
            true_params = {'Mean': mean, 'Variance': variance}

        elif distribution_name == "Gamma":
            target_shape = st.slider("Shape (k)", 0.1, 10.0, 2.0, 0.1)
            target_scale = st.slider("Scale (θ)", 0.1, 10.0, 1.0, 0.1)
            target_pdf = lambda x: gamma.pdf(x, a=target_shape, scale=target_scale)
            true_params = {'Mean': target_shape * target_scale, 'Variance': target_shape * target_scale**2}

        st.subheader("Algorithm Settings")
        initial_val = st.number_input("Initial Value", value=0.0)
        proposal_std = st.slider("Proposal Std Dev", 0.1, 10.0, 1.0, 0.1, help="Controls the step size in Metropolis-Hastings.")

    else: # Gibbs Sampling
        st.subheader("Target Distribution (Bivariate Normal)")
        distribution_name = "Bivariate Normal"
        st.markdown("Set the parameters for the target `N(μ, Σ)` distribution.")
        
        mu_x = st.slider("Mean (μ_x)", -5.0, 5.0, 0.0, 0.1)
        mu_y = st.slider("Mean (μ_y)", -5.0, 5.0, 0.0, 0.1)
        
        sigma_x = st.slider("Std Dev (σ_x)", 0.1, 5.0, 1.0, 0.1)
        sigma_y = st.slider("Std Dev (σ_y)", 0.1, 5.0, 1.0, 0.1)
        
        rho = st.slider("Correlation (ρ)", -0.99, 0.99, 0.5, 0.01)
        
        # Construct parameters for the sampler
        gibbs_mu = [mu_x, mu_y]
        gibbs_cov = [
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2]
        ]
        
        true_params = {
            'Mean (X)': mu_x, 'Mean (Y)': mu_y,
            'Variance (X)': sigma_x**2, 'Variance (Y)': sigma_y**2,
            'Covariance': rho * sigma_x * sigma_y
        }

        st.subheader("Algorithm Settings")
        initial_x = st.number_input("Initial Value (X)", value=0.0)
        initial_y = st.number_input("Initial Value (Y)", value=0.0)
        initial_vals_gibbs = [initial_x, initial_y]

    st.markdown("---")
    st.subheader("Simulation Settings")
    num_samples = st.number_input("Number of Samples", min_value=100, max_value=50000, value=10000, step=100)
    burn_in = st.slider("Burn-in Period", 0, int(num_samples * 0.5), int(num_samples * 0.1), 100,
                        help="Number of initial samples to discard to let the chain converge.")

    generate_button = st.button("🚀 Generate Data & Analyze")

# --- Main Area for Output ---

if generate_button:
    with st.spinner(f"Running {mcmc_algorithm}... this may take a moment."):
        # --- Data Generation ---
        if mcmc_algorithm == "Metropolis-Hastings":
            samples_raw = metropolis_hastings(target_pdf, proposal_std, initial_val, num_samples)
            samples_burned = samples_raw[burn_in:]
            df = pd.DataFrame(samples_burned, columns=['Sample'])
            df_raw = pd.DataFrame(samples_raw, columns=['Sample'])

        else: # Gibbs Sampling
            samples_raw = gibbs_sampler(gibbs_mu, gibbs_cov, initial_vals_gibbs, num_samples)
            samples_burned = samples_raw[burn_in:, :]
            df = pd.DataFrame(samples_burned, columns=['X', 'Y'])
            df_raw = pd.DataFrame(samples_raw, columns=['X', 'Y'])
    
    st.success(f"✅ Simulation Complete! Generated {len(df)} samples after a burn-in of {burn_in}.")
    
    # --- Analysis and Visualization ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Trace Plots")
        st.markdown("Shows the history of the sampled values. A good trace plot should look like stationary noise, without any clear trend.")
        
        fig_trace, ax_trace = plt.subplots(figsize=(10, 5))
        if mcmc_algorithm == "Metropolis-Hastings":
            ax_trace.plot(df_raw.index, df_raw['Sample'], lw=0.8)
            ax_trace.axvline(x=burn_in, color='r', linestyle='--', label=f'Burn-in ({burn_in})')
            ax_trace.set_title(f"Trace Plot for {distribution_name}")
            ax_trace.set_xlabel("Iteration")
            ax_trace.set_ylabel("Sampled Value")
            ax_trace.legend()
        else:
            # For Gibbs, plot traces for both variables
            ax_trace.plot(df_raw.index, df_raw['X'], lw=0.8, label='Trace of X')
            ax_trace.plot(df_raw.index, df_raw['Y'], lw=0.8, label='Trace of Y', alpha=0.7)
            ax_trace.axvline(x=burn_in, color='r', linestyle='--', label=f'Burn-in ({burn_in})')
            ax_trace.set_title("Trace Plots for X and Y")
            ax_trace.set_xlabel("Iteration")
            ax_trace.set_ylabel("Sampled Value")
            ax_trace.legend()
        
        st.pyplot(fig_trace)

    with col2:
        st.subheader("📊 Sample Distribution")
        st.markdown("Visualizes the distribution of the samples after the burn-in period.")
        
        fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
        if mcmc_algorithm == "Metropolis-Hastings":
            sns.histplot(df['Sample'], kde=True, ax=ax_dist, stat="density", label="MCMC Samples")
            
            # Overlay true PDF
            xmin, xmax = ax_dist.get_xlim()
            x_vals = np.linspace(xmin, xmax, 200)
            y_vals = [target_pdf(x) for x in x_vals]
            ax_dist.plot(x_vals, y_vals, 'r--', label=f'True {distribution_name} PDF')
            
            ax_dist.set_title(f"Histogram of Samples vs. True PDF")
            ax_dist.set_xlabel("Value")
            ax_dist.set_ylabel("Density")
            ax_dist.legend()
        else: # Gibbs
            sns.kdeplot(x=df['X'], y=df['Y'], ax=ax_dist, cmap="viridis", fill=True)
            ax_dist.set_title("2D Kernel Density Estimate of Samples")
            ax_dist.set_xlabel("X Value")
            ax_dist.set_ylabel("Y Value")
            
            # Overlay true mean
            ax_dist.plot(gibbs_mu[0], gibbs_mu[1], 'r+', markersize=15, markeredgewidth=2, label='True Mean')
            ax_dist.legend()

        st.pyplot(fig_dist)

    st.markdown("---")
    
    # --- Data Display and Statistics ---
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🔢 Generated Data Sample")
        st.markdown(f"Showing the first 200 samples (out of {len(df)}).")
        st.dataframe(df.head(200))

    with col4:
        st.subheader("📋 Descriptive Statistics")
        st.markdown("Comparison of empirical stats from samples vs. theoretical values.")
        
        stats_df = df.describe().transpose()
        
        # Create a dataframe for comparison
        comparison_df = pd.DataFrame({
            'Statistic': list(true_params.keys()),
            'True Value': list(true_params.values())
        }).set_index('Statistic')
        
        if mcmc_algorithm == "Metropolis-Hastings":
            # --- FIX STARTS HERE ---
            # Dynamically create the list of estimates based on the statistics required
            mcmc_estimates = []
            sample_mean = stats_df.loc['Sample', 'mean']
            sample_std = stats_df.loc['Sample', 'std']
            sample_var = sample_std**2
            
            for stat_name in comparison_df.index:
                if stat_name == 'Mean':
                    mcmc_estimates.append(sample_mean)
                elif stat_name == 'Std Dev':
                    mcmc_estimates.append(sample_std)
                elif stat_name == 'Variance':
                    mcmc_estimates.append(sample_var)
            
            comparison_df['MCMC Estimate'] = mcmc_estimates
            # --- FIX ENDS HERE ---

        else: # Gibbs
            comparison_df['MCMC Estimate (X)'] = [stats_df.loc['X', 'mean'], np.nan, stats_df.loc['X', 'std']**2, np.nan, np.nan]
            comparison_df['MCMC Estimate (Y)'] = [np.nan, stats_df.loc['Y', 'mean'], np.nan, stats_df.loc['Y', 'std']**2, np.nan]
            # Calculate covariance from samples
            sample_cov = np.cov(df['X'], df['Y'])[0, 1]
            comparison_df.loc['Covariance', 'MCMC Estimate (X)'] = sample_cov
        
        st.table(comparison_df.style.format("{:.4f}"))

else:
    st.info("👈 Select your parameters in the sidebar and click 'Generate Data & Analyze' to begin.")
