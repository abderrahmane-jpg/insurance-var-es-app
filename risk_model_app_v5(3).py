from __future__ import annotations

import io
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import genpareto, norm, gaussian_kde
from scipy.optimize import minimize
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


APP_VERSION = "5.1"
DEFAULT_LAMBDA = 0.95
DEFAULT_CONFIDENCE = 0.99
DEFAULT_THRESHOLD = 100_000_000.0
DEFAULT_GPD_SHAPE = 0.05
DEFAULT_GPD_SCALE = 2_000_000_000.0
DEFAULT_SIMULATIONS = 10_000
DEFAULT_SEED = 42

st.set_page_config(page_title="Insurance VaR & ES", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------
# Data input and common scenario construction
# ---------------------------------------------------------------------
def read_workbook(uploaded_file):
    """Read all sheets; allow the user to select the sheet and columns."""
    xls = pd.ExcelFile(uploaded_file)
    frames = {}
    for sheet in xls.sheet_names:
        try:
            frames[sheet] = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue
    if not frames:
        raise ValueError("No readable worksheet was found.")
    return frames


def clean_input_frame(frame, year_col, risk_col):
    out = frame[[year_col, risk_col]].copy()
    out.columns = ["Year", "Risk Factor"]
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["Risk Factor"] = pd.to_numeric(out["Risk Factor"], errors="coerce")
    out = out.dropna(subset=["Year", "Risk Factor"]).copy()
    if len(out) < 4:
        raise ValueError("At least four valid observations are required to estimate returns.")
    if (out["Risk Factor"] <= 0).any():
        raise ValueError(
            "Risk Factor values must be strictly positive because returns are calculated as X_t / X_(t-1) - 1."
        )
    out = out.sort_values("Year").drop_duplicates(subset=["Year"], keep="last").reset_index(drop=True)
    if len(out) < 4:
        raise ValueError("At least four unique, valid observations are required.")
    return out


def construct_scenarios(df, target):
    """Workbook convention: scenario_t = target * X_t / X_(t-1); loss_t = target - scenario_t."""
    x = df["Risk Factor"].to_numpy(dtype=float)
    years = df["Year"].to_numpy()
    returns = x[1:] / x[:-1] - 1.0
    # Revalue the LAST observed risk-factor level
    # under each historical return; target is a separate loss benchmark.
    base_value = float(x[-1])
    scenario = base_value * x[1:] / x[:-1]
    losses = target - scenario
    return pd.DataFrame({
        "Scenario": np.arange(1, len(returns) + 1),
        "Year": years[1:],
        "Previous Risk Factor": x[:-1],
        "Risk Factor": x[1:],
        "Return": returns,
        "Scenario Estimate": scenario,
        "Loss": losses,
    })


def empirical_var_es(losses, confidence):
    """Linear empirical quantile; ES is the mean of observed losses strictly beyond VaR."""
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, confidence, method="linear"))
    tail = losses[losses > var]
    if len(tail) == 0:
        # This can happen when quantile interpolation lands on the maximum.
        tail = losses[losses >= var]
    es = float(np.mean(tail)) if len(tail) else var
    return var, es


# ---------------------------------------------------------------------
# Method 1 — normal parametric method fitted to positive losses only
# ---------------------------------------------------------------------
def parametric_method(losses, confidence):
    losses = np.asarray(losses, dtype=float)
    positive = losses[losses > 0]
    if len(positive) < 2:
        raise ValueError("Parametric method needs at least two positive historical losses.")
    mu = float(np.mean(positive))
    sigma = float(np.std(positive, ddof=1))
    z = float(norm.ppf(confidence))
    var = mu + z * sigma
    es = mu + sigma * norm.pdf(z) / (1.0 - confidence)
    return {
        "VaR": float(var), "ES": float(es), "mu": mu, "sigma": sigma,
        "z": z, "positive_losses": positive,
    }


# ---------------------------------------------------------------------
# Method 2 — equally weighted historical simulation
# ---------------------------------------------------------------------
def equally_weighted_method(losses, confidence):
    var, es = empirical_var_es(losses, confidence)
    return {"VaR": var, "ES": es, "losses": np.asarray(losses, dtype=float)}


# ---------------------------------------------------------------------
# Method 3 — exponentially weighted historical simulation
# ---------------------------------------------------------------------
def exponential_weights(n, lam):
    if not 0 < lam < 1:
        raise ValueError("Lambda must be strictly between 0 and 1.")
    # Oldest observation receives lambda**(n-1); newest receives the largest weight.
    powers = np.arange(n - 1, -1, -1, dtype=float)
    weights = (1.0 - lam) * lam ** powers / (1.0 - lam ** n)
    return weights / weights.sum()


def weighted_var_es(losses, weights, confidence):
    """Weighted VaR/ES. ES uses fractional tail mass so the tail probability is exactly 1-confidence."""
    losses = np.asarray(losses, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(losses) != len(weights) or len(losses) == 0:
        raise ValueError("Losses and weights must have the same non-zero length.")
    weights = weights / weights.sum()
    order = np.argsort(losses)  # ascending for VaR
    x = losses[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    var_idx = min(int(np.searchsorted(cumulative, confidence, side="left")), len(x) - 1)
    var = float(x[var_idx])

    # Expected shortfall = weighted average of the worst 1-alpha probability mass.
    tail_probability = 1.0 - confidence
    order_desc = np.argsort(losses)[::-1]
    x_desc = losses[order_desc]
    w_desc = weights[order_desc]
    remaining = tail_probability
    weighted_tail_sum = 0.0
    for value, weight in zip(x_desc, w_desc):
        take = min(float(weight), remaining)
        if take > 0:
            weighted_tail_sum += float(value) * take
            remaining -= take
        if remaining <= 1e-14:
            break
    es = weighted_tail_sum / tail_probability
    return var, float(es), var_idx


def exponential_weighted_method(scenarios, confidence, lam):
    losses = scenarios["Loss"].to_numpy(dtype=float)
    weights_chronological = exponential_weights(len(losses), lam)
    var, es, var_idx = weighted_var_es(losses, weights_chronological, confidence)
    table = scenarios[["Scenario", "Year", "Loss"]].copy()
    table["Weight"] = weights_chronological
    table["Cumulative Weight (ascending loss)"] = np.nan
    ascending = np.argsort(losses)
    table.loc[ascending, "Cumulative Weight (ascending loss)"] = np.cumsum(weights_chronological[ascending])
    return {
        "VaR": var, "ES": es, "table": table, "weights": weights_chronological,
        "cumulative_weights": np.cumsum(weights_chronological), "var_index": var_idx,
    }


# ---------------------------------------------------------------------
# Method 4 — data volatility estimation
# ---------------------------------------------------------------------
def data_volatility_method(df, target, confidence, lam):
    x = df["Risk Factor"].to_numpy(dtype=float)
    years = df["Year"].to_numpy()
    returns = x[1:] / x[:-1] - 1.0
    n = len(returns)
    initial_var = float(np.var(returns, ddof=1))
    if not np.isfinite(initial_var) or initial_var <= 0:
        raise ValueError("The return series does not have a positive initial variance.")

    # Match the workbook's timing: scenario 1 uses the initial variance.
    # The next variance is updated using the previous return, not the current return.
    variance_for_scenario = np.empty(n, dtype=float)
    state_var = initial_var
    for i in range(n):
        variance_for_scenario[i] = state_var
        state_var = lam * state_var + (1.0 - lam) * returns[i] ** 2
    final_vol = math.sqrt(state_var)
    vol = np.sqrt(variance_for_scenario)
    vol_ratio = final_vol / vol
    # Apply adjusted historical returns to
    # the last observed risk-factor level, then compare with the target.
    base_value = float(x[-1])
    scenario = base_value * (x[:-1] + (x[1:] - x[:-1]) * vol_ratio) / x[:-1]
    adjusted_losses = target - scenario
    var, es = empirical_var_es(adjusted_losses, confidence)
    table = pd.DataFrame({
        "Scenario": np.arange(1, n + 1), "Year": years[1:], "Return": returns,
        "Variance before scenario": variance_for_scenario, "Volatility": vol,
        "Final volatility": final_vol, "Volatility ratio": vol_ratio,
        "Scenario estimate": scenario, "Adjusted loss": adjusted_losses,
    })
    return {
        "VaR": var, "ES": es, "table": table, "vol_ratio": vol_ratio,
        "final_volatility": final_vol, "initial_variance": initial_var,
    }


# ---------------------------------------------------------------------
# Method 5 — losses adjusted by standard deviation
# ---------------------------------------------------------------------
def losses_adjusted_sd_method(scenarios, confidence, lam):
    losses = scenarios["Loss"].to_numpy(dtype=float)
    n = len(losses)
    initial_var = float(np.var(losses, ddof=1))
    if not np.isfinite(initial_var) or initial_var <= 0:
        raise ValueError("Historical losses do not have a positive initial variance.")

    # Match the workbook's timing: the first loss uses the initial variance;
    # subsequent variance states use the previous observation's squared loss.
    variance_for_loss = np.empty(n, dtype=float)
    state_var = initial_var
    for i in range(n):
        variance_for_loss[i] = state_var
        state_var = lam * state_var + (1.0 - lam) * losses[i] ** 2
    final_sd = math.sqrt(state_var)
    sd = np.sqrt(variance_for_loss)
    ratio = final_sd / sd
    adjusted = losses * ratio
    var, es = empirical_var_es(adjusted, confidence)
    table = pd.DataFrame({
        "Scenario": scenarios["Scenario"], "Year": scenarios["Year"], "Loss": losses,
        "Variance before scenario": variance_for_loss, "Standard deviation": sd,
        "Final standard deviation": final_sd, "SD ratio": ratio,
        "Adjusted loss": adjusted,
    })
    return {
        "VaR": var, "ES": es, "table": table, "sd_ratio": ratio,
        "final_sd": final_sd, "initial_variance": initial_var,
    }


# ---------------------------------------------------------------------
# Method 6 — Peaks Over Threshold / Generalized Pareto Distribution
# ---------------------------------------------------------------------
def validate_gpd_parameters(shape, scale):
    if not np.isfinite(shape):
        raise ValueError("GPD shape must be finite.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("GPD scale must be strictly positive.")
    if shape >= 1:
        raise ValueError("GPD shape must be below 1 for a finite Expected Shortfall.")
    return float(shape), float(scale)


def fit_gpd_mle(losses, threshold):
    """Fit a GPD to threshold exceedances by maximum likelihood (loc fixed at 0)."""
    losses = np.asarray(losses, dtype=float)
    excesses = losses[losses > threshold] - threshold
    if len(excesses) < 3:
        raise ValueError("GPD MLE needs at least three losses strictly above the POT threshold.")

    # scipy's fit uses maximum likelihood. Fixing loc=0 is required because
    # the excess variable is defined relative to the chosen threshold.
    xi, loc, beta = genpareto.fit(excesses, floc=0.0)
    xi, beta = float(xi), float(beta)
    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0:
        raise ValueError("GPD maximum-likelihood estimation returned invalid parameters.")
    if xi >= 1.0:
        raise ValueError(
            f"GPD MLE estimated xi={xi:.4f} >= 1, so the theoretical ES is infinite. "
            "Try a different POT threshold or use the manual parameter option."
        )
    # Check support for every fitted excess.
    if np.any(1.0 + xi * excesses / beta <= 0):
        raise ValueError("The fitted GPD parameters violate the support condition for an observed exceedance.")
    return {"shape": xi, "scale": beta, "exceedances": excesses,
            "log_likelihood": float(np.sum(genpareto.logpdf(excesses, c=xi, loc=0.0, scale=beta)))}


def evt_pot_method(losses, confidence, threshold, shape, scale):
    losses = np.asarray(losses, dtype=float)
    shape, scale = validate_gpd_parameters(shape, scale)
    if threshold < 0:
        raise ValueError("POT threshold should be non-negative for this loss-tail setup.")
    exceedances = losses[losses > threshold] - threshold
    n = len(losses)
    nu = len(exceedances)
    if nu < 3:
        raise ValueError("EVT needs at least three losses strictly above the POT threshold.")
    tail_probability = 1.0 - confidence
    exceedance_rate = nu / n
    if tail_probability >= exceedance_rate:
        raise ValueError(
            "The requested confidence level is not in the fitted POT tail. Lower the threshold or use a higher confidence level."
        )
    if abs(shape) < 1e-10:
        var = threshold + scale * math.log(exceedance_rate / tail_probability)
        es = var + scale
    else:
        var = threshold + (scale / shape) * (
            (exceedance_rate / tail_probability) ** shape - 1.0
        )
        es = (var + scale - shape * threshold) / (1.0 - shape)
    return {
        "VaR": float(var), "ES": float(es), "threshold": float(threshold),
        "shape": shape, "scale": scale, "n": n, "nu": nu,
        "exceedance_rate": exceedance_rate, "exceedances": exceedances,
        "positive_losses": losses[losses > 0],
    }


# ---------------------------------------------------------------------
# Method 7 — lognormal Monte Carlo fitted to ALL positive historical losses
# ---------------------------------------------------------------------
def lognormal_mc_method(losses, confidence, simulations, seed):
    losses = np.asarray(losses, dtype=float)
    positive = losses[losses > 0]
    if len(positive) < 2:
        raise ValueError("Lognormal Monte Carlo needs at least two positive historical losses.")
    if simulations < 100:
        raise ValueError("Use at least 100 Monte Carlo simulations.")
    log_positive = np.log(positive)
    mu = float(np.mean(log_positive))
    # Excel STDEV uses the sample standard deviation (n-1 denominator).
    sigma = float(np.std(log_positive, ddof=1))
    if sigma <= 0:
        raise ValueError("The log-positive-loss sample has zero dispersion.")
    rng = np.random.default_rng(int(seed))
    uniforms = np.clip(rng.random(int(simulations)), 1e-12, 1 - 1e-12)
    # Use the same uniform draws that will be written into Excel, so the
    # Excel LOGNORM.INV formulas reproduce these Python simulation draws.
    simulated = np.exp(mu + sigma * norm.ppf(uniforms))
    var, es = empirical_var_es(simulated, confidence)
    return {
        "VaR": var, "ES": es, "mu": mu, "sigma": sigma,
        "positive_losses": positive, "log_positive_losses": log_positive,
        "uniforms": uniforms, "simulated": simulated,
    }


# ---------------------------------------------------------------------
# Method 8 — EVT Monte Carlo, using the same POT threshold and GPD parameters
# ---------------------------------------------------------------------
def evt_mc_method(losses, confidence, simulations, threshold, seed):
    """
    EVT Monte Carlo implementation.

    Calculation steps:
      1) Sort positive historical losses in descending order.
      2) Compute y_i = max(0, loss_i - u) for those positive losses.
      3) Method of moments: beta = 2*s^2/mean(y); xi = 1 - mean(y)/beta.
      4) Draw U~Uniform(0,1), then X = (beta/xi)*(U**(-xi)-1).
         Simulate excess losses without adding u.
      5) Set adjusted loss to zero when X >= empirical alpha-percentile.
      6) Report VaR as the mean adjusted loss (not the percentile).
      7) Report ES as the mean adjusted loss strictly greater than that VaR.
    This implementation uses the specified nonstandard VaR convention and top-tail truncation.
    """
    losses = np.asarray(losses, dtype=float)
    if simulations < 100:
        raise ValueError("Use at least 100 Monte Carlo simulations.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence level must be strictly between 0 and 1.")
    if threshold < 0:
        raise ValueError("POT threshold must be non-negative.")

    positive_losses = np.sort(losses[losses > 0])[::-1]
    if len(positive_losses) < 2:
        raise ValueError("EVT Monte Carlo needs at least two positive historical losses.")

    # Calculate y_i = MAX(0, x_i - u) for the positive-loss sample.
    fit_excesses = np.maximum(positive_losses - float(threshold), 0.0)
    mean_excess = float(np.mean(fit_excesses))
    sd_excess = float(np.std(fit_excesses, ddof=1))
    if not np.isfinite(mean_excess) or mean_excess <= 0:
        raise ValueError("The method-of-moments fit requires a positive mean excess.")
    if not np.isfinite(sd_excess) or sd_excess <= 0:
        raise ValueError("The method-of-moments fit requires a positive sample SD.")

    moment_scale = float(2.0 * sd_excess**2 / mean_excess)
    moment_shape = float(1.0 - mean_excess / moment_scale)
    if not np.isfinite(moment_scale) or moment_scale <= 0:
        raise ValueError("The method produced an invalid GPD scale.")
    if abs(moment_shape) < 1e-10:
        # Continuous limit as xi -> 0 of beta/xi * (U**(-xi) - 1).
        moment_shape = 0.0

    rng = np.random.default_rng(int(seed) + 1)
    uniforms = np.clip(rng.random(int(simulations)), 1e-12, 1.0)
    if abs(moment_shape) < 1e-10:
        raw_simulated = -moment_scale * np.log(uniforms)
    else:
        raw_simulated = (moment_scale / moment_shape) * (
            uniforms ** (-moment_shape) - 1.0
        )

    cutoff = float(np.percentile(raw_simulated, confidence * 100.0, method="linear"))
    adjusted_simulated = np.where(raw_simulated >= cutoff, 0.0, raw_simulated)
    var = float(np.mean(adjusted_simulated))
    excess_after_var = np.where(adjusted_simulated > var, adjusted_simulated, 0.0)
    positive_tail = excess_after_var[excess_after_var > 0]
    es = float(np.mean(positive_tail)) if len(positive_tail) else 0.0

    return {
        "VaR": var,
        "ES": es,
        "simulated": adjusted_simulated,
        "raw_simulated": raw_simulated,
        "adjusted_simulated": adjusted_simulated,
        "excess_after_var": excess_after_var,
        "uniforms": uniforms,
        "cutoff": cutoff,
        "threshold": float(threshold),
        "positive_losses": positive_losses,
        "fit_excesses": fit_excesses,
        "mean_excess": mean_excess,
        "sd_excess": sd_excess,
        "moment_scale": moment_scale,
        "moment_shape": moment_shape,
    }


def run_model(df, target, confidence, lam, threshold, shape, scale, simulations, seed, estimate_gpd_mle=True):
    scenarios = construct_scenarios(df, target)
    losses = scenarios["Loss"].to_numpy(dtype=float)
    gpd_fit = None
    if estimate_gpd_mle:
        gpd_fit = fit_gpd_mle(losses, threshold)
        shape, scale = gpd_fit["shape"], gpd_fit["scale"]
    shape, scale = validate_gpd_parameters(shape, scale)
    methods = {
        "Parametric": parametric_method(losses, confidence),
        "Equally Weighted": equally_weighted_method(losses, confidence),
        "Expo Weighted": exponential_weighted_method(scenarios, confidence, lam),
        "Data Vol Est": data_volatility_method(df, target, confidence, lam),
        "Losses Adj SD": losses_adjusted_sd_method(scenarios, confidence, lam),
        "EVT": evt_pot_method(losses, confidence, threshold, shape, scale),
        "Ln-Monte Carlo": lognormal_mc_method(losses, confidence, simulations, seed),
        "EVT-Monte Carlo": evt_mc_method(losses, confidence, simulations, threshold, seed),
    }
    rows = []
    approach = {
        "Parametric": "Deterministic", "Equally Weighted": "Deterministic",
        "Expo Weighted": "Deterministic", "Data Vol Est": "Deterministic",
        "Losses Adj SD": "Deterministic", "EVT": "Deterministic",
        "Ln-Monte Carlo": "Stochastic", "EVT-Monte Carlo": "Stochastic",
    }
    for name, result in methods.items():
        rows.append({"Approach": approach[name], "Model": name,
                     "Value-at-Risk": result["VaR"], "Expected Shortfall": result["ES"]})
    summary = pd.DataFrame(rows)
    average = {"Approach": "Average", "Model": "All eight methods",
               "Value-at-Risk": float(summary["Value-at-Risk"].mean()),
               "Expected Shortfall": float(summary["Expected Shortfall"].mean())}
    summary = pd.concat([summary, pd.DataFrame([average])], ignore_index=True)
    return {"scenarios": scenarios, "losses": losses, "methods": methods, "summary": summary,
            "gpd_fit": gpd_fit, "gpd_shape": shape, "gpd_scale": scale,
            "gpd_estimation_mode": "Maximum likelihood" if estimate_gpd_mle else "Manual parameters"}


# ---------------------------------------------------------------------
# Plots — each one is saved to the worksheet appropriate to that method
# ---------------------------------------------------------------------
def create_plots(df, model, target, confidence):
    scenarios = model["scenarios"]
    losses = model["losses"]
    methods = model["methods"]
    figs = {}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(df["Year"], df["Risk Factor"] / 1e9, marker="o", linewidth=1.8, label="Historical risk factor")
    ax.axhline(target / 1e9, linestyle="--", linewidth=1.5, label="Target")
    target_year = int(round(float(df["Year"].max()))) + 1
    ax.scatter([target_year], [target / 1e9], color="red", zorder=5, label=f"Target point ({target_year})")
    ax.set_xlim(float(df["Year"].min()), target_year + 0.25)
    ax.set_title("Risk factor over time and target")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value (milliards de DA)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figs["Data"] = fig

    fig, ax = plt.subplots(figsize=(9, 4.8))
    positive_losses = losses[losses > 0] / 1e9
    if len(positive_losses) >= 2:
        ax.hist(positive_losses, bins="auto", density=True, alpha=0.55,
                edgecolor="white", label="Fréquence empirique")
        grid = np.linspace(0, max(float(np.max(positive_losses)) * 1.08, 1.0), 300)
        try:
            density = gaussian_kde(positive_losses)
            ax.plot(grid, density(grid), color="darkorange", linewidth=2.2,
                    label="Densité lissée des pertes historiques")
        except (ValueError, np.linalg.LinAlgError):
            pass
    ax.set_title("Distribution historique des pertes positives")
    ax.set_xlabel("Perte (milliards de DA)")
    ax.set_ylabel("Densité")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    figs["Scenario Losses"] = fig

    expo = methods["Expo Weighted"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(expo["table"]["Year"], expo["weights"], marker="o", linewidth=1.8)
    ax.set_title("Exponential weights through time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Weight")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figs["Expo Weighted"] = fig

    dv = methods["Data Vol Est"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(dv["table"]["Year"], dv["vol_ratio"], marker="o", linewidth=1.8)
    ax.axhline(1.0, linestyle="--", linewidth=1.2)
    ax.set_title("Return volatility adjustment ratio")
    ax.set_xlabel("Year")
    ax.set_ylabel("Final volatility / volatility before scenario")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figs["Data Vol Est"] = fig

    adj = methods["Losses Adj SD"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(adj["table"]["Year"], adj["sd_ratio"], marker="o", linewidth=1.8)
    ax.axhline(1.0, linestyle="--", linewidth=1.2)
    ax.set_title("Loss standard-deviation adjustment ratio")
    ax.set_xlabel("Year")
    ax.set_ylabel("Final SD / SD before scenario")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figs["Losses Adj SD"] = fig

    evt = methods["EVT"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    exceedances = np.asarray(evt["exceedances"], dtype=float) / 1e9
    if len(exceedances):
        observed = np.sort(exceedances)
        probs = (np.arange(1, len(observed) + 1) - 0.5) / len(observed)
        theoretical = genpareto.ppf(probs, c=evt["shape"], loc=0,
                                    scale=evt["scale"] / 1e9)
        valid = np.isfinite(theoretical) & np.isfinite(observed)
        if np.any(valid):
            ax.scatter(theoretical[valid], observed[valid], s=30,
                       label="Dépassements observés")
            upper = max(float(np.max(theoretical[valid])), float(np.max(observed[valid])))
            ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1.5,
                    label="Référence 45°")
    ax.set_title("EVT - diagnostic des dépassements au-dessus du seuil")
    ax.set_xlabel("Quantiles GPD théoriques (milliards de DA)")
    ax.set_ylabel("Excès observés (milliards de DA)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    figs["EVT"] = fig

    ln = methods["Ln-Monte Carlo"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(ln["simulated"] / 1e9, bins="auto", density=True, alpha=0.75, edgecolor="white")
    ax.axvline(ln["VaR"] / 1e9, linestyle="--", linewidth=2, label=f"VaR ({confidence:.1%})")
    ax.axvline(ln["ES"] / 1e9, linestyle=":", linewidth=2, label="ES")
    ax.set_title("Lognormal Monte Carlo loss distribution")
    ax.set_xlabel("Perte simulée (milliards de DA)")
    ax.set_ylabel("Densité (par milliard de DA)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    figs["Ln-Monte Carlo"] = fig

    evmc = methods["EVT-Monte Carlo"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(evmc["adjusted_simulated"] / 1e9, bins="auto", density=True, alpha=0.75, edgecolor="white")
    ax.axvline(evmc["VaR"] / 1e9, linestyle="--", linewidth=2, label="VaR")
    ax.axvline(evmc["ES"] / 1e9, linestyle=":", linewidth=2, label="ES")
    ax.set_title("Simulation Monte Carlo EVT")
    ax.set_xlabel("Perte simulée (milliards de DA)")
    ax.set_ylabel("Densité (par milliard de DA)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    figs["EVT-Monte Carlo"] = fig

    fig, ax = plt.subplots(figsize=(10, 5.2))
    summ = model["summary"].iloc[:-1]
    x = np.arange(len(summ))
    width = 0.38
    ax.bar(x - width / 2, summ["Value-at-Risk"] / 1e9, width, label="VaR")
    ax.bar(x + width / 2, summ["Expected Shortfall"] / 1e9, width, label="ES")
    ax.set_xticks(x)
    ax.set_xticklabels(summ["Model"], rotation=35, ha="right")
    ax.set_title("VaR and Expected Shortfall comparison")
    ax.set_ylabel("Montant (milliards de DA)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figs["Summary"] = fig
    return figs


# ---------------------------------------------------------------------
# Excel export — values are written as calculated values so they display
# immediately; method formulas and assumptions are documented on sheets.
# ---------------------------------------------------------------------
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
SUB_FILL = PatternFill("solid", fgColor="D9EAF7")


def style_header(ws, row=1):
    for cell in ws[row]:
        if cell.value is not None:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = HEADER_FILL
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def write_frame(ws, frame, start_row=1, start_col=1):
    for j, col in enumerate(frame.columns, start=start_col):
        ws.cell(start_row, j, str(col))
    for i, row in enumerate(frame.itertuples(index=False, name=None), start=start_row + 1):
        for j, value in enumerate(row, start=start_col):
            if pd.isna(value):
                value = None
            elif isinstance(value, np.generic):
                value = value.item()
            ws.cell(i, j, value)
    style_header(ws, start_row)


def add_note(ws, row, label, value, col=1):
    ws.cell(row, col, label).font = Font(bold=True)
    ws.cell(row, col).fill = SUB_FILL
    ws.cell(row, col + 1, value)


def add_plot_to_sheet(ws, fig, anchor="J2"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = XLImage(buf)
    img.width = 760
    img.height = 430
    ws.add_image(img, anchor)


def autosize(ws, max_width=34):
    for col_idx in range(1, ws.max_column + 1):
        max_len = 0
        for row_idx in range(1, min(ws.max_row, 250) + 1):
            v = ws.cell(row_idx, col_idx).value
            if v is not None:
                max_len = max(max_len, len(str(v)))
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, max_width)
    ws.freeze_panes = "A2"


def build_excel(model, df, target, confidence, lam, threshold, shape, scale, simulations, seed, figs):
    """Build a formula-driven workbook. Excel formulas remain visible in the formula bar."""
    wb = Workbook()
    wb.remove(wb.active)
    try:
        wb.calculation.calcMode = "auto"
        wb.calculation.fullCalcOnLoad = True
        wb.calculation.forceFullCalc = True
    except Exception:
        pass

    def put_headers(ws, headers, row=1):
        for col, val in enumerate(headers, 1):
            ws.cell(row, col, val)
        style_header(ws, row)

    def note(ws, cell, label, value):
        ws[cell] = label
        ws[cell].font = Font(bold=True)
        ws[cell].fill = SUB_FILL
        ws.cell(ws[cell].row, ws[cell].column + 1, value)

    # Data source sheet
    ws = wb.create_sheet("Data")
    put_headers(ws, ["Year", "Fin_Resul"])
    for i, row in enumerate(df.itertuples(index=False, name=None), start=2):
        ws.cell(i, 1, float(row[0]) if isinstance(row[0], (int, float, np.number)) else str(row[0]))
        ws.cell(i, 2, float(row[1]))
    add_plot_to_sheet(ws, figs["Data"], "D2")
    autosize(ws)

    # Parameters and audit trail
    ws = wb.create_sheet("Inputs & Notes")
    put_headers(ws, ["Parameter", "Value / definition"])
    inputs = [
        ("Target", float(target)),
        ("Confidence level alpha", float(confidence)),
        ("Exponential decay lambda", float(lam)),
        ("POT threshold u", float(threshold)),
        ("GPD shape xi (initial value for Solver)", float(shape)),
        ("GPD scale beta (initial value for Solver)", float(scale)),
        ("Monte Carlo simulations", int(simulations)),
        ("Random seed", int(seed)),
        ("GPD estimation method", model["gpd_estimation_mode"]),
        ("Generated at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    for r, (name, val) in enumerate(inputs, start=2):
        ws.cell(r, 1, name)
        ws.cell(r, 2, val)
    notes = [
        "Scenario = last observed risk-factor level × current risk factor / previous risk factor.",
        "Loss = target − scenario estimate; positive losses are shortfalls.",
        "Parametric method: Normal distribution fitted to strictly positive historical losses only.",
        "Lognormal Monte Carlo: fit ln(loss) to every strictly positive historical loss; use sample SD (n−1).",
        "Monte Carlo uniforms are stored as values; simulated losses are Excel formulas so the formula bar shows the model transformation and results are reproducible.",
        "Analytical EVT: fit or enter the GPD parameters as selected in the app.",
        "EVT Monte Carlo: method-of-moments fit, inverse-GPD excess draws, percentile truncation, mean-adjusted-loss VaR, and average excess above that VaR.",
        "MLE maximizes the GPD log-likelihood. EVT sheet contains an objective cell and log-likelihood contributions for Excel Solver.",
        "Solver setup: Set objective EVT!B7 to Max; change Inputs & Notes!B6:B7; require beta > 0 and xi < 1. The likelihood formulas penalize invalid GPD support.",
        "If GPD estimation mode is Manual parameters, xi and beta are the values entered in the app; Solver is not automatically run inside Excel.",
        "VaR/ES formulas on this workbook recalculate when opened in desktop Excel. Openpyxl does not store Excel's calculated formula cache.",
    ]
    for r, txt in enumerate(notes, start=14):
        ws.cell(r, 1, f"Note {r-13}").font = Font(bold=True)
        ws.cell(r, 2, txt)
    autosize(ws, max_width=85)

    # Common scenario loss sheet
    ws = wb.create_sheet("Scenario Losses")
    put_headers(ws, ["Scenario", "Year", "Previous Risk Factor", "Risk Factor", "Return", "Scenario Estimate", "Loss"])
    n = len(df) - 1
    for i in range(n):
        r = i + 2
        data_r_prev = i + 2
        data_r_now = i + 3
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, f"='Data'!A{data_r_now}")
        ws.cell(r, 3, f"='Data'!B{data_r_prev}")
        ws.cell(r, 4, f"='Data'!B{data_r_now}")
        ws.cell(r, 5, f"=D{r}/C{r}-1")
        ws.cell(r, 6, f"='Data'!$B${len(df)+1}*D{r}/C{r}")
        ws.cell(r, 7, f"='Inputs & Notes'!$B$2-F{r}")
    add_plot_to_sheet(ws, figs["Scenario Losses"], "I2")
    note(ws, f"A{n+4}", "Scenario formula", "Last observed risk factor × current / previous risk factor")
    note(ws, f"A{n+5}", "Loss formula", "Target − scenario estimate")
    autosize(ws)

    # Parametric Normal method, fitted only to positive historical losses
    ws = wb.create_sheet("Parametric estimation")
    put_headers(ws, ["Scenario", "Historical loss", "Positive loss only"])
    for i in range(n):
        r = i + 2
        sr = i + 2
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, f"='Scenario Losses'!G{sr}")
        ws.cell(r, 3, f'=IF(B{r}>0,B{r},0)')
    for cell, val in [("E2", "Positive-loss mean"), ("E3", "Positive-loss sample SD"), ("E4", "Confidence alpha"), ("E5", "Normal z-quantile"), ("E6", "VaR"), ("E7", "ES")]:
        ws[cell] = val
        ws[cell].font = Font(bold=True)
    ws["F2"] = f"=SUM(C2:C{n+1})/COUNTIF(C2:C{n+1},\">0\")"
    ws["F3"] = f"=SQRT(SUMPRODUCT((C2:C{n+1}>0)*(C2:C{n+1}-F2)^2)/(COUNTIF(C2:C{n+1},\">0\")-1))"
    ws["F4"] = "='Inputs & Notes'!$B$3"
    ws["F5"] = "=NORMINV(F4,0,1)"
    ws["F6"] = "=F2+F5*F3"
    ws["F7"] = "=F2+F3*NORMDIST(F5,0,1,FALSE)/(1-F4)"
    autosize(ws)

    # Equally weighted historical method
    ws = wb.create_sheet("Equally weighted")
    put_headers(ws, ["Scenario", "Loss"])
    for i in range(n):
        r = i + 2
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, f"='Scenario Losses'!G{r}")
    note(ws, "D2", "VaR", f"=PERCENTILE(B2:B{n+1},'Inputs & Notes'!$B$3)")
    note(ws, "D3", "ES", f'=IFERROR(AVERAGEIF(B2:B{n+1},">"&E2,B2:B{n+1}),E2)')
    # Correct the output cells created by note() to keep labels in D and formulas in E.
    ws["E2"] = f"=PERCENTILE(B2:B{n+1},'Inputs & Notes'!$B$3)"
    ws["E3"] = f'=IFERROR(AVERAGEIF(B2:B{n+1},">"&E2,B2:B{n+1}),E2)'
    autosize(ws)

    # Exponentially weighted historical method: rows sorted by loss descending,
    # but each weight remains tied to its chronological scenario index.
    ws = wb.create_sheet("Expo weighted")
    put_headers(ws, ["Scenario", "Loss", "Weight", "Cumulative tail weight", "Tail weight used for ES"])
    scen_losses = model["scenarios"]["Loss"].to_numpy(dtype=float)
    order_desc = np.argsort(scen_losses)[::-1]
    for idx, scen_idx in enumerate(order_desc):
        r = idx + 2
        chronological_row = int(scen_idx) + 2
        ws.cell(r, 1, int(scen_idx) + 1)
        ws.cell(r, 2, f"='Scenario Losses'!G{chronological_row}")
        ws.cell(r, 3, f"=((1-'Inputs & Notes'!$B$4)*'Inputs & Notes'!$B$4^(COUNTA(A$2:A${n+1})-A{r}))/(1-'Inputs & Notes'!$B$4^COUNTA(A$2:A${n+1}))")
        ws.cell(r, 4, f"=SUM($C$2:C{r})")
        ws.cell(r, 5, f"=MAX(0,MIN(C{r},1-'Inputs & Notes'!$B$3-(D{r}-C{r})))")
    note(ws, "G2", "VaR", f"=INDEX(B2:B{n+1},MATCH(TRUE,INDEX(D2:D{n+1}>=1-'Inputs & Notes'!$B$3,0),0))")
    note(ws, "G3", "ES", f"=SUMPRODUCT(B2:B{n+1},E2:E{n+1})/(1-'Inputs & Notes'!$B$3)")
    ws["H2"] = f"=INDEX(B2:B{n+1},MATCH(TRUE,INDEX(D2:D{n+1}>=1-'Inputs & Notes'!$B$3,0),0))"
    ws["H3"] = f"=SUMPRODUCT(B2:B{n+1},E2:E{n+1})/(1-'Inputs & Notes'!$B$3)"
    add_plot_to_sheet(ws, figs["Expo Weighted"], "J2")
    autosize(ws)

    # Data Vol Est: formula-based EWMA recursion and scenario adjustment
    ws = wb.create_sheet("Data Vol Est")
    put_headers(ws, ["Scenario", "Year", "Return", "Variance before scenario", "Volatility", "Final volatility", "Vol ratio", "Scenario estimate", "Adjusted loss"])
    for i in range(n):
        r = i + 2
        dr0, dr1 = i + 2, i + 3
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, f"='Data'!A{dr1}")
        ws.cell(r, 3, f"='Data'!B{dr1}/'Data'!B{dr0}-1")
        ws.cell(r, 4, f"=VAR(C$2:C${n+1})" if i == 0 else f"='Inputs & Notes'!$B$4*D{r-1}+(1-'Inputs & Notes'!$B$4)*C{r-1}^2")
        ws.cell(r, 5, f"=SQRT(D{r})")
        ws.cell(r, 6, f"=SQRT('Inputs & Notes'!$B$4*D${n+1}+(1-'Inputs & Notes'!$B$4)*C${n+1}^2)")
        ws.cell(r, 7, f"=F{r}/E{r}")
        ws.cell(r, 8, f"='Data'!$B${n+2}*('Data'!B{dr0}+('Data'!B{dr1}-'Data'!B{dr0})*G{r})/'Data'!B{dr0}")
        ws.cell(r, 9, f"='Inputs & Notes'!$B$2-H{r}")
    note(ws, "K2", "VaR", f"=PERCENTILE(I2:I{n+1},'Inputs & Notes'!$B$3)")
    note(ws, "K3", "ES", f'=IFERROR(AVERAGEIF(I2:I{n+1},">"&L2,I2:I{n+1}),L2)')
    ws["L2"] = f"=PERCENTILE(I2:I{n+1},'Inputs & Notes'!$B$3)"
    ws["L3"] = f'=IFERROR(AVERAGEIF(I2:I{n+1},">"&L2,I2:I{n+1}),L2)'
    add_plot_to_sheet(ws, figs["Data Vol Est"], "N2")
    autosize(ws)

    # Losses adjusted by SD: formulas match the workbook's one-step lag.
    ws = wb.create_sheet("Losses adj SD")
    put_headers(ws, ["Scenario", "Loss", "Variance before scenario", "SD", "SD ratio", "Adjusted loss"])
    for i in range(n):
        r = i + 2
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, f"='Scenario Losses'!G{r}")
        ws.cell(r, 3, f"=VAR(B$2:B${n+1})" if i == 0 else f"='Inputs & Notes'!$B$4*C{r-1}+(1-'Inputs & Notes'!$B$4)*B{r-1}^2")
        ws.cell(r, 4, f"=SQRT(C{r})")
        ws.cell(r, 5, f"=SQRT('Inputs & Notes'!$B$4*C${n+1}+(1-'Inputs & Notes'!$B$4)*B${n+1}^2)/D{r}")
        ws.cell(r, 6, f"=B{r}*E{r}")
    note(ws, "H2", "VaR", f"=PERCENTILE(F2:F{n+1},'Inputs & Notes'!$B$3)")
    note(ws, "H3", "ES", f'=IFERROR(AVERAGEIF(F2:F{n+1},">"&I2,F2:F{n+1}),I2)')
    ws["I2"] = f"=PERCENTILE(F2:F{n+1},'Inputs & Notes'!$B$3)"
    ws["I3"] = f'=IFERROR(AVERAGEIF(F2:F{n+1},">"&I2,F2:F{n+1}),I2)'
    add_plot_to_sheet(ws, figs["Losses Adj SD"], "K2")
    autosize(ws)

    # EVT / POT. The shape and scale cells in Inputs & Notes are Solver's changing cells.
    ws = wb.create_sheet("EVT")
    for cell, val in [("A2", "Threshold u"), ("A3", "GPD shape xi"), ("A4", "GPD scale beta"),
                      ("A5", "Number of losses n"), ("A6", "Number of exceedances n_u"),
                      ("A7", "Log-likelihood objective"), ("A8", "Confidence alpha"),
                      ("A9", "VaR"), ("A10", "ES")]:
        ws[cell] = val
        ws[cell].font = Font(bold=True)
    ws["B2"] = "='Inputs & Notes'!$B$5"
    ws["B3"] = "='Inputs & Notes'!$B$6"
    ws["B4"] = "='Inputs & Notes'!$B$7"
    ws["B5"] = f"=COUNT(B14:B{n+13})"
    ws["B6"] = f'=COUNTIF(B14:B{n+13},">"&B2)'
    ws["B7"] = f"=SUM(D14:D{n+13})"
    ws["B8"] = "='Inputs & Notes'!$B$3"
    ws["B9"] = "=IF(ABS(B3)<1E-8,B2+B4*LN((B6/B5)/(1-B8)),B2+(B4/B3)*(((B6/B5)/(1-B8))^B3-1))"
    ws["B10"] = "=IF(B3<1,(B9+B4-B3*B2)/(1-B3),NA())"
    put_headers(ws, ["Scenario", "Loss", "Excess over threshold", "Log-likelihood contribution"], row=13)
    for i in range(n):
        r = i + 14
        sr = i + 2
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, f"='Scenario Losses'!G{sr}")
        ws.cell(r, 3, f"=IF(B{r}>$B$2,B{r}-$B$2,0)")
        ws.cell(r, 4, f'=IF(C{r}>0,IF(1+$B$3*C{r}/$B$4>0,IF(ABS($B$3)<1E-8,-LN($B$4)-C{r}/$B$4,-LN($B$4)-(1/$B$3+1)*LN(1+$B$3*C{r}/$B$4)),-1E+99),0)')
    ws["F2"] = "Solver setup"
    ws["F2"].font = Font(bold=True)
    ws["F3"] = "Set objective: B7"
    ws["F4"] = "To: Max"
    ws["F5"] = "By changing: 'Inputs & Notes'!B6:B7"
    ws["F6"] = "Constraints: beta > 0; xi < 1; keep GPD support valid"
    ws["F7"] = "The app estimates MLE in Python when MLE is selected; Solver is optional for re-optimizing inside Excel."
    add_plot_to_sheet(ws, figs["EVT"], "F10")
    autosize(ws, max_width=45)

    # Lognormal Monte Carlo: positive losses -> log parameters -> formula simulation.
    ws = wb.create_sheet("Ln-Monte Carlo")
    put_headers(ws, ["Scenario", "Historical loss", "Positive loss", "Natural log of positive loss"])
    for i in range(n):
        r = i + 2
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, f"='Scenario Losses'!G{r}")
        ws.cell(r, 3, f'=IF(B{r}>0,B{r},0)')
        ws.cell(r, 4, f'=IF(C{r}>0,LN(C{r}),0)')
    for cell, val in [("F2", "Lognormal parameter"), ("F3", "mu = mean(log positive losses)"),
                      ("F4", "sigma = sample SD(log positive losses)"), ("F5", "VaR"), ("F6", "ES")]:
        ws[cell] = val
        ws[cell].font = Font(bold=True)
    ws["G3"] = f"=SUM(D2:D{n+1})/COUNTIF(C2:C{n+1},\">0\")"
    ws["G4"] = f"=SQRT(SUMPRODUCT((C2:C{n+1}>0)*(D2:D{n+1}-$G$3)^2)/(COUNTIF(C2:C{n+1},\">0\")-1))"
    sim_header = n + 4
    put_headers(ws, ["Simulation", "Uniform U (seeded value)", "Simulated lognormal loss"], row=sim_header)
    sim_first = sim_header + 1
    sim_last = sim_header + int(simulations)
    uniforms = model["methods"]["Ln-Monte Carlo"]["uniforms"]
    for i, u in enumerate(uniforms):
        r = sim_first + i
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, float(u))
        ws.cell(r, 3, f"=EXP($G$3+$G$4*NORMINV(B{r},0,1))")
    ws["G5"] = f"=PERCENTILE(C{sim_first}:C{sim_last},'Inputs & Notes'!$B$3)"
    ws["G6"] = f'=IFERROR(AVERAGEIF(C{sim_first}:C{sim_last},">"&G5,C{sim_first}:C{sim_last}),G5)'
    add_plot_to_sheet(ws, figs["Ln-Monte Carlo"], "I2")
    autosize(ws)

    # EVT Monte Carlo calculations and simulation details.
    ws = wb.create_sheet("EVT-Monte Carlo")
    sim_first = 4
    sim_last = sim_first + int(simulations) - 1
    evt_mc = model["methods"]["EVT-Monte Carlo"]

    put_headers(ws, ["Simulation", "U(0,1)", "Simulated GPD excess", "Adjusted loss", "Excess loss above VaR"], row=3)
    for i in range(int(simulations)):
        r = sim_first + i
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, float(evt_mc["uniforms"][i]))
        ws.cell(r, 3, f'=IF(ABS($H$6)<1E-8,-$H$5*LN(B{r}),($H$5/$H$6)*(B{r}^(-$H$6)-1))')
        ws.cell(r, 4, f'=IF(C{r}>=$H$4,0,C{r})')
        ws.cell(r, 5, f'=IF(D{r}>$H$2,D{r},0)')

    # Results and method-of-moments parameters.
    for cell, val in [
        ("G2", "VaR (mean adjusted loss)"),
        ("G3", "ES (mean losses above VaR)"),
        ("G4", "Percentile cutoff"),
        ("G5", "GPD scale beta (MoM)"),
        ("G6", "GPD shape xi (MoM)"),
        ("G7", "Mean excess"),
        ("G8", "Sample SD of excess"),
        ("G9", "Threshold u"),
        ("G10", "Number of positive historical losses"),
        ("G11", "Confidence alpha"),
    ]:
        ws[cell] = val
        ws[cell].font = Font(bold=True)

    # Rank positive historical losses in descending order and calculate excesses.
    positive_count = len(evt_mc["positive_losses"])
    hist_start = 14
    hist_end = hist_start + positive_count - 1
    ws["G13"] = "Positive historical loss (descending)"
    ws["H13"] = "Excess y = MAX(0, loss-u)"
    ws["G13"].font = Font(bold=True)
    ws["H13"].font = Font(bold=True)
    scenario_last = len(model["scenarios"]) + 1
    for i in range(positive_count):
        r = hist_start + i
        rank = i + 1
        ws.cell(r, 7, f"=LARGE('Scenario Losses'!$G$2:$G${scenario_last},{rank})")
        ws.cell(r, 8, f"=MAX(0,G{r}-$H$9)")

    ws["H7"] = f"=AVERAGE(H{hist_start}:H{hist_end})"
    ws["H8"] = f"=STDEV(H{hist_start}:H{hist_end})"
    ws["H5"] = "=2*H8^2/H7"
    ws["H6"] = "=1-H7/H5"
    ws["H9"] = "='Inputs & Notes'!$B$5"
    ws["H10"] = positive_count
    ws["H11"] = "='Inputs & Notes'!$B$3"
    ws["H4"] = f"=PERCENTILE(C{sim_first}:C{sim_last},H11)"
    ws["H2"] = f"=AVERAGE(D{sim_first}:D{sim_last})"
    ws["H3"] = f'=IFERROR(AVERAGEIF(E{sim_first}:E{sim_last},">0"),0)'

    # Keep the chart clear of the calculation table.
    add_plot_to_sheet(ws, figs["EVT-Monte Carlo"], "J2")
    autosize(ws, max_width=42)

    # Summary is deliberately the LAST worksheet. Its values are formulas linked to each method sheet.
    ws = wb.create_sheet("Synthesis")
    put_headers(ws, ["Approach", "Model", "Value-at-Risk", "Expected Shortfall"])
    entries = [
        ("Deterministic", "Parametric", "='Parametric estimation'!$F$6", "='Parametric estimation'!$F$7"),
        ("Deterministic", "Equally weighted", "='Equally weighted'!$E$2", "='Equally weighted'!$E$3"),
        ("Deterministic", "Expo weighted", "='Expo weighted'!$H$2", "='Expo weighted'!$H$3"),
        ("Deterministic", "Data Vol Est", "='Data Vol Est'!$L$2", "='Data Vol Est'!$L$3"),
        ("Deterministic", "Losses adj SD", "='Losses adj SD'!$I$2", "='Losses adj SD'!$I$3"),
        ("Deterministic", "EVT", "=EVT!$B$9", "=EVT!$B$10"),
        ("Stochastic", "Ln-Monte Carlo", "='Ln-Monte Carlo'!$G$5", "='Ln-Monte Carlo'!$G$6"),
        ("Stochastic", "EVT-Monte Carlo", "='EVT-Monte Carlo'!$H$2", "='EVT-Monte Carlo'!$H$3"),
    ]
    for r, entry in enumerate(entries, start=2):
        for c, value in enumerate(entry, start=1):
            ws.cell(r, c, value)
    ws["A10"] = "Average"
    ws["B10"] = "All eight methods"
    ws["C10"] = "=AVERAGE(C2:C9)"
    ws["D10"] = "=AVERAGE(D2:D9)"
    add_plot_to_sheet(ws, figs["Summary"], "A13")
    autosize(ws)

    for sheet in wb.worksheets:
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "A2"
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, (int, float)) and cell.value is not None:
                    cell.number_format = '#,##0.00;[Red](#,##0.00);-'
        # Keep the table header visible and readable.
        if sheet.max_row >= 1:
            sheet.row_dimensions[1].height = 24
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return out.getvalue()


# ---------------------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------------------
st.title("Insurance VaR & Expected Shortfall")

st.write(
    "Upload your Excel workbook, select the sheet and columns containing Year and Risk Factor, "
    "set the parameters, then calculate the eight methods and download an Excel workbook. "
    "Historical returns are applied to the last observed risk-factor level; the target is used separately to define losses."
)

uploaded = st.file_uploader("Upload Excel workbook (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Upload your Excel workbook to get started.")
    st.stop()

try:
    frames = read_workbook(uploaded)
except Exception as exc:
    st.error(f"Could not read this workbook: {exc}")
    st.stop()

sheet_names = list(frames.keys())
default_sheet_index = sheet_names.index("Data") if "Data" in sheet_names else 0
selected_sheet = st.selectbox("Worksheet containing historical data", sheet_names, index=default_sheet_index)
raw = frames[selected_sheet]
if raw.shape[1] < 2:
    st.error("The selected worksheet needs at least two columns.")
    st.stop()

col_names = [str(c) for c in raw.columns]
year_guess = next((i for i, c in enumerate(col_names) if "year" in c.lower() or "date" in c.lower()), 0)
risk_guess = next((i for i, c in enumerate(col_names) if any(k in c.lower() for k in ["risk", "fin_resul", "value", "index", "amount"])), 1)
if risk_guess == year_guess:
    risk_guess = 1 if year_guess == 0 else 0

c1, c2 = st.columns(2)
with c1:
    year_col = st.selectbox("Year / date column", col_names, index=year_guess)
with c2:
    risk_col = st.selectbox("Risk factor column", col_names, index=risk_guess)

try:
    df = clean_input_frame(raw, year_col, risk_col)
except Exception as exc:
    st.error(f"Input data validation failed: {exc}")
    st.stop()

st.subheader("Historical data preview")
st.dataframe(df, use_container_width=True, hide_index=True)
st.caption(f"{len(df)} observations · {len(df)-1} return/loss scenarios")

with st.sidebar:
    st.header("Model parameters")
    target = st.number_input(
        "Target (same units as Risk Factor)",
        min_value=0.000001,
        value=None,
        placeholder="Enter target",
        format="%.6f",
        help="Enter the target value to use for the loss calculations.",
    )
    confidence = st.slider("Confidence level (alpha)", min_value=0.90, max_value=0.999, value=DEFAULT_CONFIDENCE, step=0.001, format="%.3f")
    lam = st.number_input("Exponential decay lambda", min_value=0.01, max_value=0.999, value=DEFAULT_LAMBDA, step=0.01, format="%.3f")
    st.subheader("POT / GPD parameters")
    threshold = st.number_input("Threshold u", min_value=0.0, value=DEFAULT_THRESHOLD, step=100_000_000.0, format="%.2f")
    estimate_gpd_mle = st.checkbox("Estimate GPD parameters by maximum likelihood (recommended)", value=True,
                                   help="Fits xi and beta to losses above the threshold. Turn this off to enter xi and beta manually.")
    shape = st.number_input("GPD shape xi (manual mode)", min_value=-0.49, max_value=0.99, value=DEFAULT_GPD_SHAPE, step=0.01, format="%.3f", disabled=estimate_gpd_mle)
    scale = st.number_input("GPD scale beta (manual mode)", min_value=1.0, value=DEFAULT_GPD_SCALE, step=100_000_000.0, format="%.2f", disabled=estimate_gpd_mle)
    st.subheader("Monte Carlo")
    simulations = st.number_input("Number of simulations", min_value=100, max_value=1_000_000, value=DEFAULT_SIMULATIONS, step=1000)
    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=DEFAULT_SEED, step=1)

st.info(
    "Enter the target value manually before running the calculations. "
    "MLE mode estimates GPD shape and scale. Turn it off to enter the GPD parameters manually."
)

if st.button("Calculate VaR & ES", type="primary", use_container_width=True):
    if target is None or float(target) <= 0:
        st.error("Please enter a positive target before calculating.")
        st.stop()
    try:
        model = run_model(df, float(target), float(confidence), float(lam), float(threshold),
                          float(shape), float(scale), int(simulations), int(seed), bool(estimate_gpd_mle))
        figs = create_plots(df, model, float(target), float(confidence))
        st.session_state["risk_model_output"] = {
            "model": model, "df": df.copy(), "target": float(target),
            "confidence": float(confidence), "lam": float(lam), "threshold": float(threshold),
            "shape": float(shape), "scale": float(scale), "simulations": int(simulations),
            "seed": int(seed), "figs": figs, "estimate_gpd_mle": bool(estimate_gpd_mle),
        }
    except Exception as exc:
        st.error(f"Calculation failed: {exc}")
        st.stop()

if "risk_model_output" in st.session_state:
    out = st.session_state["risk_model_output"]
    model = out["model"]
    figs = out["figs"]
    st.subheader("Summary of results")
    formatted = model["summary"].copy()
    for col in ["Value-at-Risk", "Expected Shortfall"]:
        formatted[col] = formatted[col].map(lambda x: f"{x:,.2f}")
    st.dataframe(formatted, use_container_width=True, hide_index=True)

    st.subheader("Graphs")
    graph_order = [
        ("Data", "Risk factor over time and target"),
        ("Scenario Losses", "Historical loss distribution"),
        ("Expo Weighted", "Exponential weights"),
        ("Data Vol Est", "Return volatility adjustment ratio"),
        ("Losses Adj SD", "Loss SD adjustment ratio"),
        ("EVT", "Positive losses and POT/GPD tail"),
        ("Ln-Monte Carlo", "Lognormal Monte Carlo"),
        ("EVT-Monte Carlo", "EVT Monte Carlo"),
        ("Summary", "VaR / ES comparison"),
    ]
    for i in range(0, len(graph_order), 2):
        left, right = st.columns(2)
        for container, item in zip([left, right], graph_order[i:i+2]):
            with container:
                st.markdown(f"**{item[1]}**")
                st.pyplot(figs[item[0]], clear_figure=False, use_container_width=True)

    excel_bytes = build_excel(
        model, out["df"], out["target"], out["confidence"], out["lam"], out["threshold"],
        model["gpd_shape"], model["gpd_scale"], out["simulations"], out["seed"], figs
    )
    st.download_button(
        "Download results workbook (.xlsx)",
        data=excel_bytes,
        file_name="Insurance_VaR_ES_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.caption(
        "Excel export contains real formulas in calculation cells, seeded simulation inputs, and a formula-driven summary sheet at the end. "
        "The EVT worksheet includes the GPD log-likelihood and Solver instructions. The app fits MLE parameters in Python when selected; Solver is optional inside Excel."
    )
