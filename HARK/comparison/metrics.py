"""
Economic metrics for model comparison.

This module implements standard metrics used to evaluate and compare
heterogeneous agent model solutions, following Maliar, Maliar, and Winant (2021)
and other standard references.
"""

import numpy as np
from scipy import stats
from typing import Callable, Union, List, Any
import warnings

from HARK.rewards import CRRAutility, CRRAutilityP
from HARK.distributions import Distribution


class EconomicMetrics:
    """
    Compute standard metrics for heterogeneous agent models.

    This class provides methods to calculate various error metrics and
    statistics for evaluating solution quality and comparing different
    solution methods.
    """

    def __init__(self):
        """Initialize EconomicMetrics calculator."""
        pass

    def euler_equation_error(
        self,
        consumption_func: Callable,
        params: dict,
        test_points: Union[np.ndarray, dict],
        shocks: Distribution = None,
    ) -> np.ndarray:
        """
        Compute normalized Euler equation errors following MMW (2021).

        The Euler equation error at a point is defined as the relative
        difference between current consumption and the consumption implied
        by the Euler equation.

        Parameters
        ----------
        consumption_func : callable
            Consumption policy function
        params : dict
            Model parameters including CRRA, DiscFac, Rfree, etc.
        test_points : np.ndarray or dict
            Points at which to evaluate errors
        shocks : Distribution, optional
            Income shock distribution for computing expectations

        Returns
        -------
        errors : np.ndarray
            Array of Euler equation errors (in consumption units)
        """
        # Extract parameters
        crra = params.get("CRRA", 2.0)
        disc_fac = params.get("DiscFac", 0.96)
        rfree = params.get("Rfree", 1.03)
        perm_gro_fac = params.get("PermGroFac", 1.01)

        # Handle different test point formats
        if isinstance(test_points, dict):
            m_points = test_points.get("mNrm", test_points.get("m", None))
            if m_points is None:
                raise ValueError("Test points must contain 'mNrm' or 'm' key")
        else:
            m_points = test_points

        # Ensure m_points is an array
        m_points = np.asarray(m_points).flatten()

        # Get consumption at test points
        c_now = consumption_func(m_points)

        # Compute assets
        a_now = m_points - c_now

        # For simple case without income shocks (perfect foresight)
        if shocks is None:
            # Next period resources
            m_next = (
                rfree * a_now / perm_gro_fac + 1.0
            )  # assuming normalized income = 1

            # Next period consumption
            c_next = consumption_func(m_next)

            # Marginal utility
            mu_now = CRRAutilityP(c_now, crra)
            mu_next = CRRAutilityP(c_next, crra)

            # Euler equation implied consumption
            c_euler = (disc_fac * rfree * perm_gro_fac ** (-crra) * mu_next) ** (
                -1 / crra
            )

        else:
            # With income shocks - need to integrate over shock distribution
            # This is a simplified version - full implementation would need
            # proper shock handling
            warnings.warn(
                "Full shock integration not yet implemented, using simplified version"
            )

            # Approximate with mean shock values
            if hasattr(shocks, "expected"):
                # Expected marginal utility next period
                def marginal_utility_next(shock_vals):
                    perm_shk = shock_vals.get("PermShk", 1.0)
                    tran_shk = shock_vals.get("TranShk", 1.0)

                    m_next = rfree * a_now / (perm_gro_fac * perm_shk) + tran_shk
                    c_next = consumption_func(m_next)
                    return perm_shk ** (-crra) * CRRAutilityP(c_next, crra)

                exp_mu_next = shocks.expected(marginal_utility_next)
            else:
                # Fallback to perfect foresight
                m_next = rfree * a_now / perm_gro_fac + 1.0
                c_next = consumption_func(m_next)
                exp_mu_next = CRRAutilityP(c_next, crra)

            # Euler equation implied consumption
            mu_now = CRRAutilityP(c_now, crra)
            c_euler = (disc_fac * rfree * perm_gro_fac ** (-crra) * exp_mu_next) ** (
                -1 / crra
            )

        # Compute errors (normalized by consumption)
        errors = np.abs((c_now - c_euler) / c_now)

        # Handle edge cases
        errors[c_now <= 0] = np.nan
        errors[~np.isfinite(errors)] = np.nan

        return errors

    def bellman_equation_error(
        self,
        value_func: Callable,
        policy_func: Callable,
        params: dict,
        test_points: Union[np.ndarray, dict],
    ) -> np.ndarray:
        """
        Compute Bellman equation residuals.

        The Bellman error measures how well the value function satisfies
        the Bellman equation at given points.

        Parameters
        ----------
        value_func : callable
            Value function
        policy_func : callable
            Policy function (consumption)
        params : dict
            Model parameters
        test_points : np.ndarray or dict
            Points at which to evaluate errors

        Returns
        -------
        errors : np.ndarray
            Array of Bellman equation errors
        """
        # Extract parameters
        crra = params.get("CRRA", 2.0)
        disc_fac = params.get("DiscFac", 0.96)
        rfree = params.get("Rfree", 1.03)
        perm_gro_fac = params.get("PermGroFac", 1.01)

        # Handle test points
        if isinstance(test_points, dict):
            m_points = test_points.get("mNrm", test_points.get("m", None))
            if m_points is None:
                raise ValueError("Test points must contain 'mNrm' or 'm' key")
        else:
            m_points = test_points

        m_points = np.asarray(m_points).flatten()

        # Get consumption and current value
        c_now = policy_func(m_points)
        v_now = value_func(m_points)

        # Compute utility from consumption
        u_now = CRRAutility(c_now, crra)

        # Compute continuation value (simplified - no shocks)
        a_now = m_points - c_now
        m_next = rfree * a_now / perm_gro_fac + 1.0
        v_next = value_func(m_next)

        # Bellman equation value
        v_bellman = u_now + disc_fac * perm_gro_fac ** (1 - crra) * v_next

        # Compute errors
        errors = np.abs(v_now - v_bellman)

        # Normalize by value magnitude (avoid division by zero)
        normalizer = np.maximum(np.abs(v_now), 1e-10)
        errors = errors / normalizer

        return errors

    def den_haan_marcet_statistic(
        self, simulated_data: dict, forecast_func: Callable
    ) -> float:
        """
        Compute Den Haan-Marcet R² statistic for forecast accuracy.

        This statistic measures how well the aggregate law of motion
        predicts future aggregate states.

        Parameters
        ----------
        simulated_data : dict
            Dictionary with simulated time series including aggregate capital
        forecast_func : callable
            Function that forecasts next period's aggregate state

        Returns
        -------
        r_squared : float
            R² statistic for forecast accuracy
        """
        # Extract aggregate capital series
        if "aggregate_capital" in simulated_data:
            k_series = simulated_data["aggregate_capital"]
        elif "AaggNow" in simulated_data:
            k_series = simulated_data["AaggNow"]
        elif "KtoLnow" in simulated_data:
            k_series = simulated_data["KtoLnow"]
        else:
            warnings.warn("No aggregate capital series found in simulation data")
            return np.nan

        # Get aggregate state if available
        if "Mrkv" in simulated_data:
            state_series = simulated_data["Mrkv"]
        else:
            # Assume single state
            state_series = np.zeros_like(k_series)

        # Ensure arrays
        k_series = np.asarray(k_series).flatten()
        state_series = np.asarray(state_series).flatten()

        # Remove any burn-in if series is long
        if len(k_series) > 1000:
            k_series = k_series[100:]
            state_series = state_series[100:]

        # Compute forecasts
        n_periods = len(k_series) - 1
        k_forecast = np.zeros(n_periods)
        k_actual = k_series[1:]

        for t in range(n_periods):
            try:
                # Try different forecast function signatures
                try:
                    k_forecast[t] = forecast_func(k_series[t], state_series[t])
                except:
                    k_forecast[t] = forecast_func(k_series[t])
            except:
                warnings.warn(f"Forecast function failed at period {t}")
                k_forecast[t] = np.nan

        # Remove any NaN values
        valid_mask = np.isfinite(k_forecast) & np.isfinite(k_actual)
        k_forecast = k_forecast[valid_mask]
        k_actual = k_actual[valid_mask]

        if len(k_forecast) < 10:
            warnings.warn("Too few valid forecast points for reliable R²")
            return np.nan

        # Compute R²
        ss_tot = np.sum((k_actual - np.mean(k_actual)) ** 2)
        ss_res = np.sum((k_actual - k_forecast) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        return r_squared

    def wealth_distribution_metrics(self, wealth_array: np.ndarray) -> dict:
        """
        Compute Gini coefficient, percentiles, and other distributional stats.

        Parameters
        ----------
        wealth_array : np.ndarray
            Array of wealth/asset holdings (can be 2D for panel data)

        Returns
        -------
        metrics : dict
            Dictionary with distributional statistics
        """
        # Flatten if panel data
        if wealth_array.ndim > 1:
            wealth_flat = wealth_array.flatten()
        else:
            wealth_flat = wealth_array

        # Remove NaN values
        wealth_flat = wealth_flat[np.isfinite(wealth_flat)]

        if len(wealth_flat) == 0:
            warnings.warn("No valid wealth data")
            return {
                "gini": np.nan,
                "mean_wealth": np.nan,
                "median_wealth": np.nan,
                "wealth_p90": np.nan,
                "wealth_p99": np.nan,
                "wealth_share_top10": np.nan,
                "wealth_share_top1": np.nan,
            }

        # Sort wealth
        wealth_sorted = np.sort(wealth_flat)
        n = len(wealth_sorted)

        # Compute Gini coefficient
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * wealth_sorted)) / (n * np.sum(wealth_sorted)) - (
            n + 1
        ) / n

        # Handle negative wealth
        if np.min(wealth_sorted) < 0:
            warnings.warn("Negative wealth detected, Gini may not be meaningful")

        # Compute percentiles
        percentiles = np.percentile(wealth_sorted, [10, 25, 50, 75, 90, 95, 99])

        # Compute wealth shares
        total_wealth = np.sum(wealth_sorted)
        if total_wealth > 0:
            top10_threshold = np.percentile(wealth_sorted, 90)
            top1_threshold = np.percentile(wealth_sorted, 99)

            wealth_share_top10 = (
                np.sum(wealth_sorted[wealth_sorted >= top10_threshold]) / total_wealth
            )
            wealth_share_top1 = (
                np.sum(wealth_sorted[wealth_sorted >= top1_threshold]) / total_wealth
            )
        else:
            wealth_share_top10 = np.nan
            wealth_share_top1 = np.nan

        # Create metrics dictionary
        metrics = {
            "gini": gini,
            "mean_wealth": np.mean(wealth_sorted),
            "median_wealth": percentiles[2],  # 50th percentile
            "wealth_p10": percentiles[0],
            "wealth_p25": percentiles[1],
            "wealth_p75": percentiles[3],
            "wealth_p90": percentiles[4],
            "wealth_p95": percentiles[5],
            "wealth_p99": percentiles[6],
            "wealth_share_top10": wealth_share_top10,
            "wealth_share_top1": wealth_share_top1,
            "wealth_std": np.std(wealth_sorted),
            "wealth_skewness": stats.skew(wealth_sorted),
            "wealth_kurtosis": stats.kurtosis(wealth_sorted),
        }

        return metrics

    def convergence_metric(self, history: List[Any], tolerance: float = 1e-6) -> dict:
        """
        Compute convergence metrics for iterative solution methods.

        Parameters
        ----------
        history : list
            History of solution iterations
        tolerance : float
            Convergence tolerance

        Returns
        -------
        metrics : dict
            Convergence statistics
        """
        if len(history) < 2:
            return {
                "converged": False,
                "iterations": len(history),
                "final_distance": np.nan,
            }

        # Compute distances between successive iterations
        distances = []
        for i in range(1, len(history)):
            # This is a placeholder - actual distance computation
            # depends on the type of objects in history
            if hasattr(history[i], "distance"):
                dist = history[i].distance(history[i - 1])
            else:
                # Simple numeric distance
                dist = np.abs(history[i] - history[i - 1])
            distances.append(dist)

        final_distance = distances[-1] if distances else np.nan
        converged = final_distance < tolerance

        return {
            "converged": converged,
            "iterations": len(history),
            "final_distance": final_distance,
            "distance_history": distances,
        }
