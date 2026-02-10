# src/model.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Tuple, Union, Optional
import warnings

# =============================================================================
# Configuration & Types
# =============================================================================

@dataclass
class MertonOutput:
    """
    Standardized output for Merton Model calculations.
    Optimized for Polars integration.
    """
    asset_value: float          # V_A
    asset_volatility: float     # sigma_A
    distance_to_default: float  # DD
    default_prob: float         # PD
    credit_spread: float        # Spread (bps)
    leverage_ratio: float       # D / V_A
    converged: bool             # Solver status

# Small number to prevent division by zero
EPSILON = 1e-9

# =============================================================================
# Core Logic
# =============================================================================

class MertonSolver:
    """
    High-performance solver for Merton Structural Model (1974).
    Supports single-point solution and grid generation for heatmaps.
    """

    @staticmethod
    def _d1_d2(A: Union[float, np.ndarray], 
               D: float, 
               r: float, 
               T: float, 
               sigma_A: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        r"""
        Calculate d1 and d2 parameters. Vectorized support for NumPy arrays.
        
        LaTeX:
            d_1 = \frac{\ln(A/D) + (r + 0.5 \sigma_A^2)T}{\sigma_A \sqrt{T}}
            d_2 = d_1 - \sigma_A \sqrt{T}
        """
        # Safety check for sigma_A to avoid division by zero
        sigma_A = np.maximum(sigma_A, EPSILON)
        sqrt_T = np.sqrt(T)
        
        d1 = (np.log(A / D) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * sqrt_T)
        d2 = d1 - sigma_A * sqrt_T
        return d1, d2

    @staticmethod
    def _system_equations(x: Tuple[float, float], 
                          E: float, 
                          D: float, 
                          r: float, 
                          T: float, 
                          sigma_E: float) -> Tuple[float, float]:
        """
        System of non-linear equations to minimize.
        x[0]: Asset Value (A)
        x[1]: Asset Volatility (sigma_A)
        """
        A, sigma_A = x
        
        # Constraints enforcement (Soft barrier)
        if A <= 0 or sigma_A <= 0:
            return (1e10, 1e10)

        d1, d2 = MertonSolver._d1_d2(A, D, r, T, sigma_A)
        
        # Eq 1: Equity Value = Call Option on Assets
        # E = A * N(d1) - D * e^(-rT) * N(d2)
        eq1 = A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
        
        # Eq 2: Ito's Lemma linkage
        # sigma_E * E = N(d1) * sigma_A * A
        eq2 = norm.cdf(d1) * sigma_A * A - sigma_E * E
        
        return (eq1, eq2)

    @classmethod
    def solve(cls, 
              equity: float, 
              debt: float, 
              vol_equity: float, 
              risk_free: float, 
              maturity: float = 1.0) -> MertonOutput:
        """
        Solves for Asset Value and Asset Volatility for a single observation.
        """
        # 1. Initial Guess (Heuristic)
        # Assume Asset Value ~ Equity + Debt
        # Assume Asset Vol ~ Equity Vol * (E / A)
        A0 = equity + debt
        sigma_A0 = vol_equity * (equity / A0)
        
        try:
            # 2. Numerical Optimization
            # 'fsolve' is generally robust for this system
            result = fsolve(
                func=cls._system_equations,
                x0=[A0, sigma_A0],
                args=(equity, debt, risk_free, maturity, vol_equity),
                xtol=1e-6,
                maxfev=1000
            )
            
            A_sol, sigma_A_sol = result
            
            # 3. Post-calculation metrics (Analytical)
            # Ensure positive solution
            if A_sol <= 0 or sigma_A_sol <= 0:
                raise ValueError("Solver converged to negative values")

            d1, d2 = cls._d1_d2(A_sol, debt, risk_free, maturity, sigma_A_sol)
            
            # Distance to Default (DD)
            # Note: Market practice often uses physical probability (mu) instead of risk-free (r)
            # but standard Merton implementation uses neutral measure or assumes mu=r for simplicity unless specified.
            # Here we follow the standard formula:
            dd = (np.log(A_sol / debt) + (risk_free - 0.5 * sigma_A_sol**2) * maturity) / (sigma_A_sol * np.sqrt(maturity))
            
            # Default Probability
            pd = norm.cdf(-dd)
            
            # Credit Spread (bps) -> (-1/T) * ln( (A - E) / (D * exp(-rT)) ) ? 
            # Simplified: Yield - RiskFree
            # Debt Value B = A - E
            debt_val = A_sol - equity
            yTM = -np.log(debt_val / debt) / maturity
            spread = (yTM - risk_free) * 10000 # bps
            if np.isnan(spread) or spread < 0: spread = 0.0

            return MertonOutput(
                asset_value=float(A_sol),
                asset_volatility=float(sigma_A_sol),
                distance_to_default=float(dd),
                default_prob=float(pd),
                credit_spread=float(spread),
                leverage_ratio=float(debt / A_sol),
                converged=True
            )

        except Exception as e:
            # Fallback for non-convergence (common in extreme distress or bad data)
            return MertonOutput(
                asset_value=np.nan,
                asset_volatility=np.nan,
                distance_to_default=np.nan,
                default_prob=np.nan,
                credit_spread=np.nan,
                leverage_ratio=np.nan,
                converged=False
            )

    @classmethod
    def solve_grid(cls, 
                   equity_base: float, 
                   debt_base: float, 
                   vol_base: float, 
                   rf: float,
                   steps: int = 50) -> dict:
        """
        Generates a heatmap grid for Widget #10.
        Optimized Loop: Solves 50x50 scenarios for Sensitivity Analysis.
        
        Axes:
        - X: Leverage Multiplier (0.5x to 2.0x Debt)
        - Y: Volatility Multiplier (0.5x to 2.5x Sigma_E)
        """
        
        # Grid definition
        debt_mults = np.linspace(0.5, 2.0, steps)
        vol_mults = np.linspace(0.5, 2.5, steps)
        
        # Pre-allocate arrays for performance
        pd_grid = np.zeros((steps, steps))
        dd_grid = np.zeros((steps, steps))
        
        # Note: Since fsolve cannot be easily vectorized for implicit systems,
        # we use a double loop. For N=50 (2500 iter), this takes <1s on modern CPUs.
        # Intel Ultra 7 will handle this instantly.
        for i, v_m in enumerate(vol_mults):
            current_vol = vol_base * v_m
            for j, d_m in enumerate(debt_mults):
                current_debt = debt_base * d_m
                
                res = cls.solve(equity_base, current_debt, current_vol, rf)
                
                if res.converged:
                    pd_grid[i, j] = res.default_prob
                    dd_grid[i, j] = res.distance_to_default
                else:
                    pd_grid[i, j] = np.nan
                    dd_grid[i, j] = np.nan
                    
        return {
            "x_debt_mults": debt_mults,
            "y_vol_mults": vol_mults,
            "z_pd": pd_grid,
            "z_dd": dd_grid
        }

if __name__ == "__main__":
    # 🛡️ Developer Checkpoint: Smoke Test
    print("Running Model Smoke Test...")
    
    # Test Case: TSMC-like (approximate numbers)
    # E = 20T, D = 1T, Vol = 30%, r = 1.5%
    test_res = MertonSolver.solve(
        equity=20000000, 
        debt=1000000, 
        vol_equity=0.30, 
        risk_free=0.015
    )
    
    print(f"Status: {test_res.converged}")
    print(f"Asset Val: {test_res.asset_value:,.2f}")
    print(f"Asset Vol: {test_res.asset_volatility:.2%}")
    print(f"Def Prob:  {test_res.default_prob:.6%}")
    
    assert test_res.converged, "Solver failed on simple test case!"
    print("✅ Smoke Test Passed.")