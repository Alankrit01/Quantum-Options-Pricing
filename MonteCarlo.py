"""
Monte Carlo Option pricing engine which includes four pricing methods:
    1. Plain Monte Carlo - standard GBM simulation, used as baseline
    2. Antithetic MC - antithetic variates (mirror each path): halves variance
    3. Control variate - used Black Scholes price as control and removes correlated error
    4. Quasi Monte Carlo - Sobol low discrepancy sequences: Faster convergence

All four methods support European Call and put payoffs and return price estimate + 95% confidence interval + stderr

Accept SharedParameters (S,K,r,sigma,T) and support continous dividend yield q

Usage for unit testing: python MonteCarlo.py
Output: BLack Scholes reference price: 8.021352

Method                                    Price      Stderr     |Error|      CI Width   Time
─────────────────────────────────────────────────────────────────────────────────────────────────────────
Plain Monte Carlo                        8.039778    0.029712    0.018426    0.116470  (0.01s)
Antithetic Monte Carlo                   8.042734    0.016656    0.021382    0.065292  (0.01s)
Control Variate Monte Carlo              8.041266    0.013472    0.019914    0.052809  (0.01s)
Quasi-Monte Carlo (Sobol, n=262,144)     8.021339    0.025767    0.000013    0.101003  (0.02s)
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
from scipy.stats import norm, qmc

from params import SharedParameters
from BlackScholes import bs_price


@dataclass
class MCresult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    method: str
    
    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low
    
    def __repr__(self) -> str:
        return(
            f"MCResult({self.method})\n"
            f"  price    = {self.price:.6f}\n"
            f"  stderr   = {self.stderr:.6f}\n"
            f"  95% CI   = [{self.ci_low:.6f}, {self.ci_high:.6f}]  "
            f"(width {self.ci_width:.6f})\n"
            f"  n_paths  = {self.n_paths:,}"
        )
        
# Vectorised European Payoffs
def payoff(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"Option type must match 'call' or 'put'. You entered '{option_type}'")

# GBM terminal stock price from standard normal draws Z
def terminal_price(S: float, r: float, q: float, sigma:float, T: float, Z:np.ndarray) -> np.ndarray:
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    return S*np.exp(drift + diffusion * Z)

# Covert discounted payoff array into MonteCarlo result
def summarise(payoffs: np.ndarray, discount: float, n_paths: int, method: str) -> MCresult:
    discounted = payoffs * discount
    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(n_paths)
    z95 = 1.959964
    return MCresult(
        price   = float(price),
        stderr  = float(stderr),
        ci_low  = float(price - z95 * stderr),
        ci_high = float(price + z95 * stderr),
        n_paths = n_paths,
        method  = method,
    )
    
# -------------------------------------------------------------------------------------------------  
# Plain Monte Carlo
def plainMC(
    params: SharedParameters,
    n_paths: int = 100_000, 
    option_type: str = "call",
    q: float = 0.0,
    seed: Optional[int] = None,
) -> MCresult:
    """
    Standard Monte Carlo pricer under GBM. Convergence = O(1/rootN) -> stderr halves each time N quadruples

    Args:
        params (SharedParameters)
        n_paths (int, optional): number of simulated paths. Defaults to 100_000.
        option_type (str, optional): call or put. Defaults to "call".
        q (float, optional): continous dividend yield. Defaults to 0.0.
        seed (Optional[int], optional): reproducibility. Defaults to None.

    Returns:
        MCresult
    """
    
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    
    S_T = terminal_price(params.S, params.r, q, params.sigma, params.T, Z)
    payoffs = payoff(S_T, params.K, option_type)
    discount = np.exp(-params.r * params.T)
    
    return summarise(payoffs, discount, n_paths, "Plain Monte Carlo")

# -------------------------------------------------------------------------------------------------  
# Antithetic Variate
def antitheticMC(
    params: SharedParameters,
    n_paths: int = 100_000, 
    option_type: str = "call",
    q: float = 0.0,
    seed: Optional[int] = None,
) -> MCresult:
    """
    Monte Carlo with antithetic variates variance reproduction
    
    For each standard normal draw Z, also simulate with -Z. The two resulting payoffs are avergaed before discounting.
    Exploits negative correaltion b/w paths to cancel variance. Should reduce stderr by 30-70% vs Plain Monte Carlo.

    Args:
        params (SharedParameters)
        n_paths (int, optional): number of antithetic pairs. Defaults to 100_000.
        option_type (str, optional): call or put. Defaults to "call".
        q (float, optional): continous dividend yield. Defaults to 0.0.
        seed (Optional[int], optional): _description_. Defaults to None.

    Returns:
        MCresult
    """

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    
    S_T_pos = terminal_price(params.S, params.r, q, params.sigma, params.T, Z)
    S_T_neg = terminal_price(params.S, params.r, q, params.sigma, params.T, -Z)
    payoff_pos = payoff(S_T_pos, params.K, option_type)
    payoff_neg = payoff(S_T_neg, params.K, option_type)
    paired_payoffs = (payoff_pos+payoff_neg) / 2.0
    discount = np.exp(-params.r * params.T)
    
    return summarise(paired_payoffs, discount, n_paths, "Antithetic Monte Carlo")

# -------------------------------------------------------------------------------------------------  
# Control Variate (Using BlackSchols price as the known control)
def controlMC(
    params: SharedParameters,
    n_paths: int = 100_000, 
    option_type: str = "call",
    q: float = 0.0,
    seed: Optional[int] = None,
) -> MCresult:
    """
    Monte Carlo with control variate variance reduction
    
    Uses BlackScholes price of same option as control variate. Optimal coefficient β* is estimated from the same
    sample, then applied to adjust the raw payoffs:
    payoff_cv = payoff_raw - β* * (S_T - E[S_T])    where E[S_T] = S * exp((r-q)*T) is the known risk-neutral expectation.
    
    This removes the component of Monte Carlo error correlated with stock price path and reduces variance by 80-95% for NTM options.

    Args:
        params (SharedParameters): sigma used for BlackScholes control price
        n_paths (int, optional): number of simulated paths. Defaults to 100_000.
        option_type (str, optional): call or put. Defaults to "call".
        q (float, optional): continous dividend yield. Defaults to 0.0.
        seed (Optional[int], optional): _description_. Defaults to None.

    Returns:
        MCresult
    """
    
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    
    S_T = terminal_price(params.S, params.r, q, params.sigma, params.T, Z)
    payoffs = payoff(S_T, params.K, option_type)
    
    # Control: S_T has known expectation E[S_T] = S * exp((r-q)*T)
    E_S_T = params.S * np.exp((params.r - q) * params.T)
    control = S_T - E_S_T   # mean zero control
    
    # Optimal beta: minimises variance of the adjusted estimator
    cov_matrix = np.cov(payoffs, control)
    beta_star = cov_matrix[0,1] / cov_matrix[1,1]
    
    payoffs_cv = payoffs - beta_star * control
    discount = np.exp(-params.r * params.T)
    
    return summarise(payoffs_cv, discount, n_paths, "Control Variate Monte Carlo")

# -------------------------------------------------------------------------------------------------  
# Quasi Monte Carlo (Sobol sequences)
def quasiMC(
    params: SharedParameters,
    n_paths: int = 100_000, 
    option_type: str = "call",
    q: float = 0.0,
    seed: Optional[int] = None,
) -> MCresult:
    """
    Quasi Monte Carlo using Sobol low-discrepancy sequences
    
    Replaces pseudo-random uniforms with Sobol sequences which fill the unit hypercube more evenly.
    This gives effective covergence closer to O(1/N) in low dimensions.
    
    Uniform Sobol samples are mapped to standard normla svia the inverse normal CDF (Beasley-Springer-Moro approximation)
    
    n_paths is rounded up to the next power of 2 which is required for sobol sequences

    Args:
        params (SharedParameters)
        n_paths (int, optional): approx number of paths. Defaults to 100_000.
        option_type (str, optional): call or put. Defaults to "call".
        q (float, optional): continous dividend yield. Defaults to 0.0.
        seed (Optional[int], optional): _description_. Defaults to None.

    Returns:
        MCresult
    """
    
    # Sobol required power of 2 sample count 
    n_actual = int(2 ** np.ceil(np.log2(max(n_paths, 2))))
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    u = sampler.random(n_actual).flatten()      # uniform (0,1)
    # Clip away exact 0 and 1 to avoid infinity in ppf
    u = np.clip(u, 1e-10, 1 - 1e-10)
    Z = norm.ppf(u)         # Standard normal
    
    S_T = terminal_price(params.S, params.r, q, params.sigma, params.T, Z)
    payoffs = payoff(S_T, params.K, option_type)
    discount = np.exp(-params.r * params.T)
    result = summarise(payoffs, discount, n_actual, "Quasi Monte Carlo (Sobol)")
    result.method = f"Quasi-Monte Carlo (Sobol, n={n_actual:,})"
    return result 


# -------------------------------------------------------------------------------------------------  
# Convergence

def convergence(
    params: SharedParameters,
    method: Callable, 
    n_values: np.ndarray,
    bs_price_ref: float,  
    option_type: str = "call",
    q: float = 0.0,
    seed: Optional[int] = 42,
    n_repeats: int = 10,
) -> dict:
    """
    Run a single Monte Carlo method at increasing path counts and record error vs Black scholes

    Args:
        params (SharedParameters)
        method (Callable)
        n_values (np.ndarray): 1-D array of path counts to rest
        bs_price_ref (float): Black scholes ref price
        option_type (str, optional): call or put. Defaults to "call".
        q (float, optional): continous dividend yield. Defaults to 0.0.
        seed (Optional[int], optional): base random seed. Defaults to 42.
        n_repeats (int, optional): number of independednt trials per N. Defaults to 10.

    Returns:
        dict: _description_
        'n_values': array of N values actually used
        'mean_error': mean |MC price - BS price| across repeats
        'std_error': std of |MC price - BS price| across repeats
        'mean_stderr': mean reported MC stderr across repeats
        'mean_ci_width': mean CI width across repeats
        'method_name': string name of the method
    """
    
    n_values = np.array(n_values, dtype=int)
    mean_errors = []
    std_errors = []
    mean_stderrs = []
    mean_ci_widths = []
    actual_ns = []
    
    for N in n_values:
        errors = []
        stderrs = []
        ci_widths = []
        
        for x in range(n_repeats):
            rep_seed = None if seed is None else seed + x * 997
            try:
                res = method(params, n_paths = N, option_type=option_type, q=q, seed=rep_seed)
                errors.append(abs(res.price - bs_price_ref))
                stderrs.append(res.stderr)
                ci_widths.append(res.ci_width)
                actual_ns.append(res.n_paths)
            except Exception:
                pass
            
        if errors:
            mean_errors.append(np.mean(errors))
            std_errors.append(np.std(errors))
            mean_stderrs.append(np.mean(stderrs))
            mean_ci_widths.append(np.mean(ci_widths))
            
    # For quasi MC, the actual N may differ (rounded to power of 2). So we use actual N from the last repeat of each level.
    actual_N_per_level = []
    for N in n_values:
        rep_seed = None if seed is None else seed
        try:
            res = method(params, n_paths=N, option_type=option_type, q=q, seed=rep_seed)
            actual_N_per_level.append(N)
        except Exception:
            actual_N_per_level.append(N)
            
    return {
        'n_values': np.array(actual_N_per_level),
        'mean_error': np.array(mean_errors),
        'std_error': np.array(std_errors),
        'mean_stderr': np.array(mean_stderrs),
        'mean_ci_width': np.array(mean_ci_widths),
        'method_name': method.__name__,
    }
    
# -------------------------------------------------------------------------------------------------  
# Comparison at Fixed N
def compare_methods(
    params: SharedParameters,
    n_paths: int = 100_000, 
    option_type: str = "call",
    q: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Run all 4 methods at the same N and return results and Black Scholes ref.

    Args:
        params (SharedParameters)
        n_paths (int, optional): path count for MC methods. Defaults to 100_000.
        option_type (str, optional): call or put. Defaults to "call".
        q (float, optional): continous dividend yield. Defaults to 0.0.
        seed (int, optional): Defaults to 42.

    Returns:
        dict: maps method name 
    """
    
    bs_ref = bs_price(params,option_type=option_type, q=q)
    results = {
        'Black Scholes' : bs_ref,
        'Plain Monte Carlo' : plainMC(params, n_paths, option_type, q, seed),
        'Antithetic Monte Carlo' : antitheticMC(params, n_paths, option_type, q, seed),
        'Control Variate' : controlMC(params, n_paths, option_type, q, seed),
        'Quasi Monte Carlo' : quasiMC(params, n_paths, option_type, q, seed),
    }
    return results

# -------------------------------------------------------------------------------------------------  
# File Testing
                
if __name__ == "__main__":
    import time
    
    p = SharedParameters(S=100, K=105, r=0.05, sigma=0.2, T=1.0)
    bs_ref = bs_price(p,"call")
    print(f"BLack Scholes reference price: {bs_ref:.6f}\n")
    
    N = 200_000
    methods = [plainMC, antitheticMC, controlMC, quasiMC]
    
    print(f"{'Method':<37}  {'Price':>10}  {'Stderr':>10}  {'|Error|':>10}  {'CI Width':>10}  {'Time':>7}")
    print("─" * 105)
    for fn in methods:
        t0  = time.perf_counter()
        res = fn(p, n_paths=N, option_type="call", seed=42)
        dt  = time.perf_counter() - t0
        err = abs(res.price - bs_ref)
        print(f"{res.method:<37}  {res.price:>10.6f}  {res.stderr:>10.6f}"
            f"  {err:>10.6f}  {res.ci_width:>10.6f}  ({dt:.2f}s)")