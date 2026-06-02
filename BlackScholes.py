"""
Black Scholes option pricing engine
Usage for file check: python BlackScholes.py

    Provides:
        - European call and put prices
        - Full Greeks: Delta, Gamma, Theta, Vega, Rho
        - Dividend yield support (continuous, via cost-of-carry)
        - Implied volatility via Newton-Raphson
"""
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from params import SharedParameters
from copy import copy

def d1_d2(S: float, K: float, r: float, sigma: float, T: float, q:float = 0.0):
    """
    Compute d1 and d2 for BlackScholes

    Args:
        S (float): spot price
        K (float): strike price
        r (float): risk free rate (annualised, continous)
        sigma (float): volatility (annualised)
        T (float): time to maturity (years)
        q (float, optional): continous dividend yield. Defaults to 0.0.
    """
    sqrtT = np.sqrt(T)
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2

def bs_price(params: SharedParameters, option_type: str = "call", q:float = 0.0,) -> float:
    """
    Black Scholes European Option price

    Args:
        params (SharedParameters)
        option_type (str, optional): call or put. Defaults to "call".
        q (float, optional): continous dividend yield. Defaults to 0.0.
        
    Returns:
        float: option price
    """
    S, K, r, sigma, T = params.S, params.K, params.r, params.sigma, params.T
    d1, d2 = d1_d2(S, K, r, sigma, T, q)
    discount = np.exp(-r*T)
    forward_S = S*np.exp(-q*T)
    
    if option_type == "call":
        return forward_S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    elif option_type == "put":
        return K * discount * norm.cdf(-d2) - forward_S * norm.cdf(-d1)
    else:
        raise ValueError(f"Option type must match 'call' or 'put'. You entered '{option_type}'")
# -------------------------------------------------------------------------------------------------  
# GREEKS
@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    def __repr__(self) -> str:
        return(
            f"Greeks(\n"
            f"  delta = {self.delta:+.6f}\n"
            f"  gamma = {self.gamma:+.6f}\n"
            f"  theta = {self.theta:+.6f}  (per day)\n"
            f"  vega  = {self.vega:+.6f}  (per 1% σ)\n"
            f"  rho   = {self.rho:+.6f}  (per 1% r)\n"
            f")"
        )
        
        
def bs_greeks(params: SharedParameters, option_type: str = "call", q:float = 0.0,) -> Greeks:
    """
    Compute Black Scholes Greeks

    Args:
        params (SharedParameters)
        option_type (str, optional): call or put. Defaults to "call".
        q (float, optional): continous dividend yield. Defaults to 0.0.

    Returns:
        Greeks dataclass
    """
    S, K, r, sigma, T = params.S, params.K, params.r, params.sigma, params.T
    d1, d2 = d1_d2(S, K, r, sigma, T, q)
    
    sqrtT = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    discount = np.exp(-r*T)
    forward_S = S*np.exp(-q*T)
    
    # Gamma and Vega are same for calls and puts
    gamma = (np.exp(-q*T)*pdf_d1) / (S*sigma*sqrtT)
    vega = forward_S * pdf_d1 * sqrtT / 100     # per 1 percentage point
    
    if option_type == "call":
        delta = np.exp(-q*T)*cdf_d1
        theta = (
            -(forward_S * pdf_d1 * sigma) / (2 * sqrtT)
            - r * K * discount * cdf_d2
            + q * forward_S * cdf_d1
        ) / 365  # convert to per-day
        rho = K * T * discount * cdf_d2 / 100
        
    elif option_type == "put":
        delta = np.exp(-q*T)*(cdf_d1 - 1)
        theta = (
            -(forward_S * pdf_d1 * sigma) / (2 * sqrtT)
            + r * K * discount * norm.cdf(-d2)
            - q * forward_S * norm.cdf(-d1)
        ) / 365
        rho = -K * T * discount * norm.cdf(-d2) / 100
    else:
        raise ValueError(f"Option type must match 'call' or 'put'. You entered '{option_type}'")
    
    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
# -------------------------------------------------------------------------------------------------  
# IMPLIED VOLATILITY

def implied_volatility(market_pric: float, params: SharedParameters, option_type: str = "call", q:float = 0.0, tol: float = 1e-6, maxx_iter: int = 100,)->float:
    """
    Implied Volatility from market price computed using Newton-Rahpson

    Args:
        market_pric (float): observed market price of the option
        params (SharedParameters): SharedParameters — sigma field is ignored (it is the unknown)
        option_type (str, optional): call or put. Defaults to "call".
        q (float, optional): continous dividend yield. Defaults to 0.0.
        tol (float, optional): convergence tolerance on price error. Defaults to 1e-6.
        maxx_iter (int, optional): max Newton-Raphson iterations. Defaults to 100.

    Raises:
        ValueError: market_pric is below intrinsic value (no real IV exists)
        RuntimeError: Newton Rhapson fails to converge

    Returns:
        float: implied volatility (annualised, as a decimal)
    """
    S, K, r, T = params.S, params.K, params.r, params.T
    discount = np.exp(-r*T)
    
    # Intrinsic valye bounds check
    if option_type == "call":
        intrinsic = max(S*np.exp(-q*T)-K*discount, 0.0)
    else:
        intrinsic = max(K*discount - S*np.exp(-q*T), 0.0)
        
    if market_pric<intrinsic-tol:
        raise ValueError(
            f"market_price={market_pric:.4f} is below intrinsic value "
            f"{intrinsic:.4f}. No real implied volatility exists."
        )
        
    # Initial guess: Brenner Subrahmanyam approximation:
    sigma = np.sqrt(2*np.pi/T) * (market_pric/S)
    sigma = max(sigma, 1e-4)
    
    for x in range(maxx_iter):
        trial = copy(params)        # Temp params with current sigma guess
        object.__setattr__(trial, "sigma", sigma)
        price = bs_price(trial, option_type=option_type, q=q)
        greeks = bs_greeks(trial, option_type=option_type, q=q)
        
        vega_raw = greeks.vega*100  # Use full vega instead of scaled 
        diff = price - market_pric
        
        if abs(diff)<tol:
            return sigma
        if abs(vega_raw)<1e-12:
            raise RuntimeError(
                f"Vega is approx 0 at iteration {x}, Newton Raphson cannot contunue."
                f"Try another intiial guess or check param values"
            )
        sigma -= diff/vega_raw
        sigma = max(sigma, 1e-8)
        
    raise RuntimeError(
        f"Implied volatility did not converge after {maxx_iter} iterations. "
        f"Last sigma = {sigma:.6f}, price error = {diff:.2e}"
    )
# -------------------------------------------------------------------------------------------------  
# Price surface helpers (vectorised)

def price_vs_strike(params: SharedParameters, strikes: np.ndarray, option_type: str = "call", q: float = 0.0,) -> np.ndarray:
    """Vectorised BS prices across a range of strikes.
 
    Parameters
    ----------
    params      : SharedParameters (K field is overridden by strikes array)
    strikes     : 1-D array of strike prices
    option_type : 'call' or 'put'
    q           : continuous dividend yield
 
    Returns
    -------
    np.ndarray : option prices, same shape as strikes
    """
 
    prices = np.empty_like(strikes, dtype=float)
    for i, K in enumerate(strikes):
        trial = copy(params)
        object.__setattr__(trial, "K", K)
        prices[i] = bs_price(trial, option_type=option_type, q=q)
    return prices
 
def delta_vs_strike(
    params: SharedParameters,
    strikes: np.ndarray,
    option_type: str = "call",
    q: float = 0.0,
) -> np.ndarray:
    """Vectorised Delta across a range of strikes."""
 
    deltas = np.empty_like(strikes, dtype=float)
    for i, K in enumerate(strikes):
        trial = copy(params)
        object.__setattr__(trial, "K", K)
        deltas[i] = bs_greeks(trial, option_type=option_type, q=q).delta
    return deltas 

# -------------------------------------------------------------------------------------------------  
# File Testing

if __name__ == "__main__":
    p = SharedParameters(S=100, K=105, r=0.05, sigma=0.2, T=1.0)
 
    call_price = bs_price(p, "call")
    put_price  = bs_price(p, "put")
    call_greeks = bs_greeks(p, "call")
 
    print(f"Call price : {call_price:.4f}")
    print(f"Put price  : {put_price:.4f}")
    print(f"\nPut-call parity check:")
    print(f"  C - P          = {call_price - put_price:.4f}")
    print(f"  S - K·e^(-rT)  = {p.S - p.K * np.exp(-p.r * p.T):.4f}")
    print(f"\n{call_greeks}")
 
    # Implied vol round-trip
    iv = implied_volatility(call_price, p, "call")
    print(f"Implied vol (round-trip): {iv:.6f}  (input sigma was {p.sigma})")