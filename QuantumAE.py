"""
Quantum Amplitude Estimation Engine

This engine implements QAE for European option pricing using classical simulation of quantum circuits. 
Circuits are simulated via statevector or shot-based methods using numpy and scipy
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings

from params import SharedParameters
from BlackScholes import bs_price


@dataclass
class QAEResult:
    """
    Output from QAE pricers 
    
    Extra fields vs MCResult
      - amplitude: raw quantum amplitude 'a' before price conversion
      - n_oracle: total oracle (Grover operator) calls (QAE equiv of n_paths)
      - n_qubits: how many qubits encode the distribution
      - noise_level: depolarising noise rate applied per oracle call
      - circuit_depth: deepest circuit used (key hardware constraint)
    """
    price: float
    amplitude: float
    stderr: float
    ci_low: float
    ci_high: float
    n_oracle: int
    n_qubits: int
    method: str
    noise_level: float
    circuit_depth: int
    
    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low
    
    def __repr__(self) -> str:
        return(
            f"QAEResult({self.method})\n"
            f"  price        = {self.price:.6f}\n"
            f"  amplitude    = {self.amplitude:.6f}\n"
            f"  stderr       = {self.stderr:.6f}\n"
            f"  95% CI       = [{self.ci_low:.6f}, {self.ci_high:.6f}]"
            f"  (width {self.ci_width:.6f})\n"
            f"  oracle calls = {self.n_oracle:,}\n"
            f"  qubits       = {self.n_qubits}\n"
            f"  max depth    = {self.circuit_depth}\n"
            f"  noise p      = {self.noise_level:.4f}"
        )
        
# -------------------------------------------------------------------------------------------------  
# State preparation: discretise log-normal onto n qubits

class LogNormalStatePrep:
    def __init__(self, params:SharedParameters, n_qubits:int=5, q:float=0.0, s_min_mul:float=0.01, s_max_mul:float=4.0):
        self.params = params
        self.n_qubits = n_qubits
        self.q = q
        self.N = 2**n_qubits        # 5 qubits would give 32 price bins
        
        S, K, r, sigma, T = params.S, params.K, params.r, params.sigma, params.T
        self.s_min = S * s_min_mul
        self.s_max = S * s_max_mul
        
        # Price grid: N evenly spaced points from s_min to s_max More qubits = finer grid = less discretisation error 
        # (at the cost of deeper quantum circuits and more qubits on hardware)
        self.s_grid = np.linspace(self.s_min, self.s_max, self.N)
        self.ds = self.s_grid[1] - self.s_grid[0]
        
        #Risk neutral log normal PDF
        mu_ln = np.log(S) + (r - q - 0.5 * sigma**2) * T
        sig_ln = sigma * np.sqrt(T)
        
        pdf = norm.pdf(np.log(self.s_grid), loc=mu_ln, scale=sig_ln) / self.s_grid
        pdf = np.maximum(pdf, 0)
        
        # Convert PDF to discrete probabilities via Riemann sum (multiply by bin width) then normalise
        prob = pdf * self.ds
        total = prob.sum()
        if total <= 0:
            raise ValueError
        self.prob = prob/total 
        self.amplitudes = np.sqrt(self.prob)
        assert abs(np.dot(self.amplitudes, self.amplitudes) - 1.0) < 1e-6, \
            "Amplitude vector not normalised."
            
    def payoff_vector(self, option_type:str = "call") -> np.ndarray:
        """
        Discounted payoff at each grid point normalised to [0,1]
        PPayoff Scale is stored to convert amplitude estimates back to dollar prices.

        Args:
            option_type (str, optional): Defaults to "call".
        """
        K = self.params.K
        r = self.params.r
        T = self.params.T
        discount = np.exp(-r * T)
        
        if option_type == "call":
            raw = np.maximum(self.s_grid - K, 0.0) * discount
        elif option_type == "put":
            raw = np.maximum(K - self.s_grid, 0.0) * discount
        else:
            raise ValueError("Option type must be call or put")
        
        self.payoff_scale = raw.max()
        if self.payoff_scale == 0:
            warnings.warn("All payoffs are zero. Option may be deep OTM at this discretisation")
            return np.zeros(self.N)
        return raw/self.payoff_scale
    
    def true_amplitude(self, option_type: str = "call") -> float:
        """
        Exact amplitude a = E_Q[normalised_payoff]
        Option price = a * payoff_scale
        """
        f = self.payoff_vector(option_type)
        return float(np.dot(self.prob, f))
    
    def optionPrice_fromAmplitude(self, amplitude: float) -> float:
        return float(amplitude * self.payoff_scale)
        
        
        