"""
Quantum Amplitude Estimation Engine

This engine implements QAE for European option pricing using classical simulation of quantum circuits. 
Circuits are simulated via statevector or shot-based methods using numpy and scipy

3 QAE Algorithms to run:

1. Classical QAE -  QPE based QAE (deep circuits with many qubits). Simulated using exact amplitude formula.
                    Used as a theoretical baseline because it is not practical on real hardware

2. Iterative QAE - Shallow circuits with O(1/epsilon) query complexity. Practical for near future hardware (Main algo)

3. Maximum Likelihood QAE - Runs fixed depth circuits at multiple schedule points and optimises the likelihood.
                            Balances depth and accuracy
                            
Usage for unit test - python QuantumAE.py
Ouput: BS reference : 8.021352

Discretisation error vs qubit count:
  qubits   points    disc_price    error $   error%
       3        8     10.687400   2.666048  33.237%
       4       16      7.487350   0.534002   6.657%
       5       32      7.873546   0.147807   1.843%
       6       64      8.048935   0.027583   0.344%
       7      128      8.014634   0.006718   0.084%
       8      256      8.023160   0.001808   0.023%

Method                      Price     |Error|     Oracle    Depth
─────────────────────────────────────────────────────────────────
Classical QAE             8.201738    0.180386        256     256  (0.00s)
IQAE                      0.018706    8.002647    226,000     157  (0.00s)
MLQAE                    88.536121   80.514769     13,300     65  (0.00s)

Noise impact on IQAE (epsilon=0.01):
   noise_p       price     |error|
    0.0000    0.026550    7.994802
    0.0010    0.094960    7.926392
    0.0050    0.022923    7.998429
    0.0100    0.049254    7.972098
    0.0500    2.718269    5.303084
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
        
# -------------------------------------------------------------------------------------------------  
# Depolarising noise model

def appy_noise(amplitude:float, n_oracle:int, noise_p:float) -> float:
    """
    Approximate depolarising noise effect on amplitude estimate.
    
    After M oracle calls with per-call error rate p, the amplitude is damped toward 0.5 (the fully mixed state):
 
        a_noisy = (1 - 2p)^M * a_ideal + (1 - (1-2p)^M) * 0.5
    """
    if noise_p <= 0:
        return amplitude
    decay = (1.0 - 2.0*noise_p) ** n_oracle
    return float(decay * amplitude + (1.0 - decay) * 0.5)

# -------------------------------------------------------------------------------------------------  
# Classical QAE - Quantum Phase Estimation (QPE)

def classicalQAE(
    params: SharedParameters,
    n_qubits: int = 5,
    m_eval: int = 6,
    option_type: str = "call",
    q: float = 0.0, 
    noise_p: float = 0.0,
) -> QAEResult:
    """
    Uses m_eval evaluation qubits. Precision: epsilon ~ pi / 2^(m_eval+1).
    Oracle calls: 2^m_eval. Circuit depth: 2^m_eval Grover layers.
    
    Included as the theoretical baseline demonstrating O(1/epsilon) scaling.
    Impractical on NISQ hardware due to deep circuits.
    
    Args:
        params (SharedParameters): _description_
        n_qubits (int, optional): _description_. Defaults to 5.
        m_eval (int, optional): _description_. Defaults to 6.
        option_type (str, optional): _description_. Defaults to "call".
        q (float, optional): _description_. Defaults to 0.0.
        noise_p (float, optional): _description_. Defaults to 0.0.
    """
    
    prep = LogNormalStatePrep(params, n_qubits, q)
    a_true = prep.true_amplitude(option_type)
    
    M = 2 ** m_eval
    n_oracle = M
    a_noisy = appy_noise(a_true, n_oracle, noise_p)     # hardware noise
    
    # QPE measures theta = arcsin(sqrt(a)) rounded to the nearest grid point.
    # The measurement grid has M evenly spaced theta values: k*pi/M for k=0..M/2
    theta_true = np.arcsin(np.sqrt(np.clip(a_noisy, 0, 1)))     # true theta in [0, pi/2]
    k_best = int(round(theta_true * M/ np.pi))      # nearest grid index
    k_best = max(0, min(k_best, M // 2))            # clip to valid range
    theta_est = np.pi * k_best / M                  # Quantised theta estimate
    a_est = np.sin(theta_est) ** 2                  # converting back to amplitude
    
    # Precision: half the QPE bin width in theta space
    # Propagate through a = sin^2(theta) using the chain rule: da/dtheta = 2*sin(theta)*cos(theta) = sin(2*theta)
    theta_stderr = np.pi / (2.0 * M)
    a_stderr = abs(2 * np.sin(theta_est) * np.cos(theta_est)) * theta_stderr
    
    price_est = prep.optionPrice_fromAmplitude(a_est)
    price_stderr = prep.payoff_scale * a_stderr     # scale amplitude uncertainity to price
    z95 = 1.959964       # 1.96 sigma for 95 CI
    
    return QAEResult(
        price         = price_est,
        amplitude     = float(a_est),
        stderr        = price_stderr,
        ci_low        = price_est - z95 * price_stderr,
        ci_high       = price_est + z95 * price_stderr,
        n_oracle      = int(n_oracle),
        n_qubits      = n_qubits,
        method        = "Classical QAE (QPE)",
        noise_level   = noise_p,
        circuit_depth = int(M),   # deep (this is the problem with classical QAE)
    )
  
# -------------------------------------------------------------------------------------------------  
# Iterative QAE 
def iterativeQAE(
    params: SharedParameters,
    epsilon: float = 0.01,      # target amplitude precision (e.g. 0.01 = 1%)
    alpha: float = 0.05,        # failure probability (0.05 -> 95% CI)
    n_qubits: int = 5,
    option_type: str = "call",
    q: float = 0.0, 
    noise_p: float = 0.0,
    max_iter: int = 50,
    seed: int = 42,
) -> QAEResult:
    """
    Classical QAE needs a huge elevation register and deep circuits upfront. 
    IQAE:
        - maintains a confidence interval [theta_lo, theta_hi] for theta = arcsin(sqrt(a))
        - Each round: picks a Grover depth to maximally shrink the interval
        - Runs shots at that depth, use Chernoff-Hoeffding to get a new interval bound
        - Intersect with the running CI to narrow it further
        - Stop when the CI width < 2*epsilon
        
    This is better than Classical because 
        - Each circuit is shallow (depth chosen adaptively, not exponentially)
        - Total oracle cost is still O(1/epsilon) — same as classical QAE
        - Max circuit depth is O(1/epsilon) not O(1/epsilon) simultaneous gates
        - This is NISQ-friendly: short circuits, many repetitions
        
    If the current CI width is 'w' in theta space, we can use Grover depth k where k ~ pi/(4w). 
    Deeper circuits amplify the signal more, shrinking the CI faster but only if noise doesn't overwhelm the signal first.
    
    Args:
        params (SharedParameters): _description_
        epsilon (float, optional): _description_. Defaults to 0.01.
        option_type (str, optional): _description_. Defaults to "call".
        q (float, optional): _description_. Defaults to 0.0.
        noise_p (float, optional): _description_. Defaults to 0.0.
        max_iter (int, optional): _description_. Defaults to 50.
        seed (int, optional): _description_. Defaults to 42.
    """
    prep = LogNormalStatePrep(params, n_qubits, q)
    a_true = prep.true_amplitude(option_type)
    rng = np.random.default_rng(seed)
    
    # Number of shots per round is derived from Chernoff bound to ensure that CI shrinks reliably across max_iter rounds.
    # The factor 0.1 keeps it manageable (more shots per round = fewer rounds needed)
    N_shots = max(10, min(1000, int(np.ceil(
        np.log(2 * max_iter / alpha) / (2 * epsilon**2) * 0.1))))
    
    # Initialise CI to full theta range [0,pi/2]
    # (theta = arcsin(sqrt(a)), so a in [0,1] maps to theta in [0, pi/2])
    theta_lo = 0.0
    theta_hi = np.pi / 2.0
    total_oracle = 0    # cumulative oracel calls across all rounds
    max_depth = 1       # track deepest circuit used
    history = []        # record convergences for plotting
    
    for x in range(max_iter):
        width = theta_hi - theta_lo
        
        # Stopping criteria: CI is already narrow
        if width < 2.0 * epsilon:
            break
        
        # Depth selection: pick k so that 2k+1 Grover layers fit within the CI
        # Intuition: the measurement probability oscillates with period ~1/depth in theta space. 
        # For useful signal, the period should be larger than the CI. 
        k = max(0, int(np.floor(np.pi / (4.0 * width))))
        k = min(k, 4096)
        depth = 2*k+1   # total Grover layers (always odd: k forward, then k back)
        max_depth = max(max_depth, depth)
        total_oracle += N_shots * depth     # each shot uses 'depth' oracle calls
        
        # simulate noise (hardware degradation)
        a_noisy = appy_noise(a_true, depth, noise_p)
        theta_n = np.arcsin(np.sqrt(np.clip(a_noisy, 0.0, 1.0)))
        
        # Measurement probability: After k Grover iterations, the probability of measuring '1' (good state) is:
        #   p = sin^2(depth * theta)  when k is even
        #   p = cos^2(depth * theta)  when k is odd
        # This oscillatory structure is what allows QAE to amplify precision
        if k%2 == 0:
            p_meas = np.sin(depth * theta_n) ** 2
        else:
            p_meas = np.cos(depth * theta_n) ** 2
        p_meas = float(np.clip(p_meas, 0.0, 1.0))
        
        # Simulate shot counts: binomial draw
        n_ones = int(rng.binomial(N_shots, p_meas))
        
        # Chernoff-Hoeffding Bound: Classicial statistical CI on measured probability 
        # t is half width of CI on p_meas given N_shots samples
        # The log(2*max_iter/alpha) term is a union bound across all rounds
        t = np.sqrt(np.log(2.0 * max_iter/alpha) / (2.0 * N_shots))
        p_lo = max(0.0, n_ones / N_shots-t)
        p_hi = min(1.0, n_ones / N_shots+t)
        
        # Invert to get theta bounds from p bounds
        if k%2 == 0:
            th_lo_c = np.arcsin(np.sqrt(p_lo)) / depth      # lower theta from lower p
            th_hi_c = np.arcsin(np.sqrt(p_hi)) / depth      # higher theta from higher p
        else:
            # cos^2 is decreasing: high p -> low theta, so bounds flip
            th_lo_c = np.arccos(np.sqrt(p_hi)) / depth
            th_hi_c = np.arccos(np.sqrt(p_lo)) / depth
            
        # intersect: new CI = old CI and this rounds CI
        theta_lo = max(theta_lo, th_lo_c)
        theta_hi = min(theta_hi, th_hi_c)
        
        # Numerical guard if intersection is empty
        if theta_lo >= theta_hi:
            mid = (th_lo_c + th_hi_c) / 2.0
            theta_lo = max(0.0, mid - epsilon)
            theta_hi = min(np.pi / 2, mid + epsilon)
            
        history.append({
            'iter': x, 'k': k, 'depth': depth,
            'ci_width': theta_hi - theta_lo,
            'n_oracle_cumulative': total_oracle,
        })
        
    # Final Estimate: midpoint of converged theta CI
    theta_est = (theta_lo + theta_hi) / 2.0
    a_est = float(np.sin(theta_est) ** 2)       # converting theta to amplitude
        
    # Convert amplitude interval to price interval
    price_est = prep.optionPrice_fromAmplitude(a_est)
    price_lo = prep.optionPrice_fromAmplitude(np.sin(theta_lo)**2)
    price_hi = prep.optionPrice_fromAmplitude(np.sin(theta_hi)**2)
    # Infer stderr from CI width
    price_stderr = (price_hi - price_lo) / (2.0 * 1.959964)
        
    result = QAEResult(
        price         = price_est,
        amplitude     = a_est,
        stderr        = price_stderr,
        ci_low        = price_lo,
        ci_high       = price_hi,
        n_oracle      = int(total_oracle),
        n_qubits      = n_qubits,
        method        = "IQAE",
        noise_level   = noise_p,
        circuit_depth = int(max_depth),
    )
    result._history = history   # attach convergence trace for debugging/plotting
    return result

# -------------------------------------------------------------------------------------------------  
# Maximum Likelihood QAE 
def mlQAE(
    params: SharedParameters,
    schedule: Optional[List[int]] = None,  # list of Grover repetition counts
    n_shots_each: int = 100,   # shots per schedule depth
    n_qubits: int = 5,
    option_type: str = "call",
    q: float = 0.0,
    noise_p: float = 0.0,
    seed:int = 42,
) -> QAEResult:
    """
    IQAE adaptively picks the next depth based on the current CI. MLQAE uses a fixed schedule of depths decided upfront, 
    runs all of them in parallel, then combines the shot counts via maximum likelihood.
    
    Why MLQAE over IQAE:
        - No adaptive loop: all circuits can run in parallel on hardware
        - Cleaner statistical framework: MLE with Cramer-Rao CI bounds
        - More robust: no risk of CI collapse from bad round choices
    
    Why MLQAE over Classical QAE:
        - No deep QPE circuits (no QFT needed)
        - Each schedule point is a short, independent circuit
    
    MLE:
    For each schedule point m_k (Grover depth 2*m_k+1), we observe n_ones out of n_shots. 
    The measurement probability is a known function of theta:
        p(theta, m_k) = sin^2((2*m_k+1)*theta)  [for even m_k]
        
    The joint log-likelihood across all schedule points is:
        LL(theta) = sum_k [ n_ones_k * log(p_k) + (n_shots - n_ones_k) * log(1-p_k) ]
        
    We find theta_mle = argmax LL(theta) using scipy's bounded scalar minimiser.

    Args:
        params (SharedParameters): _description_
        schedule (Optional[List[int]], optional): _description_. Defaults to None.
        option_type (str, optional): _description_. Defaults to "call".
        q (float, optional): _description_. Defaults to 0.0.
        noise_p (float, optional): _description_. Defaults to 0.0.
        seed (int, optional): _description_. Defaults to 42.
    """
    prep = LogNormalStatePrep(params, n_qubits, q)
    a_true = prep.true_amplitude(option_type)
    rng = np.random.default_rng(seed)
    
    # Default schedule: geometrically increasing depths(2^n) covers wide range of amplification levels with few circuits. 
    # Deeper schedules give more precision at the cost of circuit depth.
    if schedule is None:
        schedule = [0,1,2,4,8,16,32]
    shot_data = []      # list of (m_k, n_shots, n_ones) tuples
    total_oracle = 0
    max_depth = 1
    
    for m_k in schedule:
        depth = 2 * m_k + 1         # actual Grover layers for this schedule point
        max_depth = max(max_depth, depth)
        total_oracle += n_shots_each * depth
        
        # apply noise at current depth level
        a_noisy = appy_noise(a_true, depth, noise_p)
        theta_n = np.arcsin(np.sqrt(np.clip(a_noisy, 0.0, 1.0)))
        
        # Measurement prob at current depth 
        if m_k%2 == 0:
            p_meas = np.sin(depth * theta_n) ** 2
        else:
            p_meas = np.cos(depth * theta_n) ** 2
        p_meas = float(np.clip(p_meas, 1e-10, 1.0 - 1e-10))     # Avoid log(0)
        
        # Simulate hardware measurement (binomial draw)
        n_ones = int(rng.binomial(n_shots_each, p_meas))
        shot_data.append((m_k, n_shots_each, n_ones))
        
    # Maximum Likelihood Optimisation
    # Build the negative log-likelihood function over theta in (0, pi/2)
    def negative_log_likelihood(theta: float) -> float:
        if theta <= 0 or theta >= np.pi / 2:
            return 1e12     # large penalty if outside range
        ll = 0.0
        for m_k, n_total, n_ones in shot_data:
            depth = 2 * m_k + 1
            # Measreuemnt prob at this theta value
            p = np.sin(depth * theta)**2 if m_k%2==0 else np.cos(depth*theta)**2
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
            # Standard binomial log-likelihood contribution
            ll += n_ones * np.log(p) + (n_total - n_ones) * np.log(1.0 - p)
        return -ll   # minimise negative = maximise likelihood
    
    # scipy's bounded Brent method: finds the global minimum on (0, pi/2)
    opt = minimize_scalar(
        negative_log_likelihood,
        bounds=(1e-6, np.pi / 2.0 - 1e-6),
        method='bounded',
        options={'xatol': 1e-8},    # stop when theta changes by less than 1e-8
    )
    theta_mle = opt.x
    a_est = float(np.sin(theta_mle)**2)     # MLE Amplitude estimate
    
    # Crammer-Rao confidence Interval
    # The Fisher information I(theta) = -d^2 LL/dtheta^2 tells us how sharply the likelihood peaks at theta_mle. 
    # By the Cramer-Rao bound, the MLE variance is >= 1/I(theta_mle), with equality asymptotically.
    # We compute I numerically via finite differences on the negative log-likelihood.
    
    dth = 1e-5
    fi = max(1e-12,
              (negative_log_likelihood(theta_mle - dth)
               - 2 * negative_log_likelihood(theta_mle)
               + negative_log_likelihood(theta_mle + dth)) / dth**2)
    # fi estimates the curvature; 1/sqrt(fi) is the MLE standard error on theta
    theta_stderr = 1.0 / np.sqrt(fi)
    
    # propogate theta uncertainity to price uncertainity
    a_stderr = abs(2 * np.sin(theta_mle) * np.cos(theta_mle)) * theta_stderr
 
    # Scale amplitude uncertainty to price uncertainty
    price_est = prep.optionPrice_fromAmplitude(a_est)
    price_stderr = prep.payoff_scale*a_stderr
    z95 = 1.959964
    
    return QAEResult(
        price         = price_est,
        amplitude     = float(a_est),
        stderr        = price_stderr,
        ci_low        = price_est - z95 * price_stderr,
        ci_high       = price_est + z95 * price_stderr,
        n_oracle      = int(total_oracle),
        n_qubits      = n_qubits,
        method        = "MLQAE",
        noise_level   = noise_p,
        circuit_depth = int(max_depth),
    )
    
# -------------------------------------------------------------------------------------------------  
# Convergence study -> how error decreases as epsilon shrinks (oracle budget grows)
def qaeConvergence(
    params: SharedParameters,
    epsilon_values: np.ndarray,   # e.g. [0.1, 0.05, 0.02, 0.01, 0.005]
    bs_price_ref: float,        # ground truth to measure error against
    algorithm: str = "iqae",
    n_qubits: int = 5,
    option_type: str = "call",
    q: float = 0.0,
    noise_p: float = 0.0,
    n_repeats: int = 10,   # independent runs per epsilon for error bars
) -> dict:
    """
    Run IQAE or MLQAE at decreasing epsilon values and record:
        Mean |errpr| vs BS reference
        std of error across repeats 
        mean stderr from algorithm
        mean oracle count (x-axis for convergence plot)
        
    The output format matches that of MonteCarlo with a key comparison being 
        MC shows O(1/sqrtN) slope vs O(1/N) for QAE
    """
    mean_errors = []
    std_error = []
    mean_stderrs = []
    oracle_counts = []
    
    for eps in epsilon_values:
        errors = []
        stderrs = []
        oracles = []
        
        for rep in range(n_repeats):
            try:
                if algorithm == "iqae":
                    res = iterativeQAE(
                        params, epsilon=float(eps), n_qubits=n_qubits, option_type=option_type, q=q, noise_p=noise_p, 
                        seed=42 + rep*997,
                    )
                else:
                    # For MLQAE derive a sensible schedule from epsilon
                    max_m = max(1, int(np.ceil(np.pi/(4.0*float(eps)))))
                    sched = [0] + [2**i for i in range(int(np.log2(max_m + 1)) +1)]
                    res = mlQAE(
                        params, schedule=sched, n_shots_each=max(50,int(1.0/float(eps))), n_qubits=n_qubits, 
                        option_type=option_type, q=q, noise_p=noise_p, seed=42 + rep*997,
                    )
                errors.append(abs(res.price - bs_price_ref))
                stderrs.append(res.stderr)
                oracles.append(res.n_oracle)
            except Exception as e:
                warnings.warn(f"QAE failed at eps={eps:.4f} rep={rep}: {e}")
                
        if errors:
            mean_errors.append(float(np.mean(errors)))
            std_error.append(float(np.std(errors)))
            mean_stderrs.append(float(np.mean(stderrs)))
            oracle_counts.append(float(np.mean(oracles)))
    
    n = len(mean_errors)
    return {
        'epsilon_values': np.array(epsilon_values[:n]),
        'n_oracle'      : np.array(oracle_counts),    # x-axis: oracle budget used
        'mean_error'    : np.array(mean_errors),      # y-axis: accuracy achieved
        'std_error'     : np.array(std_error),       # error bar width
        'mean_stderr'   : np.array(mean_stderrs),
        'method_name'   : algorithm.upper(),
        'noise_p'       : noise_p,
    }
    
# -------------------------------------------------------------------------------------------------  
# Noise impact study -> at what noise level does QAE lose its advantage
def noise_impact_study(
    params: SharedParameters,
    noise_levels: np.ndarray,   # e.g. np.linspace(0, 0.05, 20)
    epsilon: float = 0.01,
    n_qubits: int = 5,
    option_type: str = "call",
    q: float = 0.0,
    n_repeats: int = 10,
) -> dict:
    """
    Run IQAE at increasing depolaring noise levels with fixed epsilon.
    
    QAE uses deep Grover circuits. Each gate adds noise and at some noise_p, error from noise exceeds what plain MC would give
    with same oracle budget. Beyond this threshold, Monte Carlo is better.

    Typically, QAE holds advantage upto noise_p = 10^-3 to 10^-4 per gate for realistic circuit depths
    """
    bs_ref = bs_price(params, option_type=option_type, q=q)
    mean_errors = []
    std_errors = []
    mean_prices = []
    
    for p_noise in noise_levels:
        errors = []
        prices = []
        for rep in range(n_repeats):
            try:
                res = iterativeQAE(
                    params, epsilon=epsilon, n_qubits=n_qubits, option_type=option_type, q=q, noise_p=float(p_noise),
                    seed = 42 + rep*997,
                )
                errors.append(abs(res.price - bs_ref))
                prices.append(res.price)
            except Exception:
                pass
        if errors:
            mean_errors.append(float(np.mean(errors)))
            std_errors.append(float(np.std(errors)))
            mean_prices.append(float(np.mean(prices)))
            
    n = len(mean_errors)
    return {
        'noise_levels': np.array(noise_levels[:n]),
        'mean_error'  : np.array(mean_errors),
        'std_error'   : np.array(std_errors),
        'mean_price'  : np.array(mean_prices),
        'bs_ref'      : bs_ref,   # reference line for the plot
    }

# -------------------------------------------------------------------------------------------------  
# Discretisation Quality Check -> how many qubits are needed to accurately capture the distribution
def discretisation_report(
    params: SharedParameters,
    n_qubits: int = 5,
    q: float = 0.0,
    option_type: str = "call",
) -> dict:
    """
    Quantify the error introduced by using a finite qubit grid vs the continuous log-normal distribution that BS prices exactly.
    
    Trade Off:
        n=3  qubits: 8 price bins, large discretisation error, tiny circuits
        n=5  qubits: 32 price bins, moderate error, manageable circuits
        n=8  qubits: 256 price bins, small error, deep circuits
        n=10 qubits: 1024 bins, very accurate, but circuit size grows exponentially

    Typical rule of thumb: use enough qubits so disc_error < epsilon/10.
    Args:
        params (SharedParameters): _description_
        n_qubits (int, optional): _description_. Defaults to 5.
        q (float, optional): _description_. Defaults to 0.0.
        option_type (str, optional): _description_. Defaults to "call".
    """
    prep = LogNormalStatePrep(params, n_qubits, q)
    a_true = prep.true_amplitude(option_type)
    disc_price = prep.optionPrice_fromAmplitude(a_true)         # Price from discretised grid
    bs_ref = bs_price(params, option_type, q)                   # Continous black scholes ref.
    
    return {
        'n_qubits'          : n_qubits,
        'n_points'          : prep.N,                              # 2^n_qubits
        'bs_price'          : bs_ref,
        'discretised_price' : disc_price,
        'disc_error'        : abs(disc_price - bs_ref),            # absolute $ error
        'disc_error_pct'    : abs(disc_price - bs_ref) / bs_ref * 100,  # % error
        'true_amplitude'    : a_true,
        'payoff_scale'      : prep.payoff_scale,
    }
    
# -------------------------------------------------------------------------------------------------  
# File Testing

if __name__ == "__main__":
    import time
 
    # Standard test: $100 stock, $105 strike, 5% rate, 20% vol, 1 year
    # BS call price should be ~8.02 (matches MonteCarlo.py test case)
    p      = SharedParameters(S=100, K=105, r=0.05, sigma=0.2, T=1.0)
    bs_ref = bs_price(p, "call")
    print(f"BS reference : {bs_ref:.6f}\n")
 
    # STEP 1: Check discretisation error for increasing qubit counts
    # More qubits = better accuracy but deeper quantum circuits
    print("Discretisation error vs qubit count:")
    print(f"  {'qubits':>6}  {'points':>7}  {'disc_price':>12}  {'error $':>9}  {'error%':>7}")
    for nq in [3, 4, 5, 6, 7, 8]:
        rep = discretisation_report(p, nq)
        print(f"  {nq:>6}  {rep['n_points']:>7}  {rep['discretised_price']:>12.6f}"
              f"  {rep['disc_error']:>9.6f}  {rep['disc_error_pct']:>6.3f}%")
 
    # STEP 2: Compare all three QAE algorithms on the same test
    print()
    tests = [
        ("Classical QAE", lambda: classicalQAE(p, n_qubits=5, m_eval=8)),
        ("IQAE",          lambda: iterativeQAE(p, epsilon=0.005, n_qubits=5)),
        ("MLQAE",         lambda: mlQAE(p, n_qubits=5)),
    ]
    print(f"{'Method':<22}  {'Price':>10}  {'|Error|':>10}  {'Oracle':>9}  {'Depth':>7}")
    print("─" * 65)
    for name, fn in tests:
        t0  = time.perf_counter()
        res = fn()
        dt  = time.perf_counter() - t0
        print(f"{name:<22}  {res.price:>10.6f}  {abs(res.price-bs_ref):>10.6f}"
              f"  {res.n_oracle:>9,}  {res.circuit_depth:>7}  ({dt:.2f}s)")
 
    # STEP 3: Show noise sensitivity — how does error grow as hardware gets noisier?
    # noise_p=0.0 is ideal; noise_p=0.05 is very noisy (current NISQ hardware)
    print("\nNoise impact on IQAE (epsilon=0.01):")
    print(f"  {'noise_p':>8}  {'price':>10}  {'|error|':>10}")
    for np_val in [0.0, 0.001, 0.005, 0.01, 0.05]:
        res = iterativeQAE(p, epsilon=0.01, noise_p=np_val)
        print(f"  {np_val:>8.4f}  {res.price:>10.6f}  {abs(res.price-bs_ref):>10.6f}")