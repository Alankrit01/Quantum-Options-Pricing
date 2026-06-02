from dataclasses import dataclass, field
from typing import Union

Number = Union[int, float]


@dataclass
class SharedParameters:
    """Shared parameters for classical and quantum finance code.

    Fields
    - S: Spot price (positive) [current price of the underlying asset]
    - K: Strike price (positive) [price at which the option can be exercised]
    - r: Risk-free rate (as decimal, e.g. 0.01 for 1%) [annualized interest rate for a risk-free investment]
    - sigma: Volatility (σ), non-negative [annualized standard deviation of the underlying asset's returns]
    - T: Time to maturity in years (positive) [time until the option expires, expressed in years]
    """

    S: Number
    K: Number
    r: Number
    sigma: Number
    T: Number
    q: Number = field(default=0.0)  # Dividend yield (as decimal, e.g. 0.02 for 2%)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate parameter values and raise ValueError with clear messages.

        Raises
        ------
        ValueError: If one or more parameters are invalid. The error message
            lists all problems found for easier debugging.
        """

        errors = []

        if not isinstance(self.S, (int, float)):
            errors.append("S (spot price) must be a number.")
        elif self.S <= 0:
            errors.append("S (spot price) must be > 0.")

        if not isinstance(self.K, (int, float)):
            errors.append("K (strike price) must be a number.")
        elif self.K <= 0:
            errors.append("K (strike price) must be > 0.")

        if not isinstance(self.r, (int, float)):
            errors.append("r (risk-free rate) must be a number (e.g. 0.01 for 1%).")

        if not isinstance(self.sigma, (int, float)):
            errors.append("σ (volatility) must be a number.")
        elif self.sigma < 0:
            errors.append("σ (volatility) must be >= 0.")

        if not isinstance(self.T, (int, float)):
            errors.append("T (time to maturity) must be a number (in years).")
        elif self.T <= 0:
            errors.append("T (time to maturity) must be > 0 (years).")

        if errors:
            msg = "Invalid SharedParameters:\n - " + "\n - ".join(errors)
            raise ValueError(msg)
