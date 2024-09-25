import numpy as np


class RBFVolatilitySurface:
    def __init__(
        self,
        coefficients,
        maturity_times,
        strike_prices,
        strike_std,
        maturity_std,
        constant_volatility,
    ):
        """
        Initialize the RBFVolatilitySurface class.

        Parameters:
        - coefficients: Array of RBF coefficients ω_j.
        - maturity_times: Array of RBF centers for time to maturity T_j.
        - strike_prices: Array of RBF centers for strike price K_j.
        - maturity_std: Standard deviation (spread) for time to maturity in the RBF.
        - strike_std: Standard deviation (spread) for strike prices in the RBF.
        - constant_volatility: Constant term φ_0 representing the weighted average of Black-Scholes implied volatilities.
        """
        self.coefficients = np.array(coefficients)
        self.maturity_times = np.array(maturity_times)
        self.strike_prices = np.array(strike_prices)
        self.strike_std = strike_std
        self.maturity_std = maturity_std
        self.constant_volatility = constant_volatility


    def implied_volatility_surface(
        self, 
        time_to_maturity, 
        strike_price
    ):
        """
        Calculate the implied volatility surface at a given pair (T, K).

        Parameters:
        - time_to_maturity: The time to maturity T.
        - strike_price: The strike price K.

        Returns:
        - The implied volatility σ(T, K) at (T, K).
        """

        rbf_values = np.exp(
            - ((time_to_maturity - self.maturity_times) ** 2) / (2 * self.maturity_std ** 2)
            - ((strike_price - self.strike_prices) ** 2) / (2 * self.strike_std ** 2)
        )
        rbf_sum = np.dot(self.coefficients, rbf_values)

        # Volatility surface σ(T, K) = φ_0 + Σ_j ω_j φ_j(T, K)
        return self.constant_volatility + rbf_sum


    @staticmethod
    def calculate_constant_volatility(
        data_implied_volatilities, 
        data_maturity_times, 
        data_strike_prices, 
        risk_free_rate, 
        underlying_price, 
        epsilon=1e-6
    ):
        """
        Calculate the constant term φ_0 as a weighted average of the Black-Scholes implied volatilities.

        Parameters:
        - data_implied_volatilities: Array of Black-Scholes implied volatilities σ_{BS}(T_i, K_i).
        - data_maturity_times: Array of maturity times T_i.
        - data_strike_prices: Array of strike prices K_i.
        - risk_free_rate: Risk-free rate r.
        - underlying_price: Current spot price of the underlying asset S.
        - epsilon: Small constant to avoid division by zero.

        Returns:
        - The constant volatility φ_0.
        """
        weights = []
        weighted_volatilities = []

        for t_i, k_i, implied_volatility in zip(
            data_maturity_times, data_strike_prices, data_implied_volatilities
        ):
            forward_strike = k_i * np.exp(-risk_free_rate * t_i)
            weight = 1 / ((underlying_price - forward_strike) ** 2 + epsilon)
            weights.append(weight)
            weighted_volatilities.append(weight * implied_volatility)

        constant_volatility = np.sum(weighted_volatilities) / np.sum(weights)
        return constant_volatility

