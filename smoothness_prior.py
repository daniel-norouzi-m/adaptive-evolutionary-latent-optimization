import numpy as np
from scipy.special import roots_genlaguerre
from numpy.linalg import pinv


class RBFQuadraticSmoothnessPrior:
    def __init__(
        self,
        maturity_times,
        strike_prices,
        sigma_t,
        sigma_k,
        n_roots,
        gamma,
        random_state=None,
    ):
        """
        Initialize the RBFQuadraticSmoothnessPrior class.

        Parameters:
        - maturity_times: Array of maturity times T_j.
        - strike_prices: Array of strike prices K_j.
        - sigma_t: Standard deviation for time to maturity.
        - sigma_k: Standard deviation for strike prices.
        - n_roots: Number of roots for Generalized Gauss-Laguerre Quadrature.
        - gamma: Smoothness control parameter.
        - random_state: Random seed for reproducibility.
        """
        self.maturity_times = np.array(maturity_times)
        self.strike_prices = np.array(strike_prices)
        self.sigma_t = sigma_t
        self.sigma_k = sigma_k
        self.n_roots = n_roots
        self.gamma = gamma
        self.random_generator = np.random.default_rng(random_state)

        # Precompute the roots and weights for the generalized Gauss-Laguerre quadrature with α = -1/2
        self.roots, self.weights = roots_genlaguerre(self.n_roots, -0.5)

        # Initialize covariance matrix placeholder
        self.covariance_matrix = None

    def compute_psi(
        self, 
        t, 
        k, 
        t_j, 
        k_j, 
        t_k, 
        k_k
    ):
        """
        Compute the polynomial factor Ψ(T, K) for the lambda matrix elements.

        Parameters:
        - t, k: Transformed time and strike price using Gauss-Laguerre quadrature.
        - t_j, k_j: Centers of the RBFs for maturity and strike for the first RBF.
        - t_k, k_k: Centers of the RBFs for maturity and strike for the second RBF.
        """
        term_1 = ((k - k_j) ** 2 / self.sigma_k ** 4 - 1 / self.sigma_k ** 2) * (
            (k - k_k) ** 2 / self.sigma_k ** 4 - 1 / self.sigma_k ** 2
        )
        term_2 = ((t - t_j) * (t - t_k)) / self.sigma_t**4
        return term_1 + term_2

    def calculate_lambda_matrix(self):
        """
        Calculate the lambda matrix (Λ) based on the smoothness prior distribution of the RBF surface parameters,
        leveraging symmetry to avoid redundant calculations.
        """
        n = len(self.maturity_times)
        lambda_matrix = np.zeros((n, n))

        for j in range(n):
            for k in range(
                j, n
            ):  # Only compute for upper triangle (including diagonal)
                delta_t = self.maturity_times[j] - self.maturity_times[k]
                delta_k = self.strike_prices[j] - self.strike_prices[k]

                # Exponential factor in the lambda matrix formula
                exp_factor = np.exp(
                    -(delta_t ** 2) / (4 * self.sigma_t ** 2)
                    - delta_k ** 2 / (4 * self.sigma_k ** 2)
                )

                # Average points for the RBFs
                t_avg = (self.maturity_times[j] + self.maturity_times[k]) / 2
                k_avg = (self.strike_prices[j] + self.strike_prices[k]) / 2

                # Gauss-Laguerre quadrature integration
                integral_sum = 0.0
                for i in range(self.n_roots):
                    for m in range(self.n_roots):
                        t_val_pos = t_avg + self.sigma_t * np.sqrt(self.roots[i])
                        t_val_neg = t_avg - self.sigma_t * np.sqrt(self.roots[i])
                        k_val_pos = k_avg + self.sigma_k * np.sqrt(self.roots[m])
                        k_val_neg = k_avg - self.sigma_k * np.sqrt(self.roots[m])

                        # Compute Ψ(T, K) with transformed variables
                        psi_val = (
                            self.compute_psi(
                                t_val_pos,
                                k_val_pos,
                                self.maturity_times[j],
                                self.strike_prices[j],
                                self.maturity_times[k],
                                self.strike_prices[k],
                            )
                            - self.compute_psi(
                                t_val_pos,
                                k_val_neg,
                                self.maturity_times[j],
                                self.strike_prices[j],
                                self.maturity_times[k],
                                self.strike_prices[k],
                            )
                            - self.compute_psi(
                                t_val_neg,
                                k_val_pos,
                                self.maturity_times[j],
                                self.strike_prices[j],
                                self.maturity_times[k],
                                self.strike_prices[k],
                            )
                            + self.compute_psi(
                                t_val_neg,
                                k_val_neg,
                                self.maturity_times[j],
                                self.strike_prices[j],
                                self.maturity_times[k],
                                self.strike_prices[k],
                            )
                        )

                        integral_sum += (
                            self.weights[i] * self.weights[m] * psi_val / \
                                np.sqrt(self.roots[i] * self.roots[m])
                        )

                lambda_matrix[j, k] = (
                    exp_factor * self.sigma_t * self.sigma_k / 4 * integral_sum
                )

                if j != k:
                    lambda_matrix[k, j] = lambda_matrix[j, k]  # Use symmetry

        return lambda_matrix

    def prior_covariance(self):
        """
        Calculate the covariance matrix of the smoothness prior distribution.

        Returns:
        - covariance_matrix: The covariance matrix as gamma^2 * Lambda^(-1).
        """
        lambda_matrix = self.calculate_lambda_matrix()

        # Covariance matrix is gamma^2 * Λ^(-1)
        self.covariance_matrix = self.gamma ** 2 * pinv(lambda_matrix)
        return self.covariance_matrix

    def sample_smooth_surfaces(
        self, 
        n_samples
    ):
        """
        Sample smooth surface parameters from the Gaussian smoothness prior distribution.

        Parameters:
        - n_samples: Number of samples to generate.

        Returns:
        - samples: A (n_samples, N) array where each row represents a sample of the RBF coefficients.
        """
        # Check if the covariance matrix has been calculated
        if self.covariance_matrix is None:
            self.prior_covariance()

        # Generate samples using multivariate normal distribution
        samples = self.random_generator.multivariate_normal(
            mean=np.zeros(len(self.maturity_times)),
            cov=self.covariance_matrix,
            size=n_samples,
        )

        return samples
