import numpy as np
from scipy.special import roots_genlaguerre
from numpy.linalg import pinv


class RBFQuadraticSmoothnessPrior:
    def __init__(
        self,
        maturity_times,
        strike_prices,
        maturity_std,
        strike_std,
        n_roots,
        gamma,
        random_state=None,
    ):
        """
        Initialize the RBFQuadraticSmoothnessPrior class.

        Parameters:
        - maturity_times: Array of maturity times T_j.
        - strike_prices: Array of strike prices K_j.
        - maturity_std: Standard deviation for time to maturity.
        - strike_std: Standard deviation for strike prices.
        - n_roots: Number of roots for Generalized Gauss-Laguerre Quadrature.
        - gamma: Smoothness control parameter.
        - random_state: Random seed for reproducibility.
        """
        self.maturity_times = np.array(maturity_times)
        self.strike_prices = np.array(strike_prices)
        self.maturity_std = maturity_std
        self.strike_std = strike_std
        self.n_roots = n_roots
        self.gamma = gamma
        self.random_generator = np.random.default_rng(random_state)

        # Precompute the roots and weights for the generalized Gauss-Laguerre quadrature with α = -1/2
        self.roots, self.weights = roots_genlaguerre(self.n_roots, -0.5)

        # Initialize covariance and lambda matrix placeholder
        self.covariance_matrix = None
        self.lambda_matrix = None

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
        term_1 = ((k - k_j) ** 2 / self.strike_std ** 4 - 1 / self.strike_std ** 2) * (
            (k - k_k) ** 2 / self.strike_std ** 4 - 1 / self.strike_std ** 2
        )
        term_2 = ((t - t_j) * (t - t_k)) / self.maturity_std ** 4
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
                    -(delta_t ** 2) / (4 * self.maturity_std ** 2)
                    - delta_k ** 2 / (4 * self.strike_std ** 2)
                )

                # Average points for the RBFs
                t_avg = (self.maturity_times[j] + self.maturity_times[k]) / 2
                k_avg = (self.strike_prices[j] + self.strike_prices[k]) / 2

                # Vectorized Gauss-Laguerre quadrature integration
                t_sqrt_roots = np.sqrt(self.roots) * self.maturity_std  # Precompute sqrt(roots) * maturity_std
                k_sqrt_roots = np.sqrt(self.roots) * self.strike_std  # Precompute sqrt(roots) * strike_std

                # Generate all possible combinations of t_val_pos, t_val_neg, k_val_pos, k_val_neg
                t_vals_pos = t_avg + t_sqrt_roots
                t_vals_neg = t_avg - t_sqrt_roots
                k_vals_pos = k_avg + k_sqrt_roots
                k_vals_neg = k_avg - k_sqrt_roots

                # Initialize psi_vals for all combinations and apply conditions
                psi_vals = np.zeros((self.n_roots, self.n_roots, 4))

                # First combination: t_val_pos, k_val_pos
                mask_pos_pos = (t_vals_pos[:, np.newaxis] >= 0) & (k_vals_pos[np.newaxis, :] >= 0)
                psi_vals[:, :, 0] = np.where(
                    mask_pos_pos,
                    self.compute_psi(
                        t_vals_pos[:, np.newaxis],
                        k_vals_pos[np.newaxis, :],
                        self.maturity_times[j],
                        self.strike_prices[j],
                        self.maturity_times[k],
                        self.strike_prices[k]
                    ),
                    0
                )

                # Second combination: t_val_pos, k_val_neg
                mask_pos_neg = (t_vals_pos[:, np.newaxis] >= 0) & (k_vals_neg[np.newaxis, :] >= 0)
                psi_vals[:, :, 1] = np.where(
                    mask_pos_neg,
                    self.compute_psi(
                        t_vals_pos[:, np.newaxis],
                        k_vals_neg[np.newaxis, :],
                        self.maturity_times[j],
                        self.strike_prices[j],
                        self.maturity_times[k],
                        self.strike_prices[k]
                    ),
                    0
                )

                # Third combination: t_val_neg, k_val_pos
                mask_neg_pos = (t_vals_neg[:, np.newaxis] >= 0) & (k_vals_pos[np.newaxis, :] >= 0)
                psi_vals[:, :, 2] = np.where(
                    mask_neg_pos,
                    self.compute_psi(
                        t_vals_neg[:, np.newaxis],
                        k_vals_pos[np.newaxis, :],
                        self.maturity_times[j],
                        self.strike_prices[j],
                        self.maturity_times[k],
                        self.strike_prices[k]
                    ),
                    0
                )

                # Fourth combination: t_val_neg, k_val_neg
                mask_neg_neg = (t_vals_neg[:, np.newaxis] >= 0) & (k_vals_neg[np.newaxis, :] >= 0)
                psi_vals[:, :, 3] = np.where(
                    mask_neg_neg,
                    self.compute_psi(
                        t_vals_neg[:, np.newaxis],
                        k_vals_neg[np.newaxis, :],
                        self.maturity_times[j],
                        self.strike_prices[j],
                        self.maturity_times[k],
                        self.strike_prices[k]
                    ),
                    0
                )

                # Compute psi_val by summing and subtracting according to the original logic
                psi_val_combined = psi_vals[:, :, 0] - psi_vals[:, :, 1] - psi_vals[:, :, 2] + psi_vals[:, :, 3]

                # Compute the integral sum
                integral_sum = np.sum(
                    self.weights[:, np.newaxis] * self.weights[np.newaxis, :] * psi_val_combined / \
                    np.sqrt(self.roots[:, np.newaxis] * self.roots[np.newaxis, :])
                )

                # Update the lambda matrix
                lambda_matrix[j, k] = (
                    exp_factor * self.maturity_std * self.strike_std / 4 * integral_sum
                )

                if j != k:
                    lambda_matrix[k, j] = lambda_matrix[j, k]  # Use symmetry

        lambda_matrix[lambda_matrix < 1e-12] = 0
        lambda_matrix += 1e-6 * np.eye(lambda_matrix.shape[0])

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(lambda_matrix)

        # Clip negative eigenvalues to zero
        eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)

        # Reconstruct the matrix with the clipped eigenvalues
        lambda_matrix_positive = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        lambda_matrix_positive += 1e-6 * np.eye(lambda_matrix_positive.shape[0])
        lambda_matrix_positive = (lambda_matrix_positive + lambda_matrix_positive.T) / 2  # Ensure symmetry

        self.lambda_matrix = lambda_matrix_positive            

        return self.lambda_matrix    

    def prior_covariance(self):
        """
        Calculate the covariance matrix of the smoothness prior distribution.

        Returns:
        - covariance_matrix: The covariance matrix as gamma^2 * Lambda^(-1).
        """
        # Check if the covariance matrix has been calculated
        if self.lambda_matrix is None:
            self.calculate_lambda_matrix()

        inverse_lambda = pinv(self.lambda_matrix)    

        # Covariance matrix is gamma^2 * Λ^(-1)
        self.covariance_matrix = self.gamma ** 2 * (inverse_lambda + inverse_lambda.T) / 2
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
