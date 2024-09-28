import numpy as np
import torch
import torch.nn as nn


class CallOptionPINN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_layers,
        maturity_min,
        maturity_max,
        strike_min,
        strike_max,
        volatility_mean,
        volatility_std,
    ):
        super(CallOptionPINN, self).__init__()

        # Save normalization parameters
        self.maturity_min = maturity_min
        self.maturity_max = maturity_max
        self.strike_min = strike_min
        self.strike_max = strike_max
        self.volatility_mean = volatility_mean
        self.volatility_std = volatility_std

        # Define the layers of the network
        layers = []
        input_dim = 3  # Inputs: time_to_maturity, strike_price, volatility

        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        # Final output layer (produces call option price)
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights with Xavier normal initialization (gain for ELU)
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, 
        time_to_maturity, 
        strike_price, 
        volatility
    ):
        # Normalize the inputs:

        # Time to maturity and strike price normalization (uniform normalization)
        time_to_maturity_norm = (
            time_to_maturity - self.maturity_min) / (
            self.maturity_max - self.maturity_min
        )

        strike_price_norm = (
            strike_price - self.strike_min) / (
            self.strike_max - self.strike_min
        )

        # Volatility normalization (using CDF of the normal distribution)
        volatility_standardized = (
            volatility - self.volatility_mean
        ) / self.volatility_std
        volatility_norm = 0.5 * (
            1 + torch.erf(volatility_standardized / torch.sqrt(torch.tensor(2.0)))
        )

        # Concatenate inputs for the network
        inputs = torch.cat(
            [
                time_to_maturity_norm.unsqueeze(1),
                strike_price_norm.unsqueeze(1),
                volatility_norm.unsqueeze(1),
            ],
            dim=1,
        )

        # Pass through the network
        call_option_price = self.network(inputs)

        return call_option_price


def pinn_dupire_loss(
    call_option_price,
    time_to_maturity,
    strike_price,
    volatility,
    strike_infinity=2.5,
    risk_free_rate=np.log(1.02),
    underlying_price=1.0,
):
    """
    Compute the Dupire PDE loss and boundary condition losses for the PINN with batched inputs.

    Parameters:
    - call_option_price: Batched tensor of predicted call option prices from the PINN (batch_size, num_points).
    - time_to_maturity: Batched tensor of time to maturities T_i (batch_size, num_points).
    - strike_price: Batched tensor of strike prices K_i (batch_size, num_points).
    - volatility: Batched tensor of volatilities σ(T_i, K_i) (batch_size, num_points).
    - strike_infinity: Large value for the high strike price boundary condition, default is 2.5.
    - risk_free_rate: The risk-free interest rate r, default is log(1.02).
    - underlying_price: The current spot price S_0, default is 1.0.

    Returns:
    - pde_loss: The average Dupire forward PDE loss across all batches.
    - maturity_zero_loss: The average loss for the boundary condition at maturity T=0 across all batches.
    - strike_zero_loss: The average loss for the boundary condition at K=0 across all batches.
    - strike_infinity_loss: The average loss for the boundary condition as K→∞ (approximated by strike_infinity) across all batches.
    """

    # Initialize loss accumulators
    total_pde_loss = 0.0
    total_maturity_zero_loss = 0.0
    total_strike_zero_loss = 0.0
    total_strike_infinity_loss = 0.0

    # Iterate over each batch
    batch_size = call_option_price.shape[0]
    for batch_idx in range(batch_size):
        # Select individual batch
        call_option_price_b = call_option_price[batch_idx]
        time_to_maturity_b = time_to_maturity[batch_idx]
        strike_price_b = strike_price[batch_idx]
        volatility_b = volatility[batch_idx]

        # Dupire Forward PDE Loss:
        # Mask to exclude boundary points (non-boundary data)
        non_boundary_mask = (
            (time_to_maturity_b > 0) & (strike_price_b > 0) & (strike_price_b < strike_infinity)
        )

        # Masked tensors for non-boundary data
        time_to_maturity_nb = time_to_maturity_b[non_boundary_mask]
        strike_price_nb = strike_price_b[non_boundary_mask]
        volatility_nb = volatility_b[non_boundary_mask]
        call_option_price_nb = call_option_price_b[non_boundary_mask]

        # Compute the derivatives for the non-boundary points
        call_price_t = torch.autograd.grad(
            call_option_price_nb.sum(),
            time_to_maturity_nb,
            create_graph=True,
        )[0]

        call_price_k = torch.autograd.grad(
            call_option_price_nb.sum(),
            strike_price_nb,
            create_graph=True,
        )[0]

        call_price_k2 = torch.autograd.grad(
            call_price_k.sum(),
            strike_price_nb,
            create_graph=True,
        )[0]

        # Dupire PDE residuals for non-boundary points
        pde_residual = (
            call_price_t
            + 0.5 * volatility_nb ** 2 * strike_price_nb ** 2 * call_price_k2
            + risk_free_rate * strike_price_nb * call_price_k
            - risk_free_rate * call_option_price_nb
        )

        # Sum of squared residuals for the non-boundary data
        pde_loss = torch.sum(pde_residual ** 2)

        # Accumulate total PDE loss
        total_pde_loss += pde_loss

        # Direct comparisons for T=0 and K=0 boundaries
        maturity_zero_mask = (time_to_maturity_b == 0)
        strike_zero_mask = (strike_price_b == 0)
        strike_infinity_mask = (strike_price_b == strike_infinity)

        # Boundary Condition Loss at T=0 (Maturity)
        maturity_zero_loss = torch.sum(
            (
                call_option_price_b[maturity_zero_mask]
                - torch.clamp(underlying_price - strike_price_b[maturity_zero_mask], min=0)
            )
            ** 2
        )

        # Boundary Condition Loss at K=0 (Zero Strike Price)
        strike_zero_loss = torch.sum(
            (call_option_price_b[strike_zero_mask] - underlying_price) ** 2
        )

        # Boundary Condition Loss as K→∞ (High Strike Price)
        strike_infinity_loss = torch.sum(
            (call_option_price_b[strike_infinity_mask] - 0) ** 2
        )

        # Accumulate total boundary condition losses
        total_maturity_zero_loss += maturity_zero_loss
        total_strike_zero_loss += strike_zero_loss
        total_strike_infinity_loss += strike_infinity_loss

    # Average the losses across batches
    avg_pde_loss = total_pde_loss / batch_size
    avg_maturity_zero_loss = total_maturity_zero_loss / batch_size
    avg_strike_zero_loss = total_strike_zero_loss / batch_size
    avg_strike_infinity_loss = total_strike_infinity_loss / batch_size

    return avg_pde_loss, avg_maturity_zero_loss, avg_strike_zero_loss, avg_strike_infinity_loss