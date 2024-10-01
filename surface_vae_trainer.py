from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from coefficients_vae_net import CoefficientsVAE, CoefficientsDataset, coefficients_beta_vae_loss
from torch.optim import Adam
import torch
from tqdm import tqdm


class SurfaceVAETrainer:
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        n_layers,
        data_dim,
        latent_diagonal,
        batch_size,
        beta,
        pre_train_learning_rate,
        fine_tune_learning_rate,
        pre_train_epochs,
        fine_tune_epochs,
        device,
    ):
        """
        Initialize the SurfaceVAETrainer class with the given hyperparameters and model configuration.
        """
        self.device = device
        self.latent_diagonal = torch.tensor(latent_diagonal, dtype=torch.float32, device=device)
        self.beta = beta
        self.batch_size = batch_size
        self.pre_train_learning_rate = pre_train_learning_rate
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.pre_train_epochs = pre_train_epochs
        self.fine_tune_epochs = fine_tune_epochs

        # Initialize the VAE model and optimizer
        self.model = CoefficientsVAE(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            data_dim=data_dim,
        ).to(self.device)

        self.pre_train_optimizer = Adam(self.model.parameters(), lr=self.pre_train_learning_rate)
        self.fine_tune_optimizer = Adam(self.model.parameters(), lr=self.fine_tune_learning_rate)

    def _train(
        self, 
        sampled_surface_coefficients, 
        n_epochs, 
        optimizer, 
        loss_history, 
        experiment_name=None
    ):
        """
        Generic training function for the VAE model. It handles both pre-training and fine-tuning.

        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the Dataset.
        - n_epochs: The number of training epochs.
        - optimizer: The optimizer to use (Adam for pre-training or fine-tuning).
        - loss_history: A dictionary to keep track of the reconstruction and KL losses.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        # Setup SummaryWriter for TensorBoard logging
        writer = SummaryWriter(log_dir=f"runs/{experiment_name}") if experiment_name else None

        # Create dataset and dataloader
        dataset = CoefficientsDataset(sampled_surface_coefficients)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        # Begin training
        for epoch in tqdm(range(n_epochs)):
            for batch_idx, batch_surface_coefficients in enumerate(dataloader):
                # Move data to the appropriate device
                batch_surface_coefficients = batch_surface_coefficients.to(self.device)

                # Zero gradients before backpropagation
                optimizer.zero_grad()

                # Forward pass through the VAE model
                reconstructed_surface_coefficients, latent_mean, latent_log_var = self.model(batch_surface_coefficients)

                # Compute the beta-VAE loss
                total_loss, reconstruction_loss, kl_divergence = coefficients_beta_vae_loss(
                    surface_coefficients=batch_surface_coefficients,
                    reconstructed_surface_coefficients=reconstructed_surface_coefficients,
                    latent_mean=latent_mean,
                    latent_log_var=latent_log_var,
                    latent_diagonal=self.latent_diagonal,
                    beta=self.beta
                )

                # Backpropagation and optimization
                total_loss.backward()
                optimizer.step()

                # Update loss dictionary
                loss_history["Reconstruction Loss"].append(reconstruction_loss.item())
                loss_history["KL Loss"].append(kl_divergence.item())
                loss_history["Total Loss"].append(total_loss.item())

                # Current loss dict
                current_loss = {
                    "Reconstruction Loss": reconstruction_loss.item(),
                    "KL Loss": kl_divergence.item(),
                    "Total Loss": total_loss.item(),
                }

                # Print the losses for each batch
                print(f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}, Losses: {current_loss}")

                # Log losses in TensorBoard
                if writer:
                    writer.add_scalar("Reconstruction Loss", reconstruction_loss.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("KL Loss", kl_divergence.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("Total Loss", total_loss.item(), epoch * len(dataloader) + batch_idx)

        # Close TensorBoard writer
        if writer:
            writer.close()

    def pre_train(
        self, 
        sampled_surface_coefficients, 
        experiment_name=None
    ):
        """
        Pre-train the VAE model.

        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the Dataset.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.pre_train_loss_history = {
            "Reconstruction Loss": [],
            "KL Loss": [],
            "Total Loss": [],
        }

        self._train(
            sampled_surface_coefficients,
            n_epochs=self.pre_train_epochs,
            optimizer=self.pre_train_optimizer,
            loss_history=self.pre_train_loss_history,
            experiment_name=experiment_name,
        )

    def fine_tune(
        self, 
        sampled_surface_coefficients, 
        experiment_name=None
    ):
        """
        Fine-tune the VAE model.

        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the Dataset.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.fine_tune_loss_history = {
            "Reconstruction Loss": [],
            "KL Loss": [],
            "Total Loss": [],
        }

        self._train(
            sampled_surface_coefficients,
            n_epochs=self.fine_tune_epochs,
            optimizer=self.fine_tune_optimizer,
            loss_history=self.fine_tune_loss_history,
            experiment_name=experiment_name,
        )

    def pre_train_with_sampling(
        self, 
        smoothness_prior, 
        experiment_name=None
    ):
        """
        Pre-train the VAE model with sampling from the smoothness prior.

        Parameters:
        - smoothness_prior: An instance of the RBFQuadraticSmoothnessPrior class used to sample surface coefficients.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.pre_train_loss_history = {
            "Reconstruction Loss": [],
            "KL Loss": [],
            "Total Loss": [],
        }

        # Setup SummaryWriter for TensorBoard logging
        writer = SummaryWriter(log_dir=f"runs/{experiment_name}") if experiment_name else None

        # Begin training
        for epoch in tqdm(range(self.pre_train_epochs)):
            for batch_idx in range(1):
                # Sample surface coefficients from the smoothness prior
                sampled_surface_coefficients = smoothness_prior.sample_smooth_surfaces(self.batch_size)

                # Move sampled data to the appropriate device
                sampled_surface_coefficients = torch.tensor(sampled_surface_coefficients, device=self.device, dtype=torch.float32)

                # Zero gradients before backpropagation
                self.pre_train_optimizer.zero_grad()

                # Forward pass through the VAE model
                reconstructed_surface_coefficients, latent_mean, latent_log_var = self.model(sampled_surface_coefficients)

                # Compute the beta-VAE loss
                total_loss, reconstruction_loss, kl_divergence = coefficients_beta_vae_loss(
                    surface_coefficients=sampled_surface_coefficients,
                    reconstructed_surface_coefficients=reconstructed_surface_coefficients,
                    latent_mean=latent_mean,
                    latent_log_var=latent_log_var,
                    latent_diagonal=self.latent_diagonal,
                    beta=self.beta
                )

                # Backpropagation and optimization
                total_loss.backward()
                self.pre_train_optimizer.step()

                # Update loss dictionary
                self.pre_train_loss_history["Reconstruction Loss"].append(reconstruction_loss.item())
                self.pre_train_loss_history["KL Loss"].append(kl_divergence.item())
                self.pre_train_loss_history["Total Loss"].append(total_loss.item())

                # Print the losses for each batch
                # current_loss = {
                #     "Reconstruction Loss": reconstruction_loss.item(),
                #     "KL Loss": kl_divergence.item(),
                #     "Total Loss": total_loss.item(),
                # }
                # print(f"Epoch {epoch + 1}/{self.pre_train_epochs}, Batch {batch_idx + 1}, Losses: {current_loss}")

                # Log losses in TensorBoard
                if writer:
                    writer.add_scalar("Reconstruction Loss", reconstruction_loss.item(), epoch * self.batch_size + batch_idx)
                    writer.add_scalar("KL Loss", kl_divergence.item(), epoch * self.batch_size + batch_idx)
                    writer.add_scalar("Total Loss", total_loss.item(), epoch * self.batch_size + batch_idx)

        # Close TensorBoard writer
        if writer:
            writer.close()
        
    def dupire_price_prediction_loss(
        self,
        surface_coefficients_batch,
        data_call_option_prices=None,
        data_maturity_times=None,
        data_strike_prices=None
    ):
        """
        Calculate the price prediction loss for a batch of surface coefficients.

        Parameters:
        - surface_coefficients_batch: A batch of surface coefficients with shape (batch, N).
        - data_call_option_prices: Observed call option prices. If provided, set the corresponding attribute.
        - data_maturity_times: Maturity times corresponding to the observed call option prices.
        - data_strike_prices: Strike prices corresponding to the observed call option prices.

        Returns:
        - mse_loss: The mean squared error (MSE) loss between the predicted and observed call option prices.
        """

        # Set class attributes if provided
        if data_call_option_prices is not None:
            self.data_call_option_prices = torch.tensor(data_call_option_prices, dtype=torch.float32, device=self.device)
        if data_maturity_times is not None:
            self.data_maturity_times = torch.tensor(data_maturity_times, dtype=torch.float32, device=self.device)
        if data_strike_prices is not None:
            self.data_strike_prices = torch.tensor(data_strike_prices, dtype=torch.float32, device=self.device)

        # Ensure that RBF evaluations are computed if not already cached
        if not hasattr(self, 'rbf_evaluations') or self.rbf_evaluations is None:
            # Expand dimensions to enable broadcasting and compute RBF evaluations
            time_diff = (self.data_maturity_times[:, None] - torch.tensor(self.maturity_times, device=self.device)) ** 2
            strike_diff = (self.data_strike_prices[:, None] - torch.tensor(self.strike_prices, device=self.device)) ** 2

            # rbf_evaluations: (M, N)
            self.rbf_evaluations = torch.exp(
                -time_diff / (2 * self.maturity_std ** 2)
                - strike_diff / (2 * self.strike_std ** 2)
            )

        # surface_coefficients_batch: (batch_size, N)

        # Compute the predicted volatilities for each surface coefficients batch
        predicted_volatility_batch = torch.matmul(surface_coefficients_batch, self.rbf_evaluations.T) + self.constant_volatility

        # Now predict the call option prices using the PINN for each batch element
        # Repeat the maturity and strike tensors across the batch dimension
        repeated_maturity_times = self.data_maturity_times.unsqueeze(0).repeat(surface_coefficients_batch.size(0), 1)
        repeated_strike_prices = self.data_strike_prices.unsqueeze(0).repeat(surface_coefficients_batch.size(0), 1)

        # Pass through the model
        predicted_call_option_prices = self.model(
            repeated_maturity_times,  # Shape: (batch_size, M)
            repeated_strike_prices,   # Shape: (batch_size, M)
            predicted_volatility_batch  # Shape: (batch_size, M)
        )

        # We now have predicted_call_option_prices of shape (batch_size, M)

        # Ensure that the observed prices are of the correct shape
        repeated_observed_prices = self.data_call_option_prices.unsqueeze(0).expand_as(predicted_call_option_prices)

        # Compute the squared differences between predicted and observed call option prices
        squared_errors = (predicted_call_option_prices - repeated_observed_prices) ** 2

        # Sum the squared errors over the M points (along the second dimension)
        sum_squared_errors_per_batch = torch.sum(squared_errors, dim=1)  # Shape: (batch_size,)

        # Compute the sum of the summed squared errors across the batch
        mse_loss = torch.sum(sum_squared_errors_per_batch)  # Final scalar loss

        return mse_loss
