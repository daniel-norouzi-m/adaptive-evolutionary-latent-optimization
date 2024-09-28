import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from call_option_net import CallOptionPINN, pinn_dupire_loss
from rbf_volatility_surface import SurfaceDataset

class DupirePINNTrainer:
    def __init__(
        self,
        hidden_dim,
        n_layers,
        batch_size,
        pde_loss_coefficient,
        maturity_zero_loss_coefficient,
        strike_zero_loss_coefficient,
        strike_infinity_loss_coefficient,
        pre_train_learning_rate,
        fine_tune_learning_rate,
        pre_train_epochs,
        fine_tune_epochs,
        maturity_min,
        maturity_max,
        strike_min,
        strike_max,
        volatility_mean,
        volatility_std,
        maturity_time_list,
        strike_price_list,
        strike_std,
        maturity_std,
        constant_volatility,
        strike_infinity,
        device,
    ):
        """
        Initialize the DupirePINNTrainer class with the given hyperparameters and model configuration.
        """
        self.device = device
        self.batch_size = batch_size
        self.pde_loss_coefficient = pde_loss_coefficient
        self.maturity_zero_loss_coefficient = maturity_zero_loss_coefficient
        self.strike_zero_loss_coefficient = strike_zero_loss_coefficient
        self.strike_infinity_loss_coefficient = strike_infinity_loss_coefficient
        self.pre_train_learning_rate = pre_train_learning_rate
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.pre_train_epochs = pre_train_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.maturity_time_list = maturity_time_list
        self.strike_price_list = strike_price_list
        self.strike_std = strike_std
        self.maturity_std = maturity_std
        self.constant_volatility = constant_volatility
        self.strike_infinity = strike_infinity

        # Initialize the PINN model and optimizer
        self.model = CallOptionPINN(
            hidden_dim,
            n_layers,
            maturity_min,
            maturity_max,
            strike_min,
            strike_max,
            volatility_mean,
            volatility_std,
        ).to(self.device)

        self.pre_train_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.pre_train_learning_rate)
        self.fine_tune_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fine_tune_learning_rate)

    def _train(
        self, 
        sampled_surface_coefficients, 
        n_epochs, 
        optimizer, 
        loss_history, 
        experiment_name=None
    ):
        """
        Generic training function for the PINN model. It handles both pre-training and fine-tuning.
        
        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the SurfaceDataset.
        - n_epochs: The number of training epochs.
        - optimizer: The optimizer to use (Adam for pre-training or fine-tuning).
        - loss_history: A dictionary to keep track of the individual losses and the total loss.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        # Setup SummaryWriter for TensorBoard logging
        writer = SummaryWriter(log_dir=f"runs/{experiment_name}") if experiment_name else None

        # Create dataset and dataloader
        dataset = SurfaceDataset(
            sampled_surface_coefficients=sampled_surface_coefficients,
            maturity_time_list=self.maturity_time_list,
            strike_price_list=self.strike_price_list,
            strike_std=self.strike_std,
            maturity_std=self.maturity_std,
            constant_volatility=self.constant_volatility,
            strike_infinity=self.strike_infinity,
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        # Begin training
        for epoch in range(n_epochs):
            for batch_idx, (time_to_maturity, strike_price, implied_volatility) in enumerate(dataloader):
                # Move data to the appropriate device
                time_to_maturity = time_to_maturity.to(self.device)
                strike_price = strike_price.to(self.device)
                implied_volatility = implied_volatility.to(self.device)

                # Zero gradients before backpropagation
                optimizer.zero_grad()

                # Forward pass through the model to get call option price predictions
                call_option_price = self.model(time_to_maturity, strike_price, implied_volatility)

                # Compute the PDE and boundary condition losses
                pde_loss, maturity_zero_loss, strike_zero_loss, strike_infinity_loss = pinn_dupire_loss(
                    call_option_price,
                    time_to_maturity,
                    strike_price,
                    implied_volatility,
                    strike_infinity=self.strike_infinity,
                )

                # Compute total loss with coefficients
                total_loss = (
                    self.pde_loss_coefficient * pde_loss
                    + self.maturity_zero_loss_coefficient * maturity_zero_loss
                    + self.strike_zero_loss_coefficient * strike_zero_loss
                    + self.strike_infinity_loss_coefficient * strike_infinity_loss
                )

                # Backpropagation and optimization
                total_loss.backward()
                optimizer.step()

                # Update loss dictionary
                loss_history["PDE Loss"].append(pde_loss.item())
                loss_history["Zero Maturity Loss"].append(maturity_zero_loss.item())
                loss_history["Zero Strike Loss"].append(strike_zero_loss.item())
                loss_history["Infinity Strike Loss"].append(strike_infinity_loss.item())
                loss_history["Total Loss"].append(total_loss.item())

                current_loss = {
                    "PDE Loss": pde_loss.item(),
                    "Zero Maturity Loss": maturity_zero_loss.item(),
                    "Zero Strike Loss": strike_zero_loss.item(),
                    "Infinity Strike Loss": strike_infinity_loss.item(),
                    "Total Loss": total_loss.item(),
                }

                # Print the losses for each batch
                print(f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}, Losses: {current_loss}")

                # Log losses in TensorBoard
                if writer:
                    writer.add_scalar("PDE Loss", pde_loss.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("Zero Maturity Loss", maturity_zero_loss.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("Zero Strike Loss", strike_zero_loss.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("Infinity Strike Loss", strike_infinity_loss.item(), epoch * len(dataloader) + batch_idx)
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
        Pre-train the PINN model.
        
        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the SurfaceDataset.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.pre_train_loss_history = {
            "PDE Loss": [],
            "Zero Maturity Loss": [],
            "Zero Strike Loss": [],
            "Infinity Strike Loss": [],
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
        Fine-tune the PINN model.
        
        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the SurfaceDataset.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.fine_tune_loss_history = {
            "PDE Loss": [],
            "Zero Maturity Loss": [],
            "Zero Strike Loss": [],
            "Infinity Strike Loss": [],
            "Total Loss": [],
        }

        self._train(
            sampled_surface_coefficients,
            n_epochs=self.fine_tune_epochs,
            optimizer=self.fine_tune_optimizer,
            loss_history=self.fine_tune_loss_history,
            experiment_name=experiment_name,
        )