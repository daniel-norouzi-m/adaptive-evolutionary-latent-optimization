import torch
import numpy as np
import plotly.graph_objects as go

from evolutionary_algorithm import EvolutionaryAlgorithm
from latent_dimension_assessment import sample_latent_vectors, latent_space_assessment

class AdaptiveEvolutionaryLatentOptimization:
    def __init__(
        self,
        vae_trainer,
        pinn_trainer,
        latent_diagonal,
        population_size,
        mutation_strength,
        selection_pressure_parameter,
        n_generations,
        n_cycles,
    ):
        """
        Initialize the AdELO class with pre-trained models and hyperparameters.

        Parameters:
        - vae_trainer: Pre-trained VAE trainer instance.
        - pinn_trainer: Pre-trained PINN trainer instance.
        - latent_diagonal: Diagonal of the latent prior covariance matrix.
        - population_size: Population size for the EA.
        - mutation_strength: Mutation strength parameter for the EA.
        - selection_pressure_parameter: Selection pressure parameter for the EA.
        - n_generations: Number of generations per EA optimization cycle.
        - n_cycles: Number of adaptive cycles.
        """
        self.vae_trainer = vae_trainer
        self.pinn_trainer = pinn_trainer
        self.latent_diagonal = latent_diagonal
        self.population_size = population_size
        self.mutation_strength = mutation_strength
        self.selection_pressure_parameter = selection_pressure_parameter
        self.n_generations = n_generations
        self.n_cycles = n_cycles

        # Histories for each cycle
        self.optimization_histories = []
        self.landscape_median_condition_numbers = []
        self.landscape_lipschitz_constants = []
        self.population_median_condition_numbers = []

        # Placeholder for final population after the last cycle
        self.final_population = None

    def run_cycle(self):
        """
        Runs the adaptive cycles of EA and fine-tuning.
        """
        for cycle in range(self.n_cycles):
            print(f"Starting Cycle {cycle + 1}/{self.n_cycles}")
            
            # Skip fine-tuning for the first cycle, perform fine-tuning for subsequent cycles
            if cycle > 0:
                print("Fine-Tuning VAE and PINN...")
                self.vae_trainer.fine_tune(self.final_population.values)
                self.pinn_trainer.fine_tune(self.final_population.values)

            # Run the evolutionary optimization (EA step)
            print("Running Evolutionary Optimization...")
            ea_optimizer = EvolutionaryAlgorithm(
                vae_trainer=self.vae_trainer,
                pinn_trainer=self.pinn_trainer,
                latent_diagonal=self.latent_diagonal,
                population_size=self.population_size,
                mutation_strength=self.mutation_strength,
                selection_pressure_parameter=self.selection_pressure_parameter,
                n_generations=self.n_generations
            )
            ea_optimizer.optimize()

            # Store the final population and optimization history
            self.optimization_histories.append(ea_optimizer.optimization_history)

            # # Assess the optimization landscape (condition numbers and Lipschitz constant)
            # print("Assessing Optimization Landscape...")
            # all_condition_numbers = []
            # max_lipschitz_constant = float('-inf')

            # # Assess latent space multiple times and update metrics
            # for i in range(100):
            #     torch.manual_seed(i + 2)  # Update seed for each iteration
            #     latent_samples_batch = sample_latent_vectors(100, self.latent_diagonal)
            #     condition_numbers, lipschitz_constant = latent_space_assessment(
            #         latent_samples_batch, self.vae_trainer, self.pinn_trainer
            #     )
            #     all_condition_numbers.extend(condition_numbers)
            #     max_lipschitz_constant = max(max_lipschitz_constant, lipschitz_constant)

            # # Store median of the condition numbers and Lipschitz constant after fine-tuning
            # self.landscape_median_condition_numbers.append(np.median(all_condition_numbers))
            # self.landscape_lipschitz_constants.append(max_lipschitz_constant)

            # # Assess the terminal population condition numbers
            # print("Assessing Terminal Population...")
            # terminal_population_samples = torch.tensor(ea_optimizer.population, dtype=torch.float32)
            # terminal_condition_numbers, _ = latent_space_assessment(
            #     terminal_population_samples, self.vae_trainer, self.pinn_trainer
            # )
            # self.population_median_condition_numbers.append(np.median(terminal_condition_numbers))

            # Update the final population with the current EA optimizer's population
            self.final_population = ea_optimizer.final_population

            print(f"Cycle {cycle + 1}/{self.n_cycles} Completed.\n")

    def plot_evolutions(self):
        """
        Merges all the optimization histories from each cycle and plots them in a combined figure.
        """
        # Initialize figure
        fig = go.Figure()

        # Initialize lists to hold the combined data across all cycles
        merged_best_fitness = []
        merged_average_fitness = []
        merged_percentile_fitness = []
        generation_numbers = []

        # Track the total number of generations
        total_generations = 0

        # Loop through the optimization histories of all cycles
        for cycle_idx, optimization_history in enumerate(self.optimization_histories):
            # Number of generations in the current cycle
            n_generations_cycle = len(optimization_history["Best Fitness"])

            # Generate generation numbers for the current cycle
            generation_numbers_cycle = list(range(total_generations + 1, total_generations + n_generations_cycle + 1))

            # Append the data from the current cycle to the merged lists
            merged_best_fitness.extend(optimization_history["Best Fitness"])
            merged_average_fitness.extend(optimization_history["Average Fitness"])
            merged_percentile_fitness.extend(optimization_history["0.05 Percentile Fitness"])
            generation_numbers.extend(generation_numbers_cycle)

            # Update total generations
            total_generations += n_generations_cycle

        # Add trace for best fitness
        fig.add_trace(go.Scatter(
            x=generation_numbers,
            y=merged_best_fitness,
            mode='lines+markers',
            name='Best Fitness',
            line=dict(color='royalblue', width=3),
            marker=dict(size=6, symbol='circle'),
            hovertemplate='<b>Iteration %{x}</b><br>Best Fitness: %{y:.4f}<extra></extra>'
        ))

        # Add trace for average fitness
        fig.add_trace(go.Scatter(
            x=generation_numbers,
            y=merged_average_fitness,
            mode='lines+markers',
            name='Average Fitness',
            line=dict(color='firebrick', width=3, dash='dash'),
            marker=dict(size=6, symbol='square'),
            hovertemplate='<b>Iteration %{x}</b><br>Average Fitness: %{y:.4f}<extra></extra>'
        ))

        # Add trace for 0.05 percentile fitness
        fig.add_trace(go.Scatter(
            x=generation_numbers,
            y=merged_percentile_fitness,
            mode='lines+markers',
            name='0.05 Percentile Fitness',
            line=dict(color='green', width=3, dash='dot'),
            marker=dict(size=6, symbol='triangle-up'),
            hovertemplate='<b>Iteration %{x}</b><br>0.05 Percentile Fitness: %{y:.4f}<extra></extra>'
        ))

        # Update layout to make the plot detailed and visually appealing
        fig.update_layout(
            title="Evolutionary Optimization History Across All Cycles",
            xaxis_title="Generation",
            yaxis_title="Fitness Value",
            legend=dict(
                x=0.75,
                y=0.15,
            ),
            hovermode='x unified',
            width=900,
            height=900,
        )

        # Show the plot
        fig.show()            
