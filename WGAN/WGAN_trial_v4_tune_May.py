import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import wgan

def main():
    # Set the working directory to the location of the WGAN code and dataset
    os.chdir('/project2/macs30123/DLM/torch/ds-wgan-master')

    # Load and preprocess the data
    data = pd.read_csv('Xorigin.csv')  # Load the dataset containing various variables
    data.dropna(axis=1, how='all', inplace=True)  # Remove columns that contain only NaN values

    # Define variable specifications in the dataset
    continuous_vars = []
    continuous_lower_bounds = {}
    continuous_upper_bounds = {}
    categorical_vars = ["citizen", "marst", "sex", "hispan", "sect", "state", "race", "health", "famsize", "schoolyr", "age"]
    context_vars = []

    # Initialize the WGAN data wrapper
    data_wrapper = wgan.DataWrapper(data, continuous_vars, categorical_vars, context_vars,
                                    continuous_lower_bounds, continuous_upper_bounds)

    # Set up WGAN training specifications
    specs = wgan.Specifications(data_wrapper, batch_size=4096, max_epochs=2000, critic_lr=1e-3,
                                generator_lr=1e-4, print_every=1, device="cuda")

    # Create generator and critic networks
    generator = wgan.Generator(specs).to(specs.device)
    critic = wgan.Critic(specs).to(specs.device)

    # Prepare data for training
    x, context = data_wrapper.preprocess(data)
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(context, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=specs.batch_size, shuffle=True, num_workers=4)

    # Start the training process
    train_wgan(generator, critic, loader, specs)

    # Generate and compare data
    generate_and_compare(generator, data_wrapper)

def train_wgan(generator, critic, data_loader, specifications):
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=specifications.generator_lr)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=specifications.critic_lr)
    scaler = GradScaler()

    for epoch in range(specifications.max_epochs):
        for inputs, contexts in data_loader:
            inputs, contexts = inputs.to(specifications.device), contexts.to(specifications.device)
            optimizer_critic.zero_grad()

            # Mixed precision training
            with autocast():
                fake_data = generator(contexts)
                critic_loss = critic(fake_data).mean() - critic(inputs).mean()
                critic_loss += specifications.critic_gp_factor * critic.gradient_penalty(inputs, fake_data, contexts)

            scaler.scale(critic_loss).backward()
            scaler.step(optimizer_critic)
            optimizer_critic.zero_grad()

            # Update generator every n_critic iterations
            if epoch % specifications.critic_steps == 0:
                with autocast():
                    fake_data = generator(contexts)
                    gen_loss = -critic(fake_data).mean()

                scaler.scale(gen_loss).backward()
                scaler.step(optimizer_gen)
                optimizer_gen.zero_grad()

            scaler.update()

        if epoch % specifications.print_every == 0:
            print(f"Epoch {epoch}, Critic Loss: {critic_loss.item()}, Generator Loss: {gen_loss.item()}")

def generate_and_compare(generator, data_wrapper, sample_size=1000):
    sampled_data = data.sample(sample_size)
    generated_data = data_wrapper.apply_generator(generator, sampled_data)
    wgan.compare_dfs(data, generated_data, 
                     scatterplot=dict(x=["citizen", "marst", "sex", "hispan", "age", "schoolyr"], samples=1000, smooth=0),
                     histogram=dict(variables=["sex", "hispan", "age", "schoolyr", "state", "health", "sect", "famsize", "marst", "bpl", "citizen", "race"], nrow=4, ncol=3),
                     figsize=5)

if __name__ == "__main__":
    main()
