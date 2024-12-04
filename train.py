import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from output import plot_state_norms, plot_gradient_norms, collect_gradient_norms


def train_low_energy_two_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda", test_mode=False, plot=False):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(num_epochs * len(train_loader)))
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0

        count = 0

        if plot:
            gradient_norms = {"encoder": [], "target_encoder": [], "predictor": []}

        for batch in tqdm(train_loader):
            states = batch.states.to(device)  # [B, T+1, Ch, H, W]
            actions = batch.actions.to(device)  # [B, T, action_dim]

            predicted_states, target_states = model(states, actions)
            loss = model.loss(predicted_states, target_states)
            optimizer.zero_grad()
            loss.backward()

            if plot:
                norms = collect_gradient_norms(model)
                for norm in norms:
                    gradient_norms[norm].append(norms[norm])

            optimizer.step()
            epoch_loss += loss.item()

            count = count + 1
            if test_mode and count > 10:
                return predicted_states, target_states
            if plot and count%2 == 0:
               plot_state_norms(predicted_states, target_states)
               plot_gradient_norms(gradient_norms)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.10f}")
    return predicted_states, target_states

