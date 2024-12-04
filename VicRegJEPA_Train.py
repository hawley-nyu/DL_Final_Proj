import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def train_low_energy_two_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda", test_mode=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        count = 0

        for batch in train_loader:
            states = batch.states.to(device)  # [B, T+1, Ch, H, W]
            actions = batch.actions.to(device)  # [B, T, action_dim]

            predicted_states, target_states = model(states, actions)
            loss = model.loss(predicted_states, target_states)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count = count + 1

            if count % 200 == 0:
                print(f"Batch {count}, loss: {loss.item()}")

            if test_mode and count == 10:
                return predicted_states, target_states

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.10f}")

    return predicted_states, target_states