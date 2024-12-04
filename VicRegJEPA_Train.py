import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from output import print_sample
import matplotlib.pyplot as plt


def train_low_energy_two_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda", test_mode=False):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        count = 0
        gradient_norms = {"encoder": [], "target_encoder": [], "predictor": []}

        for batch in train_loader:
            states = batch.states.to(device)  # [B, T+1, Ch, H, W]
            actions = batch.actions.to(device)  # [B, T, action_dim]

            predicted_states, target_states = model(states, actions)
            loss = model.loss(predicted_states, target_states)
            optimizer.zero_grad()
            loss.backward()

            collect_gradient_norms = {"encoder": 0, "target_encoder": 0, "predictor": 0}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if "encoder" in name and "target_encoder" not in name:  # Encoder
                        collect_gradient_norms["encoder"] += grad_norm
                    elif "target_encoder" in name:  # Target Encoder
                        collect_gradient_norms["target_encoder"] += grad_norm
                    elif "predictor" in name:  # Predictor
                        collect_gradient_norms["predictor"] += grad_norm
            gradient_norms["encoder"].append(collect_gradient_norms["encoder"])
            gradient_norms["target_encoder"].append(collect_gradient_norms["target_encoder"])
            gradient_norms["predictor"].append(collect_gradient_norms["predictor"])

            optimizer.step()
            epoch_loss += loss.item()

            print(f'{count},',end="")
            count = count + 1
            if count%200 == 0:
                print(f"last batch loss: {loss.item()}")
                predicted_norms = torch.norm(predicted_states, dim=-1).view(-1).detach().cpu().numpy()
                target_norms = torch.norm(target_states, dim=-1).view(-1).detach().cpu().numpy()
                plt.figure(figsize=(8, 6))
                plt.hist(predicted_norms, bins=50, alpha=0.5, label="Predicted States")
                plt.hist(target_norms, bins=50, alpha=0.5, label="Target States")
                plt.legend()
                plt.title("Norm Distributions of Predicted and Target States")
                plt.xlabel("Norm")
                plt.ylabel("Frequency")
                plt.show()

                plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
                for part, norms in gradient_norms.items():
                    plt.plot(norms, label=part)
                plt.title("Gradient Norms Over Training")
                plt.xlabel("Iteration")
                plt.ylabel("Mean Gradient Norm")
                plt.legend()
                plt.show()
            if test_mode and count == 10:
                return predicted_states, target_states

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.10f}")
    return predicted_states, target_states