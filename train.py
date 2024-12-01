import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from output import print_sample
import matplotlib.pyplot as plt

def train_low_energy_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        #printed = False

        for batch in train_loader:
            unnormalized_states = batch.states.to(device, non_blocking=True)  # [B, T, Ch, H, W]
            actions = batch.actions.to(device, non_blocking=True)  # [B, T-1, 2]

            states = unnormalized_states / unnormalized_states.max()  # Scale to [0, 1]

            #if not printed:
            #    print_sample(states[60])
            #    printed = True

            predictions = model(states, actions)  # [B, T, D]

            predicted_next_states = predictions[:, :-1, :]  # [B, T-1, D]
            predicted_last_state = predictions[:, -1, :]  # [B, D]

            # is this permitted?
            true_next_states = model.state_encoder(states[:, 1:, :, :, :].contiguous().view(-1, *states.shape[2:]))
            true_next_states = true_next_states.view_as(predicted_next_states)

            positive_energy = torch.norm(predicted_next_states - true_next_states, dim=2)
            shuffled_actions = actions[torch.randperm(actions.size(0))]
            shuffled_predictions = model(states, shuffled_actions)[:, :-1, :]
            negative_energy = torch.norm(shuffled_predictions - true_next_states, dim=2)

            contrastive_loss = F.relu(positive_energy - negative_energy + model.margin).mean()
            predictive_loss = mse_loss(predicted_next_states, true_next_states)
            loss = contrastive_loss + predictive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


def train_low_energy_two_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        count = 0
        #gradient_norms = []
        gradient_norms = {"encoder": [], "target_encoder": [], "predictor": []}

        for batch in train_loader:
            observations = batch.states.to(device)  # [B, T+1, Ch, H, W]
            actions = batch.actions.to(device)  # [B, T, action_dim]

            #observations[:, :, 0, :, :] *= 1000
            predicted_states, target_states = model(observations, actions)
            #augmented1 = observations + (torch.randn_like(observations) * 5)
            #if torch.rand(1).item() < 0.5:
            #    augmented2 = augmented1.flip(-1)
            #else:
            #    augmented2 = augmented1
            #predicted_states, target_states = model(augmented2, actions)

            loss = model.loss(predicted_states, target_states)

            #if count == 10:
            #    plt.plot(predicted_states[0,0].detach().cpu().numpy(), label="Predicted")
            #    plt.plot(target_states[0,0].detach().cpu().numpy(), label="Target")
            #    plt.legend()
            #    plt.show()

            optimizer.zero_grad()
            loss.backward()
            #for name, param in model.named_parameters():
            #    print(f"{name}: {param.data.norm()} -> {param.data.norm()}")
            #for param in model.parameters():
            #    if param.grad is not None:
            #        print(f"Gradient norm: {param.grad.norm()}")
            #    else:
            #        print("No gradient computed for this parameter.")

            #batch_grad_norm = sum(param.grad.norm().item() for param in model.parameters() if param.grad is not None) / len(list(model.parameters()))
            #gradient_norms.append(batch_grad_norm)

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

                #plt.plot(gradient_norms)
                plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
                for part, norms in gradient_norms.items():
                    plt.plot(norms, label=part) 
                plt.title("Gradient Norms Over Training")
                plt.xlabel("Iteration")
                plt.ylabel("Mean Gradient Norm")
                plt.legend()
                plt.show()
            #if count == 400:
            #    break


        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.10f}")
        return predicted_states, target_states
