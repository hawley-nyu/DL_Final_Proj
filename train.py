import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from output import plot_state_norms, plot_gradient_norms, collect_gradient_norms

def get_subsequences(data, seq_len):
    """
    Generates all possible subsequences of length seq_len from the input data.
    Args:
        data (Tensor): Input tensor of shape (B, T, ...).
        seq_len (int): Desired sequence length.
    Returns:
        Tensor: Subsequence tensor of shape (B * num_slices, seq_len, ...).
    """
    B, T = data.shape[:2]
    num_slices = T - seq_len + 1
    slices = []
    for b in range(B):
        for i in range(num_slices):
            slices.append(data[b, i:i+seq_len])
    new_data = torch.stack(slices, dim=0)
    return new_data  # shape (B * num_slices, seq_len, ...)


def train_low_energy_two_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda", test_mode=False, plot=False):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(num_epochs * len(train_loader)))

    max_seq_len_states = 17  # Maximum sequence length for states
    sequence_lengths = [3, 9, 17]  # Corresponding to state lengths for action lengths 1,2,4,8,16

    # Epochs at which to change the sequence length
    change_points = [0, 3, 6]  # Epochs at which sequence length changes
    seq_len_schedule = {epoch: seq_len for epoch, seq_len in zip(change_points, sequence_lengths)}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        count = 0

        if plot:
            gradient_norms = {"encoder": [], "target_encoder": [], "predictor": []}

        current_seq_len_states = sequence_lengths[-1]  # Default to max sequence length
        for change_point, seq_len in zip(change_points, sequence_lengths):
            if epoch >= change_point:
                current_seq_len_states = seq_len
            else:
                break
        seq_len_states = current_seq_len_states  # Sequence length for states
        seq_len_actions = seq_len_states - 1     # Sequence length for actions

        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            states = batch.states.to(device)  # [B, T+1, Ch, H, W]
            actions = batch.actions.to(device)  # [B, T, action_dim]

            # Generate subsequences
            states_subseq = get_subsequences(states, seq_len_states)      # [B * num_slices, seq_len_states, 2, 65, 65]
            actions_subseq = get_subsequences(actions, seq_len_actions)    # [B * num_slices, seq_len_actions, 2, 65, 65]

            batch_size = states_subseq.shape[0]
            mini_batch_size = 128  # Adjust based on your GPU memory

            for i in range(0, batch_size, mini_batch_size):
                states_mini = states_subseq[i:i+mini_batch_size]
                actions_mini = actions_subseq[i:i+mini_batch_size]

                predicted_states, target_states, encoded_wall = model(states_mini, actions_mini)
                loss = model.loss(predicted_states, target_states, encoded_wall)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if plot:
                norms = collect_gradient_norms(model)
                for norm in norms:
                    gradient_norms[norm].append(norms[norm])

            optimizer.step()
            epoch_loss += loss.item()

            count = count + 1
            if test_mode and count > 10:
                return predicted_states, target_states
            if plot and count%200 == 0:
               plot_state_norms(predicted_states, target_states)
               plot_gradient_norms(gradient_norms)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.10f}")
    return predicted_states, target_states

