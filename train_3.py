import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T


# Gaussian Noise
def add_gaussian_noise(img, mean=0., std=0.01):
    # img: [C,H,W]
    noise = torch.randn_like(img) * std + mean
    return img + noise


# flip
augment_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),  # 50% chance flip
])


def augment(x):
    # x: [B, T, C, H, W]
    B, T, C, H, W = x.shape
    x_reshaped = x.view(B * T, C, H, W)

    augmented_frames = []
    for i in range(B * T):
        frame = x_reshaped[i]  # [C,H,W]
        # use PIL for transforms
        frame_pil = T.ToPILImage()(frame)
        # randomly flip
        frame_aug = augment_transforms(frame_pil)
        # back to Tensor
        frame_aug = T.ToTensor()(frame_aug)
        # Add noise
        frame_aug = add_gaussian_noise(frame_aug, mean=0., std=0.01)

        augmented_frames.append(frame_aug)

    # get to [B*T,C,H,W]
    augmented = torch.stack(augmented_frames, dim=0)
    # reshape back to [B,T,C,H,W]
    augmented = augmented.view(B, T, C, H, W)
    return augmented


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
            slices.append(data[b, i:i + seq_len])
    new_data = torch.stack(slices, dim=0)
    return new_data  # shape (B * num_slices, seq_len, ...)


def train_low_energy_two_model(model, train_loader, num_epochs=50, learning_rate=1e-4, device="cuda", test_mode=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(num_epochs * len(train_loader)))

    max_seq_len_states = 17  # Maximum sequence length for states
    sequence_lengths = [17, 17, 17]  # Corresponding to state lengths for action lengths 1,2,4,8,16

    # Epochs at which to change the sequence length
    # For example, if num_epochs is 50, we can change the length every 10 epochs
    change_points = [0, 3, 6]  # Epochs at which sequence length changes
    seq_len_schedule = {epoch: seq_len for epoch, seq_len in zip(change_points, sequence_lengths)}

    for epoch in tqdm(range(num_epochs), desc='Epochs'):

        current_seq_len_states = sequence_lengths[-1]  # Default to max sequence length
        for change_point, seq_len in zip(change_points, sequence_lengths):
            if epoch >= change_point:
                current_seq_len_states = seq_len
            else:
                break
        seq_len_states = current_seq_len_states  # Sequence length for states
        seq_len_actions = seq_len_states - 1  # Sequence length for actions

        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            states = batch.states.to(device)  # [B, T+1, Ch, H, W]
            actions = batch.actions.to(device)  # [B, T, action_dim]

            # Generate subsequences
            states_subseq = get_subsequences(states, seq_len_states)  # [B * num_slices, seq_len_states, 2, 65, 65]
            actions_subseq = get_subsequences(actions, seq_len_actions)  # [B * num_slices, seq_len_actions, action_dim]

            batch_size = states_subseq.shape[0]
            mini_batch_size = 128  # Adjust based on your GPU memory

            for i in range(0, batch_size, mini_batch_size):
                states_mini = states_subseq[i:i + mini_batch_size]
                actions_mini = actions_subseq[i:i + mini_batch_size]

                # main loss
                predicted_states, target_states, encoded_wall = model(states_mini, actions_mini)
                loss_pred = model.loss(predicted_states, target_states, encoded_wall)

                # BYOL loss
                view1 = augment(states_mini)
                view2 = augment(states_mini)
                loss_byol = model.byol_loss(view1, view2)

                # Combine loss
                loss_total = loss_pred + loss_byol

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                # Update target NN with EMA
                model.update_target_network()

                epoch_loss += loss_total.item()

            progress_bar.update(1)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.10f}")
    return predicted_states, target_states
