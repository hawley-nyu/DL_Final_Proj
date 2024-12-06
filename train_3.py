import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms import ToTensor, ToPILImage
from typing import Optional, Tuple
import gc


class BatchAugmenter:
    def __init__(self, device: str = 'cuda', batch_size: int = 32, noise_std: float = 0.01):
        self.device = device
        self.batch_size = batch_size
        self.noise_std = noise_std

        # Initialize transforms
        self.augment_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
        ])
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

    @torch.no_grad()  # Disable gradient computation for efficiency
    def add_gaussian_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to a batch of images efficiently."""
        noise = torch.randn_like(img, device=self.device) * self.noise_std
        return img + noise

    @torch.no_grad()
    def process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Process a batch of frames efficiently."""
        augmented = []
        for frame in batch:
            # Convert to PIL and apply transforms
            frame_pil = self.to_pil(frame.cpu())
            frame_aug = self.augment_transforms(frame_pil)
            # Convert back to tensor
            frame_tensor = self.to_tensor(frame_aug).to(self.device)
            augmented.append(frame_tensor)

        # Stack batch results
        augmented = torch.stack(augmented, dim=0)
        # Add noise to entire batch at once
        augmented = self.add_gaussian_noise(augmented)
        return augmented

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augment a batch of sequences efficiently.
        Args:
            x: Input tensor of shape [B, T, C, H, W]
        Returns:
            Augmented tensor of same shape
        """
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        total_frames = B * T
        augmented_frames = []

        # Process in batches
        for start_idx in range(0, total_frames, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_frames)
            batch = x_reshaped[start_idx:end_idx]
            aug_batch = self.process_batch(batch)
            augmented_frames.append(aug_batch)

        # Combine all batches and reshape
        augmented = torch.cat(augmented_frames, dim=0)
        augmented = augmented.view(B, T, C, H, W)

        return augmented


def augment(x: torch.Tensor, device: str = 'cuda', batch_size: Optional[int] = 32) -> torch.Tensor:
    """
    Wrapper function for backwards compatibility
    """
    augmenter = BatchAugmenter(device=device, batch_size=batch_size)
    return augmenter(x)

def get_subsequences(data, seq_len, max_slices=None):
    B, T = data.shape[:2]
    num_slices = min(T - seq_len + 1, max_slices) if max_slices else T - seq_len + 1
    slices = []
    for b in range(B):
        for i in range(num_slices):
            slices.append(data[b, i:i + seq_len])
    new_data = torch.stack(slices, dim=0)
    return new_data


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
            mini_batch_size = 32  # Adjust based on your GPU memory

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
                # record loss and del paras
                current_loss = loss_total.item()
                epoch_loss += current_loss

                #release unneeded paras
                del loss_pred, loss_byol, loss_total, view1, view2
                del predicted_states, target_states, encoded_wall

            if batch_idx % 10 == 0:  # 每10个batch清理一次
                torch.cuda.empty_cache()
                gc.collect()

                # test memory
                print(
                    f'Batch {batch_idx}, GPU memory allocated: {torch.cuda.memory_allocated(device=device) / 1024 ** 2:.2f}MB')

            progress_bar.update(1)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.10f}")

        # each epoch reset memory
        torch.cuda.empty_cache()
        gc.collect()

        print(f'End of epoch {epoch+1}, GPU memory allocated: {torch.cuda.memory_allocated(device=device)/1024**2:.2f}MB')
        print(f'End of epoch {epoch+1}, GPU memory reserved: {torch.cuda.memory_reserved(device=device)/1024**2:.2f}MB')

    return predicted_states, target_states
