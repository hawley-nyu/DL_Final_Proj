import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from evaluator import ProbingEvaluator, ProbingConfig  

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, verbose=False):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
def train_low_energy_two_model(model, train_loader, num_epochs=50, learning_rate=1e-4,
                             device="cuda", test_mode=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    #earlystop
    early_stopping = EarlyStopping(patience=5, verbose=True)

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

        #model evaluation
        model.eval()
        prober = evaluator.train_pred_prober()
        avg_losses = evaluator.evaluate_all(prober=prober, device=device)

        # val loss
        val_loss = sum(avg_losses.values())

        print(f"Epoch {epoch + 1}, Validation Metrics:")
        for probe_attr, loss in avg_losses.items():
            print(f"- {probe_attr} loss: {loss}")

        # save optimal model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("save optimal model")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # update lr
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")

    return predicted_states, target_states
