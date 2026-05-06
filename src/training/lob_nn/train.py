import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import get_dataset
from src.models.lob_nn import LOB_NN
from pathlib import Path
from datetime import datetime
import os

class TrainingNN:
    def __init__(
            self, 
            model, 
            train_loader, 
            val_loader, 
            epochs, 
            device,
            lr=1e-4,
            weight_decay=1e-4
        ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.model = model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.loss_fn = nn.HuberLoss(delta=1.0)

    def initiate(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        lob_nn_results_dir = Path(f"experiments/lob_nn/run_{timestamp}")
        lob_nn_results_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            n_samples = 0   # Track sample count, since direct len() on IterableDataset is impossible
            for lob_seq, global_seq, target_bps in self.train_loader:
                lob_seq = lob_seq.to(self.device, non_blocking=True)
                global_seq = global_seq.to(self.device, non_blocking=True)
                target_bps = target_bps.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                pred_res = self.model(lob_seq, global_seq)
                loss = self.loss_fn(pred_res, target_bps)
                loss.backward()  
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                self.optimizer.step()
                batch_size = lob_seq.size(0)
                train_loss += loss.item() * batch_size
                n_samples += batch_size
            train_loss /= n_samples
            
            self.model.eval()
            val_loss = 0.0
            n_samples_val = 0
            with torch.inference_mode():
                for lob_seq, global_seq, target_bps in self.val_loader:
                    lob_seq = lob_seq.to(self.device, non_blocking=True)
                    global_seq = global_seq.to(self.device, non_blocking=True)
                    target_bps = target_bps.to(self.device, non_blocking=True)
                    pred_res = self.model(lob_seq, global_seq)
                    loss = self.loss_fn(pred_res, target_bps)
                    batch_size_val = lob_seq.size(0)
                    val_loss += loss.item() * batch_size_val
                    n_samples_val += batch_size_val
                val_loss /= n_samples_val

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), lob_nn_results_dir / 'model.pth')
                
            self.scheduler.step()


if __name__ == '__main__':
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_d = get_dataset('2026-04-26', 'train')
    val_d = get_dataset('2026-04-26', 'val')
    train_loader = DataLoader(
        train_d, 
        batch_size=None, 
        shuffle=False, 
        pin_memory=True,
        num_workers=min(4, os.cpu_count() or 1),
    )
    val_loader = DataLoader(
        val_d, 
        batch_size=None, 
        shuffle=False, 
        pin_memory=True,
        num_workers=0   # Enforce sequential processing of validation batches
    )

    model = LOB_NN()

    training = TrainingNN(
        model, train_loader, val_loader, epochs, device
    )
    training.initiate()
                
            


