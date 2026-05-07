import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import get_dataset
from src.models.lob_nn import LOB_NN
from pathlib import Path
from datetime import datetime
import os
import wandb

GRAD_MAX_NORM = 1.0
BATCH = 1024

class TrainingNN:
    def __init__(
            self, 
            model,
            results_dir, 
            train_loader, 
            val_loader, 
            epochs, 
            device,
            lr=1e-4,
            weight_decay=1e-4,
            delta=1.0
        ):
        self.results_dir = results_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.model = model.to(self.device)
        self.weight_decay = weight_decay
        self.lr = lr

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.loss_fn = nn.HuberLoss(delta=delta)

    @staticmethod
    def _attn_stats(attn: torch.Tensor) -> dict[str, float]:
        """
        Compute per-batch attention diagnositcs.
        """
        a = attn.squeeze(dim=-1)
        max_attn = a.max(dim=1).values.mean().item()
        entropy = -(a * torch.log(a + 1e-8)).sum(dim=1).mean().item()
        return {'max_attn': max_attn, 'entropy': entropy}

    def initiate(self):
        best_val_loss = float('inf')
        global_step = 0
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            n_samples = 0   # Track sample count, since direct len() on IterableDataset is impossible
            for lob_seq, global_seq, target_bps in self.train_loader:
                lob_seq = lob_seq.to(self.device, non_blocking=True)
                global_seq = global_seq.to(self.device, non_blocking=True)
                target_bps = target_bps.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                pred_res, _ = self.model(lob_seq, global_seq)
                loss = self.loss_fn(pred_res, target_bps)
                loss.backward()  
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRAD_MAX_NORM)
                
                wandb.log(
                    {
                        'step/loss': loss.item(),
                        'step/grad_total_norm': total_norm.item(),
                        **{f"step/{k}": v for k, v in self.model.diagnose().items()}
                    },
                    step=global_step
                )
                global_step += 1
                self.optimizer.step()

                batch_size = lob_seq.size(0)
                train_loss += loss.item() * batch_size
                n_samples += batch_size

            train_loss /= n_samples

            self.model.eval()
            val_loss = 0.0
            n_samples_val = 0
            attn_stats = {'max_attn': 0.0, 'entropy': 0.0}
            epoch_attn_profile = None
            with torch.inference_mode(): 
                for lob_seq, global_seq, target_bps in self.val_loader:
                    lob_seq = lob_seq.to(self.device, non_blocking=True)
                    global_seq = global_seq.to(self.device, non_blocking=True)
                    target_bps = target_bps.to(self.device, non_blocking=True)

                    pred_res, attn = self.model(lob_seq, global_seq)
                    loss = self.loss_fn(pred_res, target_bps)

                    batch_size_val = lob_seq.size(0)
                    val_loss += loss.item() * batch_size_val
                    n_samples_val += batch_size_val

                    attn_cpu = attn.detach().cpu()
                    batch_stats = self._attn_stats(attn_cpu)
                    for k in attn_stats:
                        attn_stats[k] += batch_stats[k] * batch_size_val

                    profile = attn_cpu.squeeze(-1).sum(dim=0)
                    epoch_attn_profile = profile if epoch_attn_profile is None else epoch_attn_profile + profile

            val_loss /= n_samples_val
            for k in attn_stats:
                attn_stats[k] /= n_samples_val
            epoch_attn_profile = epoch_attn_profile / n_samples_val

            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/lr': self.scheduler.get_last_lr()[0],
                'val/loss': val_loss,
                'val/attn_max_mean': attn_stats['max_attn'],
                'val/attn_entropy': attn_stats['entropy'],
                'val/attn_profile': wandb.plot.line_series(
                    xs=list(range(len(epoch_attn_profile))),
                    ys=[epoch_attn_profile.tolist()],
                    keys=['attention'],
                    title='Attention Profile',
                    xname='Time step'
                )
            }, step=global_step)

            print(
                f"Epoch {epoch+1}/{self.epochs} "
                f"- train {train_loss:.4f} - val {val_loss:.4f} "
                f"- attn max {attn_stats['max_attn']:.4f} "
                f"- attn entropy {attn_stats['entropy']:.4f}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.results_dir / 'model.pth')
                
            self.scheduler.step()


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(f"experiments/lob_nn/run_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    lr = 1e-4,
    weight_decay = 1e-4,
    delta = 1.0
    epochs = 10

    wandb.init(project='LOBM', name=f"run_{timestamp}", dir=str(results_dir))
    wandb.config.update({
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'batch_size': BATCH,
        'loss_fn': 'HuberLoss',
        'loss_delta': delta,
        'scheduler': 'CosineAnnealingLR',
        'grad_clip': GRAD_MAX_NORM,
    })

    print('Starting training...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_d = get_dataset('2026-04-26', 'train')
    val_d = get_dataset('2026-04-26', 'val')
    train_loader = DataLoader(
        train_d, 
        batch_size=None, 
        shuffle=False, 
        pin_memory=True,
        num_workers=min(4, os.cpu_count() or 1),
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_d, 
        batch_size=None, 
        shuffle=False, 
        pin_memory=True,
        num_workers=1,   # Enforce sequential processing of validation batches
        persistent_workers=True
    )

    model = LOB_NN()

    training = TrainingNN(
        model, results_dir, train_loader, val_loader, epochs, device
    )
    training.initiate()
    wandb.finish()
    print('Training finished.')
                
        