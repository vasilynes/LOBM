import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from efficient_kan import KAN
from src.data import table
from pathlib import Path
import xgboost as xgb
from datetime import datetime
from training.xgboost.train import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
import json

BATCH_SIZE = 4096

def generate_base_preds(X_train, X_val, X_test):
    print('Loading XGBoost model...')
    xgb_results_dir = Path('experiments/xgboost/run_20260429_093100')
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_results_dir / 'model.json')
    
    preds_train = xgb_model.predict(X_train)
    preds_val = xgb_model.predict(X_val)
    preds_test = xgb_model.predict(X_test)

    return preds_train, preds_val, preds_test

def get_loaders(X_train, X_val, y_train, y_val, preds_train, preds_val):
    print('Calculating residuals...')
    res_train = y_train - preds_train
    res_val = y_val - preds_val

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    R_train_t = torch.tensor(res_train, dtype=torch.float32).unsqueeze(1)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    R_val_t = torch.tensor(res_val, dtype=torch.float32).unsqueeze(1)

    print('Constructing train and val loaders...')
    train_dataset = TensorDataset(X_train_t, R_train_t)
    val_dataset = TensorDataset(X_val_t, R_val_t)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader

class TinyKAN:
    def __init__(
            self, 
            input_dim, 
            train_loader, 
            val_loader
        ):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = nn.MSELoss()
        self.kan_model = KAN([input_dim, 6, 1], grid_size=5, spline_order=3).to(self.device)
        self.optimizer = optim.Adam(self.kan_model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        self.epochs = 20
 
    def train(self, kan_results_dir):
        best_val_loss = float('inf')

        losses = {
            'epochs': self.epochs,
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(self.epochs):
            self.kan_model.train()
            train_loss = 0.0
            for X_batch, R_batch in self.train_loader:
                X_batch, R_batch = X_batch.to(self.device), R_batch.to(self.device)
                X_batch = torch.tanh(X_batch)
                self.optimizer.zero_grad()
                pred_res = self.kan_model(X_batch)
                loss = self.loss_fn(pred_res, R_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.kan_model.parameters(), max_norm=1.0)

                self.optimizer.step()
                train_loss += loss.item() * X_batch.size(0) 
            train_loss /= len(self.train_loader.dataset)
            losses['train_loss'].append(train_loss)

            self.kan_model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for X_batch, R_batch in self.val_loader:
                    X_batch, R_batch = X_batch.to(self.device), R_batch.to(self.device)
                    X_batch = torch.tanh(X_batch)
                    val_pred = self.kan_model(X_batch)
                    loss = self.loss_fn(val_pred, R_batch)
                    val_loss += loss.item() * X_batch.size(0)
                val_loss /= len(self.val_loader.dataset)
                losses['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs} - Train MSE: {train_loss:.4f} - Val MSE: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.kan_model.state_dict(), kan_results_dir / 'model.pth')
    
        return losses
        
    def get_correction(self, X, kan_model_dir=None):
        if kan_model_dir is None:
            raise ValueError('Provide model directory.')
        self.kan_model.load_state_dict(
            torch.load(kan_model_dir / 'model.pth')
        )
        self.kan_model.eval()  
        with torch.inference_mode():
            X = torch.tanh(X)
            kan_correction = self.kan_model(X.to(self.device)).cpu().numpy().squeeze()

        return kan_correction

def main():
    results = {}
    print('Loading datasets...')
    X_train, y_train = table.tabularize(TRAIN_SPLIT)
    X_val, y_val = table.tabularize(VAL_SPLIT)
    X_test, y_test = table.tabularize(TEST_SPLIT)

    print('Generating XGBoost predictions...')
    preds_train, preds_val, preds_test = generate_base_preds(X_train, X_val, X_test)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_loader, val_loader = get_loaders(X_train, X_val, y_train, y_val, preds_train, preds_val)
    print('Dataloaders for training and validation are created.')

    print('Initializing training...')
    input_dim = X_train.shape[1]
    tiny_kan = TinyKAN(
        input_dim, 
        train_loader, 
        val_loader
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    kan_results_dir = Path(f"experiments/kan/run_{timestamp}")
    kan_results_dir.mkdir(parents=True, exist_ok=True)

    losses = tiny_kan.train(kan_results_dir)
    with open(kan_results_dir / 'training_log.json', 'w') as f:
            json.dump(losses, f, indent=2)
            
    print('Training is done.')

    print('Testing XGBoost+KAN ensemble...')

    kan_correction = tiny_kan.get_correction(X_test_t, kan_results_dir)

    refined_preds = preds_test + kan_correction
    res = y_test - refined_preds

    refined_mse = mean_squared_error(y_test, refined_preds)
    refined_mae = mean_absolute_error(y_test, refined_preds)

    non_zero_idx = (y_test != 0)
    refined_dir_acc = np.mean(np.sign(refined_preds[non_zero_idx]) == np.sign(y_test[non_zero_idx])) * 100

    results.update({
        'timestamp': timestamp,
        'loss_fn': str(tiny_kan.loss_fn),
        'test_metrics': {
            'mse': float(refined_mse),
            'mae': float(refined_mae),
            'directional_accuracy': float(refined_dir_acc),
            'residual_mean': float(res.mean()),
            'residual_std': float(res.std()),
            'residual_min': float(res.min()),
            'residual_max': float(res.max())
        },
        'dataset_info': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
    })

    with open(kan_results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)

        with open(kan_results_dir / 'summary.txt', 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write('='*50 + "\n\n")

            f.write(f"  loss fn: {tiny_kan.loss_fn}\n\n")
        
            f.write('Test Metrics:\n')
            f.write(f"  MSE: {refined_mse:.6f}\n")
            f.write(f"  MAE: {refined_mae:.6f}\n")
            f.write(f"  Directional Accuracy: {refined_dir_acc:.2f}%\n")
            f.write(f"  Residual Mean: {res.mean():.6f}\n")
            f.write(f"  Residual Std: {res.std():.6f}\n\n")
            
            f.write(f'Dataset Sizes:\n')
            f.write(f"  Train: {len(X_train)}\n")
            f.write(f"  Val: {len(X_val)}\n")
            f.write(f"  Test: {len(X_test)}\n")

    print(f"Ensemble Residual Mean: {res.mean():.4f}")
    print(f"Ensemble Residual Std: {res.std():.4f}")

    print(f"Ensemble Test MSE: {refined_mse:.4f}")
    print(f"Ensemble Test MAE: {refined_mae:.4f}")
    print(f"Directional Accuracy: {refined_dir_acc:.2f}%")

if __name__ == '__main__':
    main()