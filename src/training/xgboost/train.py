import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from datetime import datetime
from pathlib import Path
from src.data import table
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Script for training XGBRegressor model')
    parser.add_argument('--params', '-p', required=True, help='Path to params.yaml for the model')
    parser.add_argument('--date', '-d', required=True, help='Date of the splits folder')

    return parser.parse_args()

args = parse_args()

SPLITS = Path(f"data/splits/{args.date}")
TRAIN_SPLIT = SPLITS / 'train/train.parquet'
VAL_SPLIT = SPLITS / 'val/val.parquet'
TEST_SPLIT = SPLITS / 'test/test.parquet' 

def train_model(X_train, y_train, X_val, y_val, **kwargs):
    model = xgb.XGBRegressor(**kwargs)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    return model

def main(params: dict, date):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(f"experiments/xgboost/{date}/run_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = table.tabularize(TRAIN_SPLIT)
    X_val, y_val = table.tabularize(VAL_SPLIT)

    print('Starting training...')

    objective = params.pop('objective')
    print(f"Training with {objective}")
    model = train_model(X_train, y_train, X_val, y_val, objective=objective, **params)

    print('Training finished.')
    print('Saving training logs...')

    eval_results = {}
    if hasattr(model, 'evals_result_'):
        eval_results = model.evals_result()
        with open(results_dir / 'training_log.json', 'w') as f:
            json.dump(eval_results, f, indent=2)

    print('Saving model artifact...')
    model.save_model(results_dir / 'model.json')

    train_size = len(X_train)
    val_size = len(X_val)

    del X_train, y_train
    del X_val, y_val

    print('Testing...')
    X_test, y_test = table.tabularize(TEST_SPLIT)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    non_zero_idx = (y_test != 0)
    dir_acc = np.mean(np.sign(preds[non_zero_idx]) == np.sign(y_test[non_zero_idx])) * 100
    res = y_test - preds

    results = {
        'timestamp': timestamp,
        'model_params': params,
        'objective': objective,
        'test_metrics': {
            'mse': float(mse),
            'mae': float(mae),
            'directional_accuracy': float(dir_acc),
            'residual_mean': float(res.mean()),
            'residual_std': float(res.std()),
            'residual_min': float(res.min()),
            'residual_max': float(res.max())
        },
        'dataset_info': {
            'date': date,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': len(X_test)
        }
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)

    with open(results_dir / 'summary.txt', 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write('='*50 + "\n\n")
        
        f.write('Model Parameters:\n')
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  objective: {objective}\n\n")
        
        f.write('Test Metrics:\n')
        f.write(f"  MSE: {mse:.6f}\n")
        f.write(f"  MAE: {mae:.6f}\n")
        f.write(f"  Directional Accuracy: {dir_acc:.2f}%\n")
        f.write(f"  Residual Mean: {res.mean():.6f}\n")
        f.write(f"  Residual Std: {res.std():.6f}\n\n")
        
        f.write(f'Dataset Sizes:\n')
        f.write(f"  Train: {train_size}\n")
        f.write(f"  Val: {val_size}\n")
        f.write(f"  Test: {len(X_test)}\n")
    
    print(f"Residual Mean: {res.mean():.4f}")
    print(f"Residual Std: {res.std():.4f}")

    print(f"XGBoost Test MSE: {mse:.4f}")
    print(f"XGBoost Test MAE: {mae:.4f}")
    print(f"Directional Accuracy: {dir_acc:.2f}%")

    plt.hist(res, bins=100, range=(-10, 10))
    plt.title(f"XGBoost Residuals ({objective})")
    plt.savefig(results_dir / 'residuals_plot.png')

if __name__ ==  '__main__':
    with open(args.params) as f:
        params = yaml.safe_load(f)

    main(params, args.date)


