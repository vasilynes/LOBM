## Limit Order Book modeling
### Overview
The project explores an architecture for modeling short-term price movements in Limit Order Book (LOB) data.
The architecture is two-stage: the Gradient Boosted Tree (XGBoost) suggests an estimate and the Kolmogorov-Arnold Network (KAN) corrects it. 

The architecture is benchmarked against a spatial-temporal sequence model being just the Convolutional Neural Network (CNN) followed by the Gated Recurrent Unit (GRU).
### Motivation
In financial time-series forecasting on LOB data, tree-based models often outperform deep models. 
[Wang (2025)](https://arxiv.org/abs/2506.05764) and [Liu et al. (2021)](https://www.ijcai.org/proceedings/2020/628) demonstrated, that proper feature engineering and denoising allowed XGBoost to surpass a CNN-based architecture.
This can be explained by the fact, that observed LOB features are discrete and contain a lot of microstructural noise.
High-frequency non-informative features hurt MLP-like deep learners and data non-smoothness is against their inductive bias, as indicated in [Grinsztajn (2022)](https://arxiv.org/abs/2207.08815).

Despite robustness, XGBoost are step-functions, hence continuous dynamics extrapolation is against their design. However, latent market dynamics (e.g., fair value, liquidity depletion) is continuous.
To bridge the gap, the project implements an ensemble: XGBoost as a base learner followed by KAN smooth corrector, trained on residuals from the base learner. 
Unlike standard MLPs, KAN here suggests two advantages:
1. The network is interpretable,
2. Its inductive bias is smoothness, as B-splines are theoretically optimal for approximating smooth, non-linear functions.

### Data & Feature Engineering
To train the models, BTCUSDT spot is sampled every 100 ms for 26 hours; a tabular dataset is constructed from this data with the following features:
* bid/Ask price and quantity for 20 levels
* per-level OFI for the first 10 levels
* mid price
* spread
#### Normalization
All the columns, excluding mid price, are then z-score normalized with a rolling window.

Macro statistics are calculated on level-1 bid quantity to estimate the optimal rolling window size for normalization.
```
>>> from src.data.helpers import norm_window_size
>>> norm_window_size(
>>> 	'data/books/2026-04-26/lob20.parquet',
>>> 	windows=[1_000, 10_000, 50_000, 100_000, 200_000],
>>> 	cols=['bid_q0']
>>> )

Window: 1000
    Column = bid_q0:
    Macro Mean:     0.0392 (ref: ~0.0)
    Macro Std:      1.2478 (ref: ~1.0)
    Outliers |Z|>3: 2.33% (ref: 1% - 5%)

Window: 10000
    Column = bid_q0:
    Macro Mean:     0.0127 (ref: ~0.0)
    Macro Std:      1.0683 (ref: ~1.0)
    Outliers |Z|>3: 1.29% (ref: 1% - 5%)

Window: 50000
    Column = bid_q0:
    Macro Mean:     0.0116 (ref: ~0.0)
    Macro Std:      1.0740 (ref: ~1.0)
    Outliers |Z|>3: 0.96% (ref: 1% - 5%)

Window: 100000
    Column = bid_q0:
    Macro Mean:     -0.0066 (ref: ~0.0)
    Macro Std:      0.9901 (ref: ~1.0)
    Outliers |Z|>3: 0.78% (ref: 1% - 5%)

Window: 200000
    Column = bid_q0:
    Macro Mean:     -0.0213 (ref: ~0.0)
    Macro Std:      1.0268 (ref: ~1.0)
    Outliers |Z|>3: 0.87% (ref: 1% - 5%)
```
For `>=50_000`, the window is too wide, the outliers drop to `<1.0`, the signal becomes weak. 

For `1000`, the value for macro std is too big, being almost 25% higher than for the standard $Z$. 

`10_000` is the most appropriate, the number of outliers is high enough and the statistics are close to $Z$. This window size corresponds to an around 16.7 minute interval. 
#### Target Calculation
The target is calculated from the mid price and expressed as basis points of return over a specified time horizon.

With 100ms ticks, it's reasonable to expect "constant" horizons, where many targets are zero. 
This may lead to data imbalance, where the model learns that &laquo;no change&raquo; prediction is the best one. 

To find the optimal horizon, the returns are calculated with different horizons:
```
>>> from src.data.helpers import test_horizons  
>>> test_horizons(
>>> 	'data/books/2026-04-26/lob20.parquet', 
>>> 	horizons=[10, 50, 100, 500], 
>>> 	price_col='mid'
>>> )  
Horizon 10: 86.82% of targets are strictly zero.  
Horizon 50: 62.45% of targets are strictly zero.  
Horizon 100: 47.33% of targets are strictly zero.  
Horizon 500: 13.89% of targets are strictly zero.
```
`horizon=100` leads to almost 50/50 balance.
### Model Architectures
#### CNN-GRU
The deep learner is a DeepLOB-like model consisting of spatial (per-level) and temporal (per-tick) convolutions, with a GRU layer of hidden dimension 64 and subsequent tanh-attention on the hidden states.
The final linear combination of hidden states weighted by attention is then fed to 2 dense layers to obtain a prediction. 
The model is then trained with Huber loss. For more details on architecture and training, see `training_lob_nn.ipynb` notebook.
#### XGBRegressor
The tree-based model consists of up to 200 weak learners with maximal depth of 3. 
It is then trained with early stopping (patience 50), learning rate 0.01, subsampling and L2 regularization on weights.
For details on parameters, see `src/training/xgboost/params.yaml`.
#### KAN
The network uses an efficient [implementation](https://github.com/Blealtan/efficient-kan) of KAN with L1 weight regularization and no pruning. 
The networked is trained for 20 epochs with early stopping on (shuffled) residuals of XGBRegressor predictions.
The resulting dimensions are $92 \rightarrow 6 \rightarrow 1$.
### Results
The resulting directional accuracy of CNN-GRU model is $61\%$.

After training XGBRegressor with three different objectives:
* L1, directional accuracy: 56.60%
* L2, directional accuracy: 50.77%
* pseudo-Huber, directional accuracy: 65.85%

L2 loss quadratically penalizes outliers, so it was pre-emptively stopped and the model gained no accuracy.
Pseudo-Huber loss is linear on outliers and quadratic on inliers, so it allows the model to learn "fat tails" and gain accuracy.

On the other hand, L1-model is theoretically guaranteed to be more robust, since it approximates the conditional median of the target, but it is closer to random guessing.

KANs with L2 loss were fit to errors of all three models, leading to improvements:
* XGB with L1: 56.60% -> 66.28%
* XGB with L2: 50.77% -> 66.53%
* XGB with pseudo-Huber: 65.85% -> 65.98%

While the biggest improvement was for XGB(L2) + KAN(L2), XGB with L1 loss is theoretically more robust. The difference can also be attributed to random noise.
