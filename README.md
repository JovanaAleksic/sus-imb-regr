# SUS: Selective Under-Sampling for Imbalanced Regression

This repository contains the official implementation of the Selective Under-Sampling (SUS) method and its iterative variant SUSiter, as described in our JAIR paper: ["A Selective Under-Sampling (SUS) method for imbalanced regression"]().

## Overview

SUS is a novel pre-processing method designed to address challenges in imbalanced regression tasks. The method performs selective under-sampling by considering the significance of each sample within a dataset at both feature and target levels. SUSiter, its iterative variant, exploits all available data while maintaining the advantages of under-sampling by selecting different subsets of data in each iteration of the learning process.

## Key Features

- Selective under-sampling considering both feature and target spaces
- Automated threshold determination for neighbor proximity
- Iterative variant (SUSiter) for use with iterative learning algorithms
- Support for high-dimensional data
- Compatible with neural network models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage with SUS

```python
import pandas as pd

# Read data
df = pd.read_csv('dataset.csv')

# Separate features and target
X = df.drop('target_column', axis=1).values  # Features (must be numeric, encode otherwise)
y = df['target_column'].values               # Target values
```

```python
from sus import SUS

# Initialize SUS with parameters
sus = SUS(
    k=7,              # number of neighbors in kNN model
    blobtr=0.75,      # threshold for close neighbors (default 75%)
    spreadtr=0.5      # threshold for cluster target values dispersion
)

# Apply SUS to your dataset
X_resampled, y_resampled = sus.fit_resample(X, y)
```

### Using SUSiter

```python
susiter = SUSiter(k=7, blobtr=0.75, spreadtr=0.5, replacement_ratio=0.3)
susiter.fit(X, y)

# Training loop
for epoch in range(num_epochs):
    X_iter, y_iter = susiter.get_iteration_sample()
    model.train(X_iter, y_iter)
```

## Parameters

- `k`: Number of neighbors in kNN model (recommended values: 5-10)
- `blobtr`: Percentile threshold for close neighbors distance (default: 75%)
- `spreadtr`: Threshold for cluster target values dispersion (default: 0.5)

## Benchmarking

The method has been evaluated on:
- 15 standard regression datasets
- 5 synthetic high-dimensional datasets
- 2 age estimation image datasets (IMDB-WIKI and AgeDB)

Results show that SUS and SUSiter typically outperform other state-of-the-art techniques like SMOGN and random under-sampling when used with neural networks.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{aleksic2024selective,
  title={A Selective Under-Sampling (SUS) method for imbalanced regression},
  author={Aleksic, Jovana and García-Remesal, Miguel},
  journal={Journal of Artificial Intelligence Research},
  year={2024}
}
```