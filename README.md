# SUS: Selective Under-Sampling for Imbalanced Regression

This repository contains the official implementation of the Selective Under-Sampling (SUS) method and its iterative variant SUSiter, as described in our JAIR paper: ["A Selective Under-Sampling (SUS) method for imbalanced regression"](https://jair.org/index.php/jair/article/view/16062).

## Overview

SUS is a novel pre-processing method designed to address challenges in imbalanced regression tasks. The method performs selective under-sampling by considering the significance of each sample within a dataset at both feature and target levels. SUSiter, its iterative variant, exploits all available data while maintaining the advantages of under-sampling by selecting different subsets of data in each iteration of the learning process.

## Key Features

- Selective under-sampling considering both feature and target spaces
- Automated threshold determination for neighbor proximity
- Iterative variant (SUSiter) for use with iterative learning algorithms
- Support for high-dimensional data
- Compatible with neural network models

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Reading in Data

```python
import pandas as pd

# Read data
df = pd.read_csv('dataset.csv')

# Separate features and target
X = df.drop('target_column', axis=1).values  # Features (must be numeric, encode otherwise)
y = df['target_column'].values               # Target values
```
### Basic Usage with SUS

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
- 2 age estimation image datasets:
	- IMDB-WIKI https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
	- AgeDB https://paperswithcode.com/dataset/agedb

Results show that SUS and SUSiter typically outperform other state-of-the-art techniques like SMOGN and random under-sampling when used with neural networks.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{ WOS:001400275900003,
Author = {Aleksic, Jovana and Garcia-Remesal, Miguel},
Title = {A Selective Under-Sampling (SUS) Method For Imbalanced
Regression},
Journal = {JOURNAL OF ARTIFICIAL INTELLIGENCE RESEARCH},
Year = {2025},
Volume = {82},
Pages = {111-136},
Abstract = {Many mainstream machine learning approaches, such as neural networks, are not well suited to work with imbalanced data. Yet, this problem is frequently present in many real-world data sets. Collection methods are imperfect, and often not able to capture enough data in a specific
range of the target variable. Furthermore, in certain tasks data is inherently imbalanced with many more normal events than edge cases. This problem is well studied within the classification context. However, only several methods have been proposed to deal with regression tasks. In addition, the proposed methods often do not yield good performance with high-dimensional data, while imbalanced high-dimensional regression has scarcely been explored. In this paper we present a selective under-sampling (SUS) algorithm for dealing with imbalanced regression and its iterative version SUSiter. We assessed this method on 15 regression data sets from different imbalanced domains, 5 synthetic high-dimensional imbalanced data sets and 2 more complex imbalanced age estimation image data sets. Our results suggest that SUS and SUSiter typically outperform other state-of-the-art techniques like SMOGN, or random under-sampling, when used with neural networks as learners.},
Publisher = {AI ACCESS FOUNDATION},
Address = {USC INFORMATION SCIENCES INST, 4676 ADMIRALITY WAY, MARINA
DEL REY, CA
    90292-6695 USA},
Type = {Article},
Language = {English},
Affiliation = {Aleksic, J (Corresponding Author), Univ Politecn Madrid,
Madrid, Spain.
    Aleksic, J (Corresponding Author), Weill Cornell Med Qatar, Ar
Rayyan, Qatar.
    Aleksic, Jovana, Univ Politecn Madrid, Madrid, Spain.
    Aleksic, Jovana, Weill Cornell Med Qatar, Ar Rayyan, Qatar.
    Garcia-Remesal, Miguel, Univ Politecn Madrid, Dept Inteligencia
Artificial, Biomed Informat Grp, Madrid, Spain.},
ISSN = {1076-9757},
EISSN = {1943-5037},
Research-Areas = {Computer Science},
Web-of-Science-Categories  = {Computer Science, Artificial
Intelligence},
Author-Email = {JOVANA.R.ALEKSIC@GMAIL.COM
    MGREMESAL@FI.UPM.ES},
Affiliations = {Universidad Politecnica de Madrid; Qatar Foundation
(QF); Weill Cornell
    Medical College Qatar; Universidad Politecnica de Madrid},
ORCID-Numbers = {Aleksic, Jovana/0000-0002-3366-8379},
Number-of-Cited-References = {41},
Times-Cited = {0},
Usage-Count-Last-180-days = {0},
Usage-Count-Since-2013 = {0},
Journal-ISO = {J. Artif. Intell. Res.},
Doc-Delivery-Number = {S7W4E},
Web-of-Science-Index = {Science Citation Index Expanded (SCI-EXPANDED)},
Unique-ID = {WOS:001400275900003},
DA = {2025-01-29},
}
```
