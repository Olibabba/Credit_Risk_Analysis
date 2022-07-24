# Credit_Risk_Analysis

## Overview

In order to build on previous machine learning work, I will now be applying what I've learned to a large, real world credit card data set in an attempt to find the best model to predict credit risk. 

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, I will will need to employ different techniques to train and evaluate models with unbalanced classes.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I will apply several models to this problem including:

- Oversampling with Random Over Sampler and Smote
- Undersampling with Cluster Centroids
- A combination over and under sampling with SMOTEEN
- Ensemble methods with Balanced Random Forest and ADABoost

In adittion, I have included two extra notable and popular models:
- XGBoost
- Light GBM


## Results

### Random Over Sampler

The RandomOverSampler proved to be a modest model with low precision and recall when targeting high credit risk.

**Results**
- Accuracy: 65%
- High-Risk Precision: 0.01 
- High-Risk Recall: 0.62
- High-Risk F1 Score: 0.02
- Low-Risk Precision: 1.00
- Low-Risk Recall: 0.68
- Low-Risk F1 Score: 0.81

![Random Over Sampler](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/RandomOverSampler.png)

### SMOTE

**Results**
- Accuracy: 64%
- High-Risk Precision: 0.01
- High-Risk Recall: 0.63
- High-Risk F1 Score: 0.02
- Low-Risk Precision: 1.00
- Low-Risk Recall: 0.66
- Low-Risk F1 Score: 0.79

![SMOTE](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/SMOTE.png)

### Cluster Centroids

**Results**
- Accuracy: 53%
- High-Risk Precision: 0.01
- High-Risk Recall: 0.61
- High-Risk F1 Score: 0.01
- Low-Risk Precision: 1.00
- Low-Risk Recall: 0.45
- Low-Risk F1 Score: 0.62

![Cluster Centroids](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/ClusterCentroids.png)

### SMOTEEN

**Results**
- Accuracy: 62%
- High-Risk Precision: 0.01
- High-Risk Recall: 0.68
- High-Risk F1 Score: 0.02
- Low-Risk Precision: 1.00
- Low-Risk Recall: 0.57
- Low-Risk F1 Score: 0.73

![SMOTEEN](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/SMOTEEN.png)

### Balanced Random Forest

**Results**
- Accuracy: 79%
- High-Risk Precision: 0.04
- High-Risk Recall: 0.67
- High-Risk F1 Score: 0.07
- Low-Risk Precision: 1.00
- Low-Risk Recall: 0.91
- Low-Risk F1 Score: 0.95

![Balanced Random Forest](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/BalancedRandomForest.png)
![BRF Important Features](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/brf_features_list.png)
![BRF Features Plot](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/brf_features_plt.png)

### ADA Boost

**Results**
- Accuracy: 92%
- High-Risk Precision: 0.07
- High-Risk Recall: 0.90
- High-Risk F1 Score: 0.14
- Low-Risk Precision: 1.00
- Low-Risk Recall: 0.94
- Low-Risk F1 Score: 0.97

![ADA Boost](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/ADABoost.png)

### XGBoost

**Results**
- Accuracy: 92%
- High-Risk Precision: 0.07
- High-Risk Recall: 0.90
- High-Risk F1 Score: 0.14
- Low-Risk Precision: 1.00
- Low-Risk Recall: 0.94
- Low-Risk F1 Score: 0.97

![XGBoost](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/XGBoost.png)

### LightGBM

**Results**
- Accuracy: 71%
- High-Risk Precision: 0.84
- High-Risk Recall: 0.41
- High-Risk F1 Score: 0.55
- Low-Risk Precision: 1.00
- Low-Risk Recall: 1
- Low-Risk F1 Score: 1

![LightGBM](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/LGBM.png)

## Summary