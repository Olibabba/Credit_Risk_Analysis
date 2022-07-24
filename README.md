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

SMOTE also had a modest performance but had slightly worse performance applied to the low credit risk.

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

Cluster Centroids had a simliarly modest precision and recall for high credit risk, however this model had the most False-Positives, and thus the lowest low credit risk recall. This may be attributed to the undersampling tecnique itself, which inherently uses a smaller data set to train.

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

The combination over and under sampling gave a small improvement to the individual models when predicting high credit risk. It landed between the individual models when predicting low credit risk however and still had a large number of False-Positives.

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

The Balanced Random Forest significantly reduced the number of False-Positives compared to the previous models. While this gave it a higher high-risk precision and F1 and a much higher low-risk recall and F1, it performed almost the same as the SMOTEEN model in True-Positive and False-Negative predictions.

**Results**
- Accuracy: 79%
- High-Risk Precision: 0.04
- High-Risk Recall: 0.67
- High-Risk F1 Score: 0.07
- Low-Risk Precision: 1.00
- Low-Risk Recall: 0.91
- Low-Risk F1 Score: 0.95

![Balanced Random Forest](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/BalancedRandomForest.png)

Top 15 Important Features:

![BRF Important Features](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/brf_features_list.png)

Importance of All Features:

![BRF Features Plot](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/brf_features_plt.png)

### ADA Boost

Easy Ensemble's ADA Boost showed significant improvements across all predictions. With the lowest number of False-Negatives and False-Positives by a wide margin, and the highest number of True-Positives, this model's high-risk F1 score is double the next best model.

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

The XGBoost model showed impressive high credit risk precision with only 5 False-Positives (compared to the next closest 975), This accomplishment was overshadowed by the model also having the lowest number of True-Positives and the highest number of False-Negatives.

**Results**
- Accuracy: 71%
- High-Risk Precision: 0.88
- High-Risk Recall: 0.41
- High-Risk F1 Score: 0.56
- Low-Risk Precision: 1.00
- Low-Risk Recall: 1
- Low-Risk F1 Score: 1

![XGBoost](https://github.com/Olibabba/Credit_Risk_Analysis/blob/main/resources/XGBoost.png)

### LightGBM

Light GBM performed alsmot the same as the XGBoost but with two more False-Positives

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

In evaluating these models it is important to pay particular attention to the high-risk recall. When looking for high credit risk, which is exeedingly rare according to this dataset, it is more desirable to minimize the false negatives. To put it another way, it is important
to minimize the number of high-risk borrowers that are incorrectly labeled as low-risk. 

In this context we can see that the under and over sampling had very similar and modest results, with the combination of the two showing a slight improvement in high-risk recall.

The popular XGBoost and Light GBM actually had the worst high risk recall, though these were notable in their very high precision. So if it is important to leadership that fewer borrowers are incorrectly labeled high-risk when they are in fact, low-risk, then these models would deserve a second look.

However, the ensemble methods showed better overall perfomrmace, and the Easy Ensemble's ADA Boost had by far the best performance.

The ADA Boost had incredible recall with only 9 False-Negatives, correctly identified far more True-Positives, and kept the False-Positives lower then all models except the XGBoost and Light GBM. 

It should be no surprise that my recommendation is to use the ADA Boost. 

The disclaimer here is that because the number of high-risk borrowers is so low, our improvements against high-risk borrowers is a matter of correctly identifying an extra dozen or two. For example the ADABoost had 9 False-Negatives, while the next best had 28. 

On the other hand, the number of False-Positives is on the scale of thousands. So even with the ADABoost model there could be almost a thousand low-risk borrowers incorrectly identified as high-risk.