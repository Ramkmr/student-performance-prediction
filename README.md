# Student Performance Prediction

## Overview

This project aims to predict student performance using various machine learning models. The primary goal is to understand the relationships between different features and their impact on the target variable `answered_correctly`. This README provides an overview of the project's methodology, findings, and instructions for usage.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation](#model-evaluation)
- [Insights](#insights)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this project, we utilized machine learning algorithms to predict whether a student answered a question correctly based on several features, including their abilities, the difficulty of the questions, and other contextual factors. By exploring various models, we aimed to identify which features are most significant in predicting student performance.

## Dataset

The dataset contains approximately 95,000 rows and includes the following key features:

- `ability`: The student's ability score.
- `difficulty`: The difficulty level of the question.
- `year`: The year when the question was posed.
- `attempts_count`: The number of attempts made by the student on the question.
- `adjusted_ability`: A normalized ability score for the student.
- `answered_correctly`: The target variable indicating whether the answer was correct (1) or incorrect (0).

### Class Imbalance
The distribution of the target variable `answered_correctly` is slightly imbalanced, with 48,669 instances of class 1 (correct) and 46,283 instances of class 0 (incorrect).

## Feature Engineering

During the feature engineering process, we explored the importance of various features using different models:

- **Logistic Regression**: The features `difficulty` and `ability` were found to have significant importance (both scored around 10 out of 20), while `correctness_rate` showed a lower importance (1 out of 20). The features `attempts_count` and `year` were deemed irrelevant.
  
- **Random Forest**: `ability` had a feature importance of 0.2, while `difficulty` scored 0.09, and `correctness_rate` was around 0.13. Again, `attempts_count` and `year` were unimportant.

- **XGBoost**: Except for `adjusted_ability`, the feature importances for all other features remained 0, indicating that they had no contribution to the model's predictions.

### Regularization Techniques

To mitigate potential overfitting and improve model performance, regularization techniques were implemented:
- **Logistic Regression**: L2 regularization was applied with a penalty term (`C=1.0`).
- **Random Forest**: Regularization was achieved by setting `max_depth=10` and `min_samples_split=5`, with `class_weight='balanced'` to handle class imbalance.
- **XGBoost**: Regularization was enforced through the `max_depth=6` and `scale_pos_weight` calculated based on class distribution.

## Model Evaluation

We evaluated three models using k-fold cross-validation (5-fold) and calculated their mean accuracy:

- **Logistic Regression Mean Accuracy**: 0.9994
- **Random Forest Mean Accuracy**: 0.9999
- **XGBoost Mean Accuracy**: 0.9989

The models performed exceptionally well, indicating a strong ability to predict student performance based on the provided features.

## Insights

1. **High Predictive Accuracy**: All models achieved very high accuracy, suggesting that the features used are highly indicative of the target variable.
2. **Feature Importance**: The feature `adjusted_ability` emerged as a crucial predictor across models, while others varied in importance.
3. **Regularization Effects**: The introduction of regularization techniques helped improve model robustness and mitigated overfitting.
4. **Class Imbalance Considerations**: Addressing the class imbalance through regularization and balanced weights enhanced the model's ability to generalize.

## Installation

To run this project, you need to have Python installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
