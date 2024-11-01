# Student Performance Prediction

## Overview
This project aims to predict student performance by analyzing various factors, such as student ability and question difficulty, to determine the likelihood of answering correctly. Through this analysis, we seek to provide a comprehensive understanding of factors influencing student performance and develop predictive models with high accuracy.

---

## Table of Contents
- [Introduction](#introduction)
- [Assignment Objectives](#assignment-objectives)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Regularization Techniques](#regularization-techniques)
- [Model Evaluation](#model-evaluation)
- [Insights](#insights)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
This project applies machine learning to predict if a student will answer a question correctly, using the provided features to assess relationships and identify patterns. By testing multiple models and techniques, we aim to identify which factors most significantly impact student responses, offering insights into student abilities, question difficulty, and other educational variables.

---

## Assignment Objectives
This project is structured to address the following questions based on student response data:

1. **How did students' ability to answer questions change over time?**
2. **Did the questions become more or less difficult?**
3. **Can a model be created to predict if a student will answer a question correctly?**
4. **Document any additional observations about the data.**

These questions are crucial for understanding student performance trends and the effectiveness of educational assessments.

---

## Dataset
The dataset consists of two CSV files representing student response data from the years 2021 and 2022, with each row corresponding to a unique student-question interaction. The dataset contains approximately 95,000 rows, with the following columns:

- **student_id**: Unique identifier for each student.
- **question_id**: Unique identifier for each question.
- **ability**: The student's skill or ability score, which may change over time.
- **difficulty**: The level of difficulty for each question, potentially varying by year.
- **answered_correctly**: Target variable indicating if the answer was correct (1) or incorrect (0).

### Additional Engineered Features
To enrich the dataset and improve model predictions, we introduced the following features:

- **year**: The year in which the question was answered (2021 or 2022).
- **correctness_rate**: The rate of correct answers for each student, calculated across all questions.
- **attempts_count**: The total number of attempts made by each student.
- **adjusted_ability**: A modified version of the `ability` score, adjusted based on prior attempts and correctness.

These additional features help capture temporal trends, individual question engagement, and adjusted metrics that are more representative of each student’s performance.

---

## Feature Engineering
To maximize model effectiveness, we performed feature engineering and evaluated feature significance across different models:

1. **Logistic Regression**: This model revealed that `difficulty` and `ability` were the most impactful features. Other features showed minimal influence, suggesting they may not add predictive value for a linear model.
  
2. **Random Forest**: This ensemble model highlighted `ability` as the strongest predictor, with `difficulty` following behind. The feature importance metrics were consistent with expectations, supporting the role of student ability and question difficulty.

3. **XGBoost**: In the case of XGBoost, the feature `adjusted_ability` (a transformed version of `ability`) emerged as the sole contributor to prediction accuracy. This finding helped us streamline our model by focusing on core predictive features.

### Data Visualization and Exploratory Analysis
- **Ability vs. Difficulty**: Scatter plots revealed that as question difficulty increased, the likelihood of correct answers decreased, especially among students with lower ability scores.
  
- **Response Patterns by Question**: Most questions had a response count of about 2000, with a noticeable drop in responses for the last four questions (IDs 47–50). This anomaly suggests these questions were either more challenging or impacted by timing constraints.

- **Student Ability Distribution**: We observed that students with lower abilities were more likely to answer questions incorrectly, particularly for the more difficult questions, aligning with expected performance patterns.

---

## Regularization Techniques
To mitigate potential overfitting and improve model generalizability, we applied specific regularization techniques based on model type:

- **Logistic Regression**: Applied L2 regularization with a penalty term (C=1.0) to handle potential multicollinearity and increase stability.
  
- **Random Forest**: Set hyperparameters to control depth (`max_depth=10`) and sample splits (`min_samples_split=5`), alongside setting `class_weight='balanced'` to address the class imbalance.

- **XGBoost**: Incorporated `scale_pos_weight` (calculated based on class distribution) and set `max_depth=6` to avoid overfitting while handling the slight class imbalance effectively.

These regularization techniques allowed each model to leverage the available data without overfitting to specific trends or patterns.

---

## Model Evaluation
Each model was evaluated using 5-fold cross-validation to obtain mean accuracy, as follows:

- **Logistic Regression**: Achieved an average accuracy of **0.9994**, indicating strong predictive capability with minimal feature engineering.
  
- **Random Forest**: Recorded an average accuracy of **0.9999**, making it the most accurate model among those tested.
  
- **XGBoost**: Yielded an average accuracy of **0.9989**, still performing well though slightly lower than Random Forest.

The models demonstrated high accuracy, reflecting the strength of the features and engineering methods used. However, care was taken to avoid data leakage by focusing only on relevant features and evaluating their importance in detail.

---

## Insights
1. **High Predictive Accuracy**: All models achieved excellent accuracy, reinforcing the high relevance of the engineered features in predicting student performance.

2. **Feature Importance**: Across models, `adjusted_ability` consistently emerged as a key predictor, suggesting that a student’s ability relative to question difficulty plays a significant role in their performance.

3. **Regularization and Robustness**: Regularization techniques, particularly in Random Forest and XGBoost, improved model robustness and the handling of slight class imbalance, enhancing generalizability.

4. **Class Imbalance Handling**: By setting balanced weights in Random Forest and adjusting `scale_pos_weight` in XGBoost, the models achieved consistent predictive power across both classes.

5. **Question Engagement and Difficulty Trends**: The low response rates on the last four questions suggested a potential increase in difficulty or constraints that warrant further investigation.

---

## Installation
To run this project, ensure you have Python installed along with the required dependencies:

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
