
# Cancer Prediction Using Machine Learning

This project implements machine learning models to predict cancer diagnosis based on demographic, genetic, and lifestyle factors. Various visualizations and metrics are used to analyze and evaluate the dataset and model performance.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Machine Learning Models](#machine-learning-models)
7. [Evaluation Metrics](#evaluation-metrics)


---

## Overview

This project predicts cancer diagnosis using multiple machine learning algorithms, such as Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM). The project workflow includes data preprocessing, exploratory data analysis, model training, evaluation, and visualization.

---

## Dataset

The dataset used in this project includes the following features:
- **Age**: Age of the individual.
- **Gender**: Gender of the individual (Male/Female).
- **Smoking**: Smoking status (Yes/No).
- **GeneticRisk**: Genetic predisposition to cancer (Low/Medium/High).
- **CancerHistory**: History of cancer in the family (Yes/No).
- **Diagnosis**: Target variable indicating cancer diagnosis (Yes/No).

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/cancer-prediction
   cd cancer-prediction
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset (`The_Cancer_data_1500_V2.csv`) in the project directory.

---

## Usage

Run the script in jupyter notebook perform data preprocessing, exploratory data analysis, and train machine learning models:
```bash
 cancer-2 (1).ipynb
```

---

## Exploratory Data Analysis

The project includes various visualizations:
1. **Boxplot of Age Distribution**: Analyzes the age distribution of patients.
2. **Pie Charts**: Visualizes distributions for features such as gender, smoking, genetic risk, cancer history, and diagnosis.
3. **Correlation Heatmap**: Displays the relationship between numerical features.

---

## Machine Learning Models

The following machine learning models are implemented:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**: `n_neighbors=10`
3. **Support Vector Machines (SVM)**: Linear kernel

Each model is trained on 80% of the dataset and tested on the remaining 20%.

---

## Evaluation Metrics

The performance of the models is evaluated using:
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Proportion of true positive predictions out of all positive predictions.
- **Recall**: Proportion of true positive predictions out of actual positives.
- **F1 Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Displays true vs predicted classifications.

---

