# ML Anomaly Detection Fundamentals

Machine learning fundamentals with focus on anomaly detection techniques — foundational implementations for fraud detection and security applications.

## Overview

This repository contains implementations of machine learning algorithms specifically applied to anomaly detection, a critical component in modern fraud detection systems. Anomaly detection enables identification of unusual patterns that deviate from normal behavior, making it essential for:

- Financial fraud detection
- Network intrusion detection
- Credit card fraud prevention
- Transaction monitoring
- Identity theft detection

## Why Anomaly Detection for Fraud Detection?

Fraud detection is fundamentally an anomaly detection problem because:

1. **Class Imbalance**: Fraudulent transactions are rare (often <1% of data)
2. **Evolving Patterns**: Fraud techniques constantly change
3. **Supervised Challenges**: Labeled fraud data is expensive and scarce
4. **Real-time Requirements**: Systems must detect anomalies as they occur

This repository explores both supervised and unsupervised approaches to address these challenges.

## Modules

### 1. Anomaly Detection (`anomaly_detection/`)

Implementation of multiple ML algorithms for fraud classification using a synthetic financial transactions dataset:

**Algorithms Implemented:**
- **Random Forest**: Ensemble method for robust classification
- **Support Vector Machine (SVC)**: Kernel-based classification with RBF kernel
- **Logistic Regression**: With SMOTE oversampling for class imbalance
- **Deep Neural Network (DNN)**: Multi-layer perceptron for binary classification

**Techniques Demonstrated:**
- Multiple Correspondence Analysis (MCA) for categorical features
- Principal Component Analysis (PCA) for dimensionality reduction
- K-means clustering for unsupervised pattern discovery
- SMOTE (Synthetic Minority Over-sampling Technique) for imbalanced data
- Time-based feature engineering (temporal grouping)

**Key Files:**
- `anomaly_detection/anomalyDetection.py` - Complete fraud detection pipeline

**Evaluation Metrics:**
- Confusion matrices for train/test sets
- ROC curves and AUC scores
- Precision, Recall, F1-scores
- Classification reports with detailed metrics

### 2. Time Series Analysis (`time_series/`)

Financial time series forecasting and technical analysis implementations:

**Phase 1 - LSTM Implementation:**
- Long Short-Term Memory networks for stock prediction
- Time series data preprocessing and feature engineering
- Model training with early stopping and callbacks

**Phase 2 - Technical Indicators:**
- Technical analysis indicators implementation
- Feature engineering for financial data
- Data pipeline with PostgreSQL integration

**Applications:**
- Time series anomaly detection
- Market behavior analysis
- Feature extraction for fraud detection models

## Fraud Detection Application Examples

### Feature Engineering for Transactions

The anomaly detection module demonstrates feature engineering techniques:

```python
# Error-based features for detecting balance inconsistencies
df['errorOrig'] = df['amount'] + df['newBalanceOrig'] - df['oldBalanceOrig']
df['errorDest'] = df['amount'] + df['oldbalanceDest'] - df['newbalanceDest']

# Temporal features
df['Day of Week'] = df['Date'].dt.day_name()
df['Group Time'] = group_time(df['Time'])  # 3-hour time buckets
```

### Handling Class Imbalance

Fraud detection datasets are typically highly imbalanced. This repository demonstrates:

1. **SMOTE**: Synthetic minority oversampling
2. **Evaluation metrics**: Focus on recall and precision rather than accuracy
3. **Threshold tuning**: Optimizing decision boundaries

### Evaluation Metrics for Imbalanced Datasets

| Metric | Use Case |
|--------|----------|
| **Recall** | Minimize false negatives (missed fraud) |
| **Precision** | Minimize false positives (false alarms) |
| **F1-Score** | Balance between precision and recall |
| **AUC-ROC** | Model performance across thresholds |

## Installation

```bash
# Clone the repository
git clone https://github.com/alazkiyai09/ml-anomaly-detection-fundamentals.git
cd ml-anomaly-detection-fundamentals

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

See `requirements.txt` for full dependencies:

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
imbalanced-learn>=0.9.0
xgboost>=1.5.0
prince>=0.7.0
```

## Usage

### Anomaly Detection

```python
cd anomaly_detection
python anomalyDetection.py
```

This will:
1. Preprocess the fraud dataset
2. Apply MCA and PCA for feature transformation
3. Train multiple ML models
4. Generate evaluation metrics and visualizations
5. Save results to `Data Output/` directory

### Time Series Analysis

```python
cd time_series
# Run Phase 1 for LSTM implementation
python "Phase 1.py"
# Run Phase 2 for technical indicators
python "Phase 2.py"
```

## Dataset

The anomaly detection module uses a synthetic financial transactions dataset with features including:

- Transaction type (TRANSFER, CASH_OUT)
- Transaction amount
- Account balances (old and new)
- Temporal information (date, time)
- Fraud labels

**Note**: The dataset should be placed as `Dataset Fraud (New_Final).csv` in the `anomaly_detection/` directory.

## Key Learnings

| Concept | Application |
|---------|-------------|
| **Dimensionality Reduction** | MCA for categorical, PCA for numerical features |
| **Class Imbalance Handling** | SMOTE oversampling for minority class |
| **Ensemble Methods** | Random Forest for robust predictions |
| **Neural Networks** | DNN for non-linear pattern detection |
| **Evaluation Strategy** | Metrics tailored for imbalanced data |

## Connection to Professional Work

This repository demonstrates foundational machine learning concepts that support advanced fraud detection and security research:

- **Feature Engineering**: Essential for real-world fraud detection systems
- **Imbalanced Learning**: Core challenge in security applications
- **Model Evaluation**: Critical for production systems where false negatives are costly

These fundamentals directly apply to building robust fraud detection pipelines and understanding the machine learning lifecycle for security applications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

This is a learning repository. Feel free to open issues or pull requests for improvements.

## Author

**Al Azkiyai**
- GitHub: [alazkiyai09](https://github.com/alazkiyai09)
- Specialization: Fraud Detection & AI Security

## References

- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
