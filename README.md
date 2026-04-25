# Credit Card Fraud Detection - Complete ML Project

## 📌 Project Overview

This project demonstrates a complete machine learning pipeline for credit card fraud detection, with emphasis on:
- **Handling imbalanced datasets** (98% legitimate vs 2% fraud)
- **Proper evaluation metrics** for imbalanced classification
- **Why accuracy is misleading** for fraud detection
- **Production-ready implementation**

---

## 🎯 Problem Statement

Credit card fraud is a significant issue for financial institutions. A naive model that always predicts "Not Fraud" would achieve 98% accuracy but would catch 0 frauds. This project teaches the importance of:

1. **Understanding the business cost** of different error types
2. **Using appropriate metrics** for imbalanced data
3. **Threshold tuning** for optimal performance
4. **Feature engineering** from transaction data

---

## 📊 Dataset

### Synthetic Dataset Generated (100,000 transactions)

**Features:**
- `transaction_id`: Unique identifier
- `customer_id`: Customer identifier
- `amount`: Transaction amount ($)
- `merchant_category`: Type of merchant (grocery, restaurant, online, etc.)
- `distance_from_home`: Distance in km
- `hour_of_transaction`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `month`: Month (1-12)
- `device_type`: mobile, online, ATM, in_person
- `is_fraud`: Target variable (0/1)

### Class Distribution
- **Legitimate transactions**: 98,000 (98.00%)
- **Fraudulent transactions**: 2,000 (2.00%)
- **Imbalance ratio**: 1:49

### Key Patterns Captured
```
LEGITIMATE TRANSACTIONS:
- Smaller amounts (avg: $150)
- Closer to home (avg: 3.6 km)
- During business hours
- Mix of device types

FRAUDULENT TRANSACTIONS:
- Larger amounts (avg: $302)
- Far from home (avg: 49.3 km)
- Mostly at night and odd hours
- Predominantly online
```

---

## 🔧 Methodology

### Step 1: Data Preprocessing
- Categorical variable encoding (merchant category, device type)
- Feature scaling using StandardScaler
- No missing values

### Step 2: Feature Engineering
- `transactions_per_hour`: Cumulative transaction count per hour
- `is_night_hour`: Binary flag for night transactions (10 PM - 6 AM)
- `is_weekend`: Binary flag for weekend transactions

### Step 3: Handling Imbalanced Data
**Problem**: Standard ML algorithms perform poorly on imbalanced data

**Solution**: Oversampling the minority class
```python
# Original: 2% fraud rate
# After oversampling: 50% fraud rate (1:1 balanced)
# This forces the model to learn fraud patterns better
```

### Step 4: Train-Test Split
- 80% training (80,000 samples)
- 20% testing (20,000 samples)
- Stratified split to maintain class distribution

### Step 5: Model Training
Trained three models:
1. **Logistic Regression** - Linear baseline
2. **Random Forest** - Ensemble with feature importance
3. **Gradient Boosting** - Best performance

---

## 📈 Results

### Model Comparison

| Metric | Logistic Regression | Random Forest | Gradient Boosting |
|--------|-------------------|---------------|-------------------|
| **Accuracy** | 0.9749 (97.49%) | 0.9986 (99.86%) | 0.9964 (99.64%) |
| **Precision** | 0.4404 (44%) | 0.9895 (99%) | 0.8631 (86%) |
| **Recall** | 0.9425 (94%) | 0.9400 (94%) | 0.9775 (98%) |
| **F1-Score** | 0.6003 | 0.9641 | 0.9168 |
| **ROC-AUC** | 0.9883 | 0.9945 | 0.9996 |

### Performance Breakdown (Gradient Boosting)

```
Confusion Matrix:
                    Predicted Legitimate    Predicted Fraud
Actual Legitimate:        19,538                  62
Actual Fraud:                 9                 391

Metrics:
- Caught 391 out of 400 frauds (97.75% recall)
- 86.31% of detected frauds were actually frauds
- Only 9 frauds missed out of 400
- 62 false alarms
```

---

## 🎓 Key Learnings

### 1. Why Accuracy Isn't Everything

```python
# Naive model that always predicts "Not Fraud":
Accuracy = 98,000 / 100,000 = 98% ✓ HIGH!
Recall = 0 / 2,000 = 0% ✗ CATCHES NO FRAUDS!

# Gradient Boosting model:
Accuracy = 19,929 / 20,000 = 99.64% ✓ HIGH!
Recall = 391 / 400 = 97.75% ✓ CATCHES ALMOST ALL FRAUDS!
```

**Lesson**: For fraud detection, missing a fraud is much more costly than a false positive.

### 2. Proper Metrics for Imbalanced Data

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Precision** | Of predicted frauds, how many are actually fraud? | Controls false alarm rate |
| **Recall** | Of actual frauds, how many did we catch? | Controls missed frauds |
| **F1-Score** | Harmonic mean of precision and recall | Balances both concerns |
| **ROC-AUC** | Model's ability across all probability thresholds | Shows overall discrimination power |
| **Precision-Recall Curve** | Better than ROC for imbalanced data | Shows performance trade-offs |

### 3. Handling Imbalanced Data

**Before Oversampling** (training set):
```
Fraud rate: 2%
Model learns: "Just predict not fraud, get 98% accuracy"
Result: High accuracy, low recall
```

**After Oversampling** (training set):
```
Fraud rate: 50%
Model learns: "Fraud patterns matter, need to detect them"
Result: High accuracy AND high recall
```

### 4. Threshold Tuning

Default threshold = 0.5 probability
```python
if fraud_probability > 0.5:
    flag_as_fraud()
```

**Problem**: Too conservative, misses many frauds

**Solution**: Lower threshold to 0.3 or 0.2
```python
if fraud_probability > 0.3:  # More sensitive
    flag_as_fraud_for_review()  # Flag, don't block automatically
```

**Trade-off**: More false positives (62 in our model), but catches more frauds (391 vs 376)

### 5. Feature Importance

```
Top Predictive Features:
1. Distance from home (59.8%) ⭐⭐⭐⭐⭐
   - Frauds happen farther away
   
2. Merchant category (12.6%) ⭐⭐
   - Online/luxury stores have more fraud
   
3. Device type (11.4%) ⭐⭐
   - Online devices more suspicious
   
4. Transaction amount (8.8%) ⭐
   - Frauds tend to be larger
   
5. Hour of transaction (4.7%)
   - Night transactions more suspicious
```

---

## 🚀 How to Run the Project

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Generate Dataset
```bash
python generate_fraud_data.py
```
Output: `credit_card_fraud.csv` (100,000 transactions)

### Step 2: Run Analysis & Training
```bash
python fraud_detection.py
```
Output:
- `fraud_detection_analysis.png` - Model comparison visualizations
- `confusion_matrices.png` - Confusion matrices for all models
- `feature_importance.png` - Feature importance ranking
- Console output with full metrics

---

## 📊 Visualizations Generated

### 1. fraud_detection_analysis.png
- **Panel 1**: Class distribution (highly imbalanced)
- **Panel 2**: Model comparison across key metrics
- **Panel 3**: ROC curves (why Gradient Boosting wins)
- **Panel 4**: Precision-Recall curves (more relevant for imbalanced data)

### 2. confusion_matrices.png
- Confusion matrices for all three models
- Shows Precision and Recall for each model
- Easy to see trade-offs

### 3. feature_importance.png
- Top 10 most important features
- Random Forest-based importance ranking
- Shows what the model learns

---

## 💡 Practical Deployment

### Using the Model in Production

```python
import pickle
import numpy as np

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# For a new transaction
def check_fraud_risk(transaction_features):
    """
    transaction_features: numpy array of shape (1, n_features)
    Returns: fraud probability and action
    """
    fraud_probability = model.predict_proba(transaction_features)[0, 1]
    
    if fraud_probability > 0.3:
        return {
            'risk_level': 'HIGH',
            'probability': fraud_probability,
            'action': 'flag_for_review',  # Request verification
            'message': f'⚠️ High fraud risk: {fraud_probability*100:.1f}%'
        }
    elif fraud_probability > 0.15:
        return {
            'risk_level': 'MEDIUM',
            'probability': fraud_probability,
            'action': 'monitor',  # Log and monitor
            'message': f'⏱️ Medium fraud risk: {fraud_probability*100:.1f}%'
        }
    else:
        return {
            'risk_level': 'LOW',
            'probability': fraud_probability,
            'action': 'approve',  # Process normally
            'message': f'✓ Low fraud risk: {fraud_probability*100:.1f}%'
        }

# Example usage
new_transaction = X_test_scaled[0:1]
result = check_fraud_risk(new_transaction)
print(result['message'])
```

### Business Impact

Assuming:
- Cost of missed fraud: $1,000
- Cost of false alarm: $5
- Processing fee: $2

**Old system** (perfect at detecting fraud, but all alerts):
- Catch all fraud: 2,000 frauds × $0 = $0
- False alarms: 50,000 false positives × $5 = $250,000
- Total: $250,000 cost

**Our model** (Gradient Boosting):
- Catch 97.75% fraud: 1,955 frauds × $0 = $0
- Miss 2.25% fraud: 45 frauds × $1,000 = $45,000
- False alarms: 62 × $5 = $310
- Total: $45,310 cost ✓ MUCH BETTER!

---

## 📚 What You've Learned

✅ **Imbalanced Data Handling**
- Understand why standard approaches fail
- Oversampling vs undersampling trade-offs
- SMOTE and other advanced techniques

✅ **Evaluation Metrics**
- When accuracy is misleading
- Precision, Recall, F1-Score fundamentals
- ROC-AUC and Precision-Recall curves
- Confusion matrix interpretation

✅ **Feature Engineering**
- Creating meaningful features from raw data
- Domain knowledge application
- Feature importance analysis

✅ **Model Selection**
- Ensemble methods vs linear models
- Hyperparameter tuning concepts
- Threshold optimization

✅ **Production Considerations**
- Real-world cost analysis
- Decision thresholds
- Monitoring and updating models

---

## 📁 Project Files

```
credit_card_fraud_detection/
├── generate_fraud_data.py           # Synthetic data generation
├── fraud_detection.py               # Main analysis & modeling
├── requirements.txt                 # Python dependencies
├── credit_card_fraud.csv            # Generated dataset (100k rows)
├── fraud_detection_analysis.png     # Model comparison plots
├── confusion_matrices.png           # Confusion matrices
├── feature_importance.png           # Feature importance
└── README.md                        # This file
```

---

## 🎯 Resume Talking Points

When discussing this project in interviews:

1. **"I handled a highly imbalanced dataset (98-2 split) by using oversampling to ensure the model learned fraud patterns effectively."**

2. **"I demonstrated why accuracy is a poor metric for fraud detection by showing how a naive model gets 98% accuracy while catching 0% of frauds."**

3. **"I implemented three different models (Logistic Regression, Random Forest, Gradient Boosting) and selected Gradient Boosting based on ROC-AUC and F1-score rather than accuracy."**

4. **"I tuned the classification threshold from the default 0.5 to 0.3, understanding the trade-off between false positives and false negatives."**

5. **"I performed feature engineering to create meaningful predictors like night_hour and transactions_per_hour, improving model performance."**

6. **"I analyzed feature importance to understand that distance from home is 60% of the model's decision-making, guiding business strategy."**

---

## 🔍 Further Improvements

For even better performance:

1. **SMOTE** - Synthetic minority oversampling technique
2. **Class weights** - Penalize fraud misclassification more heavily
3. **Cost-sensitive learning** - Directly optimize for business costs
4. **Anomaly detection** - Isolation Forest, Local Outlier Factor
5. **Time-series validation** - Proper temporal data splitting
6. **Feature interactions** - Polynomial features, interaction terms
7. **Hyperparameter tuning** - GridSearchCV, RandomizedSearchCV
8. **Ensemble stacking** - Combine multiple models

---

## 📖 Additional Resources

**Imbalanced Data:**
- https://imbalanced-learn.org/stable/
- Papers on SMOTE, class weights, cost-sensitive learning

**Fraud Detection:**
- Real credit card fraud datasets (anonymized)
- IEEE fraud detection competitions

**Evaluation Metrics:**
- Precision-Recall vs ROC curves
- Business-oriented metric selection

---

## ✨ Key Takeaway

> **"The best model isn't always the one with the highest accuracy. 
> For fraud detection, catching 97.75% of frauds with reasonable false alarms 
> is far more valuable than a 98% accurate model that catches nothing."**

---

Good luck with your data science journey! 🚀
