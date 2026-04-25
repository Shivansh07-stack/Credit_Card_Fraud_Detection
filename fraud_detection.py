"""
Credit Card Fraud Detection - Complete ML Pipeline
This project demonstrates handling imbalanced classification, 
proper evaluation metrics, and anomaly detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                            auc, precision_recall_curve, f1_score, precision_score,
                            recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("CREDIT CARD FRAUD DETECTION - COMPLETE PROJECT")
print("="*70)

# ==================== 1. LOAD DATA ====================
print("\n[STEP 1] Loading Data...")
df = pd.read_csv('/home/claude/credit_card_fraud.csv')
print(f"✓ Loaded {len(df):,} transactions")
print(f"✓ Features: {df.shape[1]}")

# ==================== 2. EXPLORATORY DATA ANALYSIS ====================
print("\n[STEP 2] Exploratory Data Analysis...")

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Class distribution
print("\n" + "-"*70)
print("CLASS DISTRIBUTION (Imbalanced Data Problem)")
print("-"*70)
fraud_count = df['is_fraud'].value_counts()
print(f"Legitimate: {fraud_count[0]:,} ({fraud_count[0]/len(df)*100:.2f}%)")
print(f"Fraudulent: {fraud_count[1]:,} ({fraud_count[1]/len(df)*100:.2f}%)")
print(f"Imbalance Ratio: 1:{fraud_count[0]/fraud_count[1]:.1f}")

# Statistical comparison
print("\n" + "-"*70)
print("LEGITIMATE vs FRAUDULENT TRANSACTIONS")
print("-"*70)
comparison = df.groupby('is_fraud')[['amount', 'distance_from_home']].describe()
print(comparison[['amount', 'distance_from_home']])

# ==================== 3. DATA PREPROCESSING ====================
print("\n[STEP 3] Data Preprocessing...")

# Create a copy for processing
data = df.copy()

# Drop timestamp and transaction_id (not useful for modeling)
data = data.drop(['timestamp', 'transaction_id'], axis=1)

# Encode categorical variables
print("Encoding categorical variables...")
le_merchant = LabelEncoder()
le_device = LabelEncoder()

data['merchant_category_encoded'] = le_merchant.fit_transform(data['merchant_category'])
data['device_type_encoded'] = le_device.fit_transform(data['device_type'])

# Drop original categorical columns
data = data.drop(['merchant_category', 'device_type'], axis=1)

print(f"✓ Features after preprocessing: {data.shape[1]}")

# ==================== 4. FEATURE ENGINEERING ====================
print("\n[STEP 4] Feature Engineering...")

# Add transaction velocity features
data['transactions_per_hour'] = data.groupby(['customer_id', 'hour_of_transaction']).cumcount() + 1

# Add day/night indicator
data['is_night_hour'] = ((data['hour_of_transaction'] >= 22) | 
                         (data['hour_of_transaction'] <= 6)).astype(int)

# Add weekend indicator
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

print(f"✓ Total features now: {data.shape[1] - 2}")  # -2 for customer_id and is_fraud

# ==================== 5. TRAIN-TEST SPLIT ====================
print("\n[STEP 5] Train-Test Split...")

# Separate features and target
X = data.drop(['customer_id', 'is_fraud'], axis=1)
y = data['is_fraud']

# Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")
print(f"✓ Training fraud rate: {y_train.mean()*100:.2f}%")
print(f"✓ Test fraud rate: {y_test.mean()*100:.2f}%")

# ==================== 6. FEATURE SCALING ====================
print("\n[STEP 6] Feature Scaling...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# ==================== 7. HANDLING IMBALANCED DATA ====================
print("\n[STEP 7] Handling Imbalanced Data - Oversampling Minority Class...")

# Oversample minority class (fraud)
X_train_fraud = X_train_scaled[y_train == 1]
X_train_legit = X_train_scaled[y_train == 0]
y_train_fraud = y_train[y_train == 1]
y_train_legit = y_train[y_train == 0]

# Oversample fraud cases
X_fraud_oversampled = resample(X_train_fraud, n_samples=len(X_train_legit), 
                               random_state=42, replace=True)
y_fraud_oversampled = resample(y_train_fraud, n_samples=len(y_train_legit),
                               random_state=42, replace=True)

# Combine
X_train_balanced = np.vstack([X_train_legit, X_fraud_oversampled])
y_train_balanced = np.concatenate([y_train_legit, y_fraud_oversampled])

print(f"✓ Original training set fraud rate: {y_train.mean()*100:.2f}%")
print(f"✓ Balanced training set fraud rate: {y_train_balanced.mean()*100:.2f}%")

# ==================== 8. MODEL TRAINING ====================
print("\n[STEP 8] Training Multiple Models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, 
                                           class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42,
                                                   max_depth=5)
}

trained_models = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_balanced, y_train_balanced)
    trained_models[name] = model
    print(f"  ✓ {name} trained")

# ==================== 9. MODEL EVALUATION ====================
print("\n" + "="*70)
print("[STEP 9] MODEL EVALUATION - WHY ACCURACY ISN'T EVERYTHING")
print("="*70)

# Store results for comparison
results = {}

for name, model in trained_models.items():
    print(f"\n{'-'*70}")
    print(f"{name.upper()}")
    print(f"{'-'*70}")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (of predicted frauds, how many are actually fraud)")
    print(f"Recall:    {recall:.4f} (of actual frauds, how many did we catch)")
    print(f"F1-Score:  {f1:.4f} (harmonic mean of precision & recall)")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0,0]:,}  | False Positives: {cm[0,1]:,}")
    print(f"  False Negatives: {cm[1,0]:,}  | True Positives:  {cm[1,1]:,}")

# ==================== 10. VISUALIZATION ====================
print("\n[STEP 10] Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Class Distribution
ax = axes[0, 0]
fraud_counts = y_test.value_counts()
colors = ['#2ecc71', '#e74c3c']
ax.bar(['Legitimate', 'Fraud'], [fraud_counts[0], fraud_counts[1]], color=colors)
ax.set_ylabel('Count')
ax.set_title('Test Set Class Distribution\n(Highly Imbalanced)', fontsize=12, fontweight='bold')
for i, v in enumerate([fraud_counts[0], fraud_counts[1]]):
    ax.text(i, v + 200, f'{v:,}\n({v/len(y_test)*100:.1f}%)', ha='center', fontweight='bold')

# 2. Model Comparison
ax = axes[0, 1]
metrics_df = pd.DataFrame(results).T
metrics_df[['precision', 'recall', 'f1', 'roc_auc']].plot(kind='bar', ax=ax)
ax.set_ylabel('Score')
ax.set_title('Model Comparison - Key Metrics\n(Recall is crucial for fraud detection)', 
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylim([0, 1])

# 3. ROC Curves
ax = axes[1, 0]
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Model Comparison', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Precision-Recall Curve
ax = axes[1, 1]
for name, result in results.items():
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, result['y_pred_proba'])
    ax.plot(recall_vals, precision_vals, label=name, linewidth=2)
ax.set_xlabel('Recall (True Positive Rate)')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve\n(More relevant for imbalanced data)', 
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fraud_detection_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: fraud_detection_analysis.png")

# ==================== 11. CONFUSION MATRIX VISUALIZATION ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    axes[idx].set_title(f'{name}\n(Precision: {result["precision"]:.3f}, Recall: {result["recall"]:.3f})',
                       fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('/home/claude/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices.png")

# ==================== 12. FEATURE IMPORTANCE ====================
print("\n[STEP 11] Feature Importance Analysis...")

# Get feature importance from Random Forest
rf_model = trained_models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
top_features = feature_importance.head(10)
ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values)
ax.set_xlabel('Importance Score')
ax.set_title('Top 10 Most Important Features for Fraud Detection\n(Random Forest Model)',
            fontsize=12, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('/home/claude/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")

# ==================== 13. KEY FINDINGS & RECOMMENDATIONS ====================
print("\n" + "="*70)
print("KEY FINDINGS & RECOMMENDATIONS")
print("="*70)

print("""
1. WHY ACCURACY ISN'T EVERYTHING:
   - If we just predicted "Not Fraud" for everything, we'd get 98% accuracy!
   - But we'd catch 0 frauds (Recall = 0)
   - For fraud detection, missing frauds is more costly than false alarms
   
2. BETTER METRICS FOR IMBALANCED DATA:
   - Precision: "Of the frauds we detected, how many were correct?"
   - Recall: "Of all actual frauds, how many did we catch?"
   - F1-Score: Balanced metric when both matter
   - ROC-AUC: Shows model's ability across all thresholds
   
3. MODEL SELECTION INSIGHTS:
   - Random Forest and Gradient Boosting typically outperform Logistic Regression
   - Ensemble methods handle complex patterns better
   - Class imbalance handling (oversampling) significantly improves recall
   
4. THRESHOLD TUNING:
   - Default 0.5 probability threshold may not be optimal
   - For fraud detection, lower threshold (e.g., 0.3) catches more frauds
   - Trade-off: more false positives but fewer missed frauds
   
5. FEATURE INSIGHTS:
   - Distance from home: Frauds happen farther away
   - Transaction amount: Frauds tend to be larger
   - Time of day: Frauds happen more at night
   - Device type: Online transactions have higher fraud risk
""")

# ==================== 14. PRACTICAL DEPLOYMENT EXAMPLE ====================
print("\n" + "="*70)
print("PRACTICAL DEPLOYMENT EXAMPLE")
print("="*70)

print("\nHow to use the model in production:")
print("""
# Load the trained model
best_model = trained_models['Gradient Boosting']

# For a new transaction:
new_transaction = X_test_scaled[0:1]  # Example transaction
fraud_probability = best_model.predict_proba(new_transaction)[0, 1]

if fraud_probability > 0.3:  # Custom threshold
    print(f"⚠️  HIGH FRAUD RISK: {fraud_probability*100:.1f}%")
    # Action: Flag for review, block, or request verification
else:
    print(f"✓ Low fraud risk: {fraud_probability*100:.1f}%")
    # Action: Process normally
""")

print("\n" + "="*70)
print("PROJECT COMPLETE!")
print("="*70)
print(f"\nFiles created:")
print(f"  • fraud_detection_analysis.png - Main analysis visualizations")
print(f"  • confusion_matrices.png - Model confusion matrices")
print(f"  • feature_importance.png - Feature importance ranking")
print(f"\nDataset: credit_card_fraud.csv ({len(df):,} transactions)")
