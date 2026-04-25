"""
Generate synthetic credit card fraud dataset
This creates a realistic imbalanced dataset suitable for fraud detection modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_fraud_dataset(n_transactions=100000, fraud_ratio=0.02):
    """
    Generate synthetic credit card transaction data
    
    Parameters:
    - n_transactions: Total number of transactions
    - fraud_ratio: Percentage of fraudulent transactions (default 2%)
    """
    
    print(f"Generating {n_transactions:,} transactions with {fraud_ratio*100}% fraud rate...")
    
    # Calculate number of fraudulent transactions
    n_fraud = int(n_transactions * fraud_ratio)
    n_legit = n_transactions - n_fraud
    
    # Generate timestamps (spread over 365 days)
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(seconds=np.random.randint(0, 365*24*3600)) 
                  for _ in range(n_transactions)]
    
    # ==================== LEGITIMATE TRANSACTIONS ====================
    print("Generating legitimate transactions...")
    
    # Transaction amounts - typically lower and right-skewed
    legit_amounts = np.random.exponential(scale=150, size=n_legit)
    legit_amounts = np.clip(legit_amounts, 0.1, 10000)
    
    # Merchant categories
    merchant_cats_legit = np.random.choice(
        ['grocery', 'gas_station', 'restaurant', 'online_shopping', 
         'entertainment', 'healthcare', 'utility', 'travel'],
        size=n_legit,
        p=[0.25, 0.15, 0.2, 0.2, 0.08, 0.05, 0.04, 0.03]
    )
    
    # Distance from home (legitimate usually closer)
    distance_legit = np.abs(np.random.normal(loc=2, scale=5, size=n_legit))
    distance_legit = np.clip(distance_legit, 0, 100)
    
    # Time of transaction (legitimate more during business hours)
    p_legit = [0.02]*6 + [0.06]*6 + [0.05]*6 + [0.03]*6
    p_legit = np.array(p_legit) / sum(p_legit)  # Normalize to sum to 1
    hour_legit = np.random.choice(range(24), size=n_legit, p=p_legit)
    
    # Device type
    device_legit = np.random.choice(['mobile', 'online', 'ATM', 'in_person'],
                                     size=n_legit, p=[0.3, 0.4, 0.15, 0.15])
    
    # ==================== FRAUDULENT TRANSACTIONS ====================
    print("Generating fraudulent transactions...")
    
    # Fraud amounts - typically higher
    fraud_amounts = np.random.exponential(scale=300, size=n_fraud)
    fraud_amounts = np.clip(fraud_amounts, 50, 15000)
    
    # Fraud merchant categories (more online/unusual)
    merchant_cats_fraud = np.random.choice(
        ['online_shopping', 'travel', 'luxury_goods', 'jewelry', 
         'electronics', 'cryptocurrency', 'money_transfer'],
        size=n_fraud,
        p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05]
    )
    
    # Distance from home (fraudsters use different locations)
    distance_fraud = np.abs(np.random.normal(loc=50, scale=30, size=n_fraud))
    distance_fraud = np.clip(distance_fraud, 0, 200)
    
    # Time of transaction (fraud more at odd hours)
    p_fraud = [0.08]*6 + [0.02]*6 + [0.02]*6 + [0.08]*6
    p_fraud = np.array(p_fraud) / sum(p_fraud)  # Normalize to sum to 1
    hour_fraud = np.random.choice(range(24), size=n_fraud, p=p_fraud)
    
    # Device type (more likely online)
    device_fraud = np.random.choice(['mobile', 'online', 'stolen_card'],
                                    size=n_fraud, p=[0.2, 0.6, 0.2])
    
    # ==================== COMBINE ALL DATA ====================
    print("Combining datasets...")
    
    # Create DataFrames
    legit_df = pd.DataFrame({
        'amount': legit_amounts,
        'merchant_category': merchant_cats_legit,
        'distance_from_home': distance_legit,
        'hour_of_transaction': hour_legit,
        'device_type': device_legit,
        'is_fraud': 0
    })
    
    fraud_df = pd.DataFrame({
        'amount': fraud_amounts,
        'merchant_category': merchant_cats_fraud,
        'distance_from_home': distance_fraud,
        'hour_of_transaction': hour_fraud,
        'device_type': device_fraud,
        'is_fraud': 1
    })
    
    # Combine and shuffle
    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df['timestamp'] = timestamps
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add additional features
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    
    # Add customer ID and transaction ID
    df['transaction_id'] = range(1, len(df) + 1)
    df['customer_id'] = np.random.randint(1000, 1000 + len(df)//10, len(df))
    
    # Reorder columns
    df = df[['transaction_id', 'customer_id', 'timestamp', 'amount', 
             'merchant_category', 'distance_from_home', 'hour_of_transaction',
             'day_of_week', 'month', 'device_type', 'is_fraud']]
    
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_fraud_dataset(n_transactions=100000, fraud_ratio=0.02)
    
    # Save to CSV
    output_path = '/home/claude/credit_card_fraud.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset saved to {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Legitimate transactions: {(df['is_fraud']==0).sum():,} ({(1-df['is_fraud'].mean())*100:.2f}%)")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nBasic statistics:")
    print(df.describe())
