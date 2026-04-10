import pandas as pd
import joblib
from pathlib import Path

def load_data():
    """Load raw dataset from data/raw/"""
    df = pd.read_csv("data/raw/ds_salaries.csv")
    return df

def encode_categorical(df):
    """
    Encode categorical features for Decision Tree.
    Returns: X (features), y (target), and encoding mappings
    """
    df_encoded = df.copy()
    
    # Ordinal mappings (order matters for Decision Tree)
    experience_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    employment_map = {'FT': 0, 'PT': 1, 'CT': 2, 'FL': 3}
    company_size_map = {'S': 0, 'M': 1, 'L': 2}
    
    # Apply ordinal encoding
    df_encoded['experience_level_enc'] = df_encoded['experience_level'].map(experience_map)
    df_encoded['employment_type_enc'] = df_encoded['employment_type'].map(employment_map)
    df_encoded['company_size_enc'] = df_encoded['company_size'].map(company_size_map)
    
    # Frequency encoding for job_title
    job_title_freq = df_encoded['job_title'].value_counts(normalize=True)
    df_encoded['job_title_enc'] = df_encoded['job_title'].map(job_title_freq)
    
    # New feature: high-cost location based on employee_residence
    high_cost_countries = ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'NL', 'CH', 'SG', 'IL', 'JP', 'KR', 'NO', 'SE', 'DK', 'FI']
    df_encoded['high_cost_location'] = df_encoded['employee_residence'].isin(high_cost_countries).astype(int)
    
    # work_year is already numerical, keep as is
    # No need to create a separate column, just use df_encoded['work_year']
    
    # Feature columns (INCLUDING the new ones)
    feature_cols  = [
    'experience_level_enc',
    'job_title_enc',
    'work_year',
    'high_cost_location'
]     # added
    
    
    X = df_encoded[feature_cols]
    y = df_encoded['salary_in_usd']
    
    # Save only necessary mappings (not the whole columns)
    mappings = {
        'experience_map': experience_map,
        'employment_map': employment_map,
        'company_size_map': company_size_map,
        'job_title_freq': job_title_freq.to_dict(),
        'feature_cols': feature_cols
        # Do NOT store work_year or high_cost_location columns here
    }
    
    return X, y, mappings

def split_data(X, y, test_size=0.2, random_state=42):
    """Train/test split"""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Quick test
    df = load_data()
    X, y, mappings = encode_categorical(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print("\nMappings keys:", list(mappings.keys()))
    
    # Save mappings
    Path("models").mkdir(exist_ok=True)
    joblib.dump(mappings, "models/encodings.pkl")
    print("Saved encodings to models/encodings.pkl")