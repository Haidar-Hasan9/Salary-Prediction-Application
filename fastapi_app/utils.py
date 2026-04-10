import pandas as pd
from fastapi_app.model_loader import load_model

def preprocess_input(experience_level, job_title, work_year, employee_residence):
    """
    Convert raw API inputs into the feature vector expected by the model.
    Must match exactly what was done in src/preprocess.py.
    """
    # Load encodings (which include the feature_cols and mappings)
    _, encodings = load_model()
    
    # Create a DataFrame with one row
    input_data = {
        'experience_level': [experience_level],
        'job_title': [job_title],
        'work_year': [work_year],
        'employee_residence': [employee_residence]
    }
    df_input = pd.DataFrame(input_data)
    
    # Apply same ordinal encoding for experience_level
    df_input['experience_level_enc'] = df_input['experience_level'].map(encodings['experience_map'])
    
    # Frequency encoding for job_title (use same frequency dict from training)
    # If job_title not seen in training, use the median frequency (0.0)
    freq = encodings['job_title_freq']
    df_input['job_title_enc'] = df_input['job_title'].map(freq).fillna(0.0)
    
    # High-cost location based on employee_residence
    high_cost_countries = ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'NL', 'CH', 'SG', 'IL', 'JP', 'KR', 'NO', 'SE', 'DK', 'FI']
    df_input['high_cost_location'] = df_input['employee_residence'].isin(high_cost_countries).astype(int)
    
    # Select features in the same order as training
    feature_cols = encodings['feature_cols']  # should be ['experience_level_enc', 'job_title_enc', 'work_year', 'high_cost_location']
    X = df_input[feature_cols]
    
    return X