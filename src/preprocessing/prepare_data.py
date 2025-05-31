import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_and_prepare_data(df: pd.DataFrame, target_column: str = 'team1_win', 
                          test_size: float = 0.20, random_state: int = 42):
    """
    Split data into features and target, then into train and test sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_column (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    feature_names = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_names

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler 