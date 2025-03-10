import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple, List
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

def split_data(
    raw_df: pd.DataFrame, 
    target_col: str,  
    test_size: float = 0.2, 
    val_size: float = 0.25, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        target_col (str): Name of the target column.
        test_size (float): Proportion of data to be used as the test set.
        val_size (float): Proportion of training data to be used as validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df.
    """

    # Check if target column exists
    if target_col not in raw_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Stratify only if target column is categorical
    stratify_col = raw_df[target_col] if raw_df[target_col].nunique() <= 10 else None

    # First split: Train + Validation and Test
    train_val_df, test_df = train_test_split(
        raw_df, test_size=test_size, stratify=stratify_col, random_state=random_state
    )

    # Second split: Train and Validation
    stratify_col = train_val_df[target_col] if stratify_col is not None else None
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, stratify=stratify_col, random_state=random_state
    )

    return train_df, val_df, test_df



def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numeric and categorical columns in the dataset.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        Tuple[List[str], List[str]]: List of numeric columns and list of categorical columns.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols


def encode_categorical_features(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Encodes categorical features using OneHotEncoder.

    Args:
        df (pd.DataFrame): DataFrame with categorical features.
        categorical_cols (List[str]): List of categorical feature names.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical features.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    return encoded_df, encoder


def scale_numeric_features(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Scales numeric features using StandardScaler.

    Args:
        df (pd.DataFrame): DataFrame with numeric features.
        numeric_cols (List[str]): List of numeric feature names.

    Returns:
        pd.DataFrame: DataFrame with scaled numeric features.
    """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def preprocess_data(
    raw_df: pd.DataFrame, 
    target_col: str, 
    drop_cols: List[str] = None,  
    scale_numeric: bool = True
) -> Dict[str, Any]: 
    """
    Preprocesses the raw dataframe.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        target_col (str): Name of the target column.
        drop_cols (List[str], optional): List of columns to drop. Defaults to None.
        scale_numeric (bool): Whether to scale numeric features.

    Returns:
        Dict[str, Any]: Dictionary containing processed inputs and targets for train, val, and test sets.
    """
    
    # Drop specified columns if provided
    if drop_cols:  
        raw_df = raw_df.drop(columns=drop_cols, errors="ignore")  # Avoids KeyError if column not found

    # Split data into train, validation, and test sets
    train_df, val_df, test_df = split_data(raw_df, target_col) 

    # Dynamically select feature columns (excluding target)
    input_cols = [col for col in raw_df.columns if col != target_col]
    
    # Select inputs and target variables
    train_inputs = train_df[input_cols]
    train_targets = train_df[target_col]
    
    val_inputs = val_df[input_cols]
    val_targets = val_df[target_col]

    # Identify numeric and categorical columns
    numeric_cols, categorical_cols = identify_column_types(train_inputs)

    # Scale numeric features
    if scale_numeric:
        train_inputs, scaler = scale_numeric_features(train_inputs, numeric_cols)
        val_inputs, _ = scale_numeric_features(val_inputs, numeric_cols)  # Scale validation data
    else:
        scaler = None

    # Encode categorical features
    encoded_train_df, encoder = encode_categorical_features(train_inputs, categorical_cols)
    encoded_val_df = encoder.transform(val_inputs[categorical_cols])
    encoded_val_df = pd.DataFrame(encoded_val_df, columns=encoder.get_feature_names_out(categorical_cols))

    # Ensure all categorical columns exist in new data
    missing_cols = set(encoder.get_feature_names_out(categorical_cols)) - set(encoded_val_df.columns)
    for col in missing_cols:
        encoded_val_df[col] = 0  # Fill missing columns with 0

    # Concatenate scaled numerical data and encoded categorical data
    train_inputs_processed = pd.concat([train_inputs[numeric_cols].reset_index(drop=True), encoded_train_df], axis=1)
    val_inputs_processed = pd.concat([val_inputs[numeric_cols].reset_index(drop=True), encoded_val_df], axis=1)

    return {
        'X_train': train_inputs_processed,
        'train_targets': train_targets,
        'X_val': val_inputs_processed,
        'val_targets': val_targets,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }


def preprocess_new_data(new_data: pd.DataFrame, input_cols: List[str], scaler: Any, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Preprocesses new data using the provided scaler and encoder.

    Args:
        new_data (pd.DataFrame): The new dataset to be processed.
        input_cols (List[str]): List of input feature names used in training.
        scaler (Any): Scaler used for numerical data (if applicable).
        encoder (OneHotEncoder): Encoder used for categorical data.

    Returns:
        pd.DataFrame: Processed new data.
    """
    new_inputs = new_data[input_cols].copy()  # Ensure a copy to avoid modifying the original DataFrame

    numeric_cols, categorical_cols = identify_column_types(new_inputs)

    # Scale numeric data if scaler is provided
    if scaler:
        new_inputs[numeric_cols] = scaler.transform(new_inputs[numeric_cols])

    # One-hot encode categorical data
    encoded_cats = encoder.transform(new_inputs[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

    # Ensure all columns match training
    missing_cols = set(encoder.get_feature_names_out(categorical_cols)) - set(encoded_df.columns)
    for col in missing_cols:
        encoded_df[col] = 0  # Fill missing columns with 0

    return pd.concat([new_inputs[numeric_cols].reset_index(drop=True), encoded_df], axis=1)

