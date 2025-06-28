import pandas as pd
import os

def load_and_process_data(filepath="data/student_habits_performance.csv", output_path="data/processed_student_habits_performance.csv"):
    """
    Loads the dataset, removes duplicate rows, handle null values, correct data types, and saves the processed dataset.

    Args:
        filepath (str): Path to the input dataset.
        output_path (str): Path to save the processed dataset.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Load the dataset
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Original dataset shape: {df.shape}")

    # Remove duplicate rows
    df = df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {df.shape}")

    # Handle null values
    null_summary = df.isnull().sum()
    print("Null values before handling:\n", null_summary[null_summary > 0])

    # Drop rows with nulls in critical columns
    critical_cols = ["exam_score", "social_media_hours", "netflix_hours", "study_hours_per_day", "attendance_percentage"]
    df.dropna(subset=critical_cols, inplace=True)

    # Fill numerical nulls with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical nulls with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"After handling nulls: {df.shape}")

    # Ensure data types are correct
    try:
        df["age"] = df["age"].astype(int)
        df["study_hours_per_day"] = df["study_hours_per_day"].astype(float)
        df["social_media_hours"] = df["social_media_hours"].astype(float)
        df["netflix_hours"] = df["netflix_hours"].astype(float)
        df["attendance_percentage"] = df["attendance_percentage"].astype(float)
        df["exam_score"] = df["exam_score"].astype(float)
    except Exception as e:
        print(f"Data type conversion failed: {e}")

    # Create screen_time column
    df["screen_time"] = df["social_media_hours"] + df["netflix_hours"]

    # Save the processed dataset
    output_path = os.path.abspath(output_path)
    try:
        print(f"Attempting to save to: {output_path}")
        df.to_csv(output_path, index=False)
        print("Processed dataset saved successfully!")
    except Exception as e:
        print(f"Failed to save file. Error: {e}")

    return df

if __name__ == "__main__":
    load_and_process_data()