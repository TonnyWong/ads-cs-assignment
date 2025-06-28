# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv("data/processed_student_habits_performance.csv")

X = df[["study_hours_per_day", "screen_time", "attendance_percentage"]]
y = df["exam_score"]

# Pipeline
model = Pipeline([
    ("regressor", LinearRegression())
])

# Train
model.fit(X, y)

#Save model
joblib.dump(model, "exam_score_model.pkl")
print("Model saved as exam_score_model.pkl")
