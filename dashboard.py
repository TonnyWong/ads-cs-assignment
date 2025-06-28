import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data():
    # Load the processed dataset
    return pd.read_csv("data/processed_student_habits_performance.csv")

def apply_filters(df):
    st.sidebar.header("🔍 Filter Options")

    # Gender filter
    genders = df["gender"].unique().tolist()
    selected_genders = st.sidebar.multiselect("Select Gender", genders, default=genders)

    # Age filter
    min_age, max_age = int(df["age"].min()), int(df["age"].max())
    selected_age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Apply filters
    filtered_df = df[
        (df["gender"].isin(selected_genders)) &
        (df["age"] >= selected_age_range[0]) &
        (df["age"] <= selected_age_range[1])
    ]
    return filtered_df

def categorize_screen_time(hours):
    if hours < 2:
        return "Low"
    elif hours < 5:
        return "Moderate"
    else:
        return "High"

def plot_study_hours_vs_exam_score(df):
    df["screen_time_category"] = df["screen_time"].apply(categorize_screen_time)
    st.subheader("Study Hours vs Exam Score (by Screen Time Category)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="study_hours_per_day", y="exam_score", hue="screen_time_category", ax=ax)
    ax.set_xlabel("Study Hours per Day")
    ax.set_ylabel("Exam Score")
    category_order = ["Low", "Moderate", "High"]
    ax.legend(title="Screen Time Category", loc="upper left", bbox_to_anchor=(1, 1),labels=category_order)
    st.pyplot(fig)

def plot_attendance_category_bar(df):
    # Define the attendance bins and labels
    bins = [50, 60, 70, 80, 90, 100]
    labels = ["51–60%", "61–70%", "71–80%", "81–90%", "91–100%"]

    # Clip values just in case and apply binning
    df["attendance_percentage"] = df["attendance_percentage"].clip(50, 100)
    df["attendance_bin"] = pd.cut(df["attendance_percentage"], bins=bins, labels=labels, include_lowest=True, right=True)

    # Group by attendance bin and calculate average exam score
    summary = df.groupby("attendance_bin")["exam_score"].mean().reset_index()

    # Plot the bar chart
    st.subheader("Attendance Percentage vs Exam Score")
    fig, ax = plt.subplots()
    sns.barplot(data=summary, x="attendance_bin", y="exam_score", palette="Blues_d", ax=ax)
    ax.set_xlabel("Attendance Percentage Range")
    ax.set_ylabel("Average Exam Score")
    st.pyplot(fig)

def main():
    st.title("Student Behaviour Influencing Academic Performance")
    df = load_data()

    # Apply filters
    df_filtered = apply_filters(df)

    # Summary metrics
    avg_exam_score = df_filtered["exam_score"].mean()
    avg_study_hours = df_filtered["study_hours_per_day"].mean()
    avg_screen_time = (df_filtered["social_media_hours"] + df_filtered["netflix_hours"]).mean()

    # Display metrics in columns
    st.markdown("### 📊 Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Exam Score", f"{avg_exam_score:.2f}")
    col2.metric("Avg Study Hours (hrs/day)", f"{avg_study_hours:.2f}")
    col3.metric("Average Screen Time (hrs/day)", f"{avg_screen_time:.2f}")

    plot_study_hours_vs_exam_score(df_filtered)
    plot_attendance_category_bar(df_filtered)

    st.subheader("Exam Score Prediction")

    # Load model
    model = joblib.load("exam_score_model.pkl")

    # User input
    Study_Hours_per_day = st.number_input("Study Hours (hrs/day)", min_value=0)
    Screen_Time = st.number_input("Screen Time (hrs/day)", min_value=0)
    Attendance_Percentage = st.slider("Attendance Percentage", 10, 100)

    # Prepare input
    input_df = pd.DataFrame({
        "study_hours_per_day": [Study_Hours_per_day],
        "screen_time": [Screen_Time],
        "attendance_percentage": [Attendance_Percentage]
    })

    # Predict
    predicted_exam_score = model.predict(input_df)[0]

    # Ensure the predicted score does not exceed 100%
    predicted_exam_score = min(predicted_exam_score, 100)

    st.metric("Predicted Exam Score", f"{predicted_exam_score:,.2f}%")

if __name__ == "__main__":
    main()