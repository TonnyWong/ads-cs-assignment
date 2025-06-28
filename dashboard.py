import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Load the processed dataset
    return pd.read_csv("data/processed_student_habits_performance.csv")

def plot_study_hours_vs_exam_score(df):
    st.subheader("Study Hours vs Exam Score")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="study_hours_per_day", y="exam_score", ax=ax)
    ax.set_xlabel("Study Hours per Day")
    ax.set_ylabel("Exam Score")
    ax.set_title("Effect of Study Hours on Exam Performance")
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
    ax.set_title("Academic Performance by Attendance Range")
    st.pyplot(fig)

def main():
    st.title("Student Behaviour Influencing Academic Performance")
    df = load_data()
    plot_study_hours_vs_exam_score(df)
    plot_attendance_category_bar(df)

if __name__ == "__main__":
    main()