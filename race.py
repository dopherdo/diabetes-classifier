import pandas as pd

# Load the CSV
df = pd.read_csv("diabetic_data.csv")



def analyze_question_marks_percentage(file_path):
    df = pd.read_csv(file_path)
    question_mark_percentages = {}
    for column in df.columns:
        total_count = len(df[column])
        question_mark_count = (df[column] == '?').sum()
        percentage = (question_mark_count / total_count) * 100
        question_mark_percentages[column] = percentage
    return question_mark_percentages

# Finding number of question marks per category
result = analyze_question_marks_percentage("diabetic_data.csv")
print(result)



# Get unique values and counts in 'citoglipton'
unique_vals = df['citoglipton'].value_counts()

print("Unique values in 'citoglipton':")
print(unique_vals)

# Check if any value is not "No"
non_no = df[df['citoglipton'] != "No"]

print(f"\nNumber of rows NOT equal to 'No': {len(non_no)}")


