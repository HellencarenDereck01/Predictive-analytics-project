# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv("../data/student-mat.csv", sep=";")
# %%
df.head()
# %%
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
# %%
df.info()
# %%
df.describe()
# %%
df.isnull().sum()
# %%

# G3 represents the student's final grade.
# This will be the target variable for our prediction model.

print("Final Grade (G3) Statistics:")
print("Mean:", df["G3"].mean())
print("Median:", df["G3"].median())
print("Minimum:", df["G3"].min())
print("Maximum:", df["G3"].max())
# %%
plt.figure(figsize=(8, 5))

sns.histplot(df["G3"], bins=21, kde=True)

plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.show()
# %%
#  STUDY TIME DISTRIBUTION
# Count how many students belong to each study-time category.

studytime_counts = df["studytime"].value_counts().sort_index()

print(studytime_counts)
# %%
# VISUALIZE STUDY TIME

# Create a bar chart showing the number of students
# in each study-time category.

sns.countplot(data=df, x="studytime")

plt.title("Distribution of Weekly Study Time")
plt.xlabel("Study Time Category")
plt.ylabel("Number of Students")

plt.show()
# %%
# STEP 12: STUDY TIME VS FINAL GRADE
# Calculate the average final grade for each study-time category.

studytime_grade = df.groupby("studytime")["G3"].mean()
print(studytime_grade)
# %%
# STEP 12B: VISUALIZE STUDY TIME VS FINAL GRADE

# Create a bar chart showing the average final grade
# for each study-time category.

studytime_grade.plot(kind="bar", figsize=(8, 5))

plt.title("Average Final Grade by Study Time")
plt.xlabel("Study Time Category")
plt.ylabel("Average Final Grade")

plt.xticks(rotation=0)

plt.show()
# %%
# ==========================================
# ANALYZE STUDENT ABSENCES
# ==========================================

# Calculate basic statistics for student absences

print("Absence Statistics:")
print("Mean:", df["absences"].mean())
print("Median:", df["absences"].median())
print("Minimum:", df["absences"].min())
print("Maximum:", df["absences"].max())
# %%
# ==========================================
#  DISTRIBUTION OF ABSENCES
# ==========================================

# Visualize how absences are distributed among students

plt.figure(figsize=(8, 5))

sns.histplot(df["absences"], bins=20, kde=True)

plt.title("Distribution of Student Absences")
plt.xlabel("Number of Absences")
plt.ylabel("Number of Students")

plt.show()
# %%
# ==========================================
# ABSENCES VS FINAL GRADE
# ==========================================

# Create a scatter plot to examine the relationship
# between absences and final grade.

plt.figure(figsize=(8, 5))

sns.scatterplot(data=df, x="absences", y="G3")

plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.show()
# %%
# ==========================================
# ANALYZE PRIOR GRADES
# ==========================================

# G1 and G2 represent grades obtained earlier in the course.
# We will examine how they relate to the final grade (G3).

print("G1 Mean:", df["G1"].mean())
print("G2 Mean:", df["G2"].mean())
print("G3 Mean:", df["G3"].mean())
# %%
# ==========================================
#  G1 VS FINAL GRADE
# ==========================================

# Examine the relationship between the first-period grade
# and the student's final grade.

plt.figure(figsize=(8, 5))

sns.scatterplot(data=df, x="G1", y="G3")

plt.title("First-Period Grade (G1) vs Final Grade (G3)")
plt.xlabel("First-Period Grade (G1)")
plt.ylabel("Final Grade (G3)")

plt.show()
# %%
# ==========================================
# G2 VS FINAL GRADE
# ==========================================

# Examine the relationship between the second-period grade
# and the student's final grade.

plt.figure(figsize=(8, 5))

sns.scatterplot(data=df, x="G2", y="G3")

plt.title("Second-Period Grade (G2) vs Final Grade (G3)")
plt.xlabel("Second-Period Grade (G2)")
plt.ylabel("Final Grade (G3)")

plt.show()
# %%
# ==========================================
# CORRELATION ANALYSIS
# ==========================================
# Correlation values range from -1 to +1:
#
# +1 = strong positive relationship
#  0 = little or no linear relationship
# -1 = strong negative relationship
correlation = df[["studytime", "absences", "G1", "G2", "G3"]].corr()
print(correlation)
# %%
# ==========================================
# CORRELATION HEATMAP
# ==========================================
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation Matrix of Key Variables")
plt.show()
# %%
# ==========================================
# LIST ALL DATASET VARIABLES
# ==========================================

# Display the names of all variables in the dataset.
print("Dataset variables:")
for column in df.columns:
    print("-", column)
# %%
# ==========================================
# IDENTIFY DATA TYPES
# ==========================================

# Identify numerical and categorical variables
# in the dataset.

numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns

categorical_columns = df.select_dtypes(
        include=["str"]
    ).columns

print("Numerical variables:")
print(list(numerical_columns))

print("\nCategorical variables:")
print(list(categorical_columns))
# %%
# ==========================================
#CHECK FOR OUTLIERS
# ==========================================

# We use the IQR (Interquartile Range) method
# to identify unusually high or low values.

important_variables = [
    "age",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2",
    "G3"
]

for column in important_variables:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"{column}: {len(outliers)} potential outliers")

# %%
# ==========================================
# OUTLIER VISUALIZATION
# ==========================================

# We will examine each important variable separately.
# This makes potential outliers easier to identify.

variables_to_plot = [
    "age",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2",
    "G3"
]

for column in variables_to_plot:

    plt.figure(figsize=(7, 4))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)

    plt.show()
# %%
# ==========================================
# STEP 21B: ABSENCE OUTLIERS
# ==========================================

    # Absences contains several unusually high values.
    # We examine it separately to understand these observations.

plt.figure(figsize=(10, 4))

sns.boxplot(x=df["absences"])

plt.title("Boxplot of Student Absences")
plt.xlabel("Number of Absences")
plt.show()
# %%
# ==========================================
# INVESTIGATE ZERO FINAL GRADES
# ==========================================

# Count students who received a final grade of zero.

zero_final_grades = (df["G3"] == 0).sum()

print("Students with G3 = 0:", zero_final_grades)

# Calculate the percentage of students with G3 = 0.

zero_percentage = (zero_final_grades / len(df)) * 100

print(f"Percentage of students with G3 = 0: {zero_percentage:.2f}%")
# %%
# ==========================================
# ANALYZE STUDENTS WITH G3 = 0
# ==========================================

# Select students whose final grade is zero.

zero_grade_students = df[df["G3"] == 0]

# Compare their key variables.

print("Average values for students with G3 = 0:")
print(
    zero_grade_students[
        ["studytime", "absences", "G1", "G2"]
    ].mean()
)
# %%
# ==========================================
# COMPARE ZERO-GRADE STUDENTS
# ==========================================

# Compare students with G3 = 0 against all students.

comparison = pd.DataFrame({
    "All Students": df[["studytime", "absences", "G1", "G2", "G3"]].mean(),
    "G3 = 0 Students": zero_grade_students[
        ["studytime", "absences", "G1", "G2", "G3"]
    ].mean()
})

print(comparison)
