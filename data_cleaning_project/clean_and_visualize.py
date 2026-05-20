import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("titanic.csv")

#Fill missing values in age
median_age = df["age"].median()
df["age"] = df["age"].fillna(median_age)

# Remove duplicate rows
df = df.drop_duplicates()


# #Detecgt and remove outliers in Fare
Q1 = df["fare"].quantile(0.25)
Q3 = df["fare"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df["fare"] >= lower_bound) & (df["fare"] <= upper_bound)]

#save cleaned dataset
df.to_csv("cleaned_titanic.csv", index = False)

# 1 : Survival by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x="sex", y="survived", data=df)
plt.title("Survival Rate by Gender")
plt.ylabel("Average Survival Rate")
plt.tight_layout()
plt.savefig("survival_by_gender.png")
plt.close()

# 2: Age Distribution

plt.figure(figsize=(6 , 4))
sns.histplot(df["age"], bins=10)
plt.title("Age Distirbution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passenges")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.close()


#create Analysis Report

survival_by_gender = df.groupby("sex")["survived"].mean()
average_age = df["age"].mean()
total_passengers = len(df)

with open("analysis_report.txt", "w") as file:
    file.write("Titanic Data Cleaning and Visualization Report\n")
    file.write("===========================================\n")
    file.write(f"Total Passengers After Cleaning: {total_passengers}\n")
    file.write(f"Average Age: {average_age:.2f}\n\n")
    file.write("Survival Rate by Gender:\n")
    for gender , rate in survival_by_gender.items():
        file.write(f"{gender}: {rate:.2%}\n ")
        
    file.write("\n Key Findings:\n")
    file.write("- Missing Age values were filed using the media.\n")
    file.write("- Duplicate row were removed.\n")
    file.write("- Outliers in fare were removed using theIQR method.n")
    file.write("- Female passengers generally had a higher survival rate.\n")
    
print("Project completed successfully")
print("Generated files:")    
print("- cleaned_titanic.csv")
print("- survival_by_gender.png")
print("- age_distribution.png")
# print("Visualization created successfully !")
# print("SAved files:")


# df.columns = df.columns.str.lower()  # Remove leading/trailing whitespace from column names
# #Display clearned dataset

# print("Cleaned Dataset Shapes:" , df.shape)

# #Verify missing values
# print(df.isnull().sum())

# print(df.columns)
df.to_csv("cleaned_titanic.csv", index = False)
# # print("\n Cleaned data saved to cleaned_titanic.csv")
print("Rows remaining after cleaning:", len(df))

# # print("First 5 rows")
# print(df.head())

# print("Missing values:")
# print(df.isnull().sum())

# print("Duplicate Rows:")
# print(df.duplicated().sum())

# print("Data Types:")
# print(df.dtypes)