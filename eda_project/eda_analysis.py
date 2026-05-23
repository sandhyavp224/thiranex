import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("titanic.csv")

df["sex_numeric"] = df["sex"].map({"male" : 0 , "female" : 1})

numeric_df = df[["survived", "pclass", "age", "fare", "sex_numeric"]]

correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


plt.figure(figsize = (8, 6))
sns.heatmap(correlation_matrix, annot = True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.savefig("correlation_heapmap.png")
plt.close()

# Survival by Passenger Class
plt.figure(figsize=(6 , 4))
sns.barplot(x="pclass",y="survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Average Survival Rate")
plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(
    df["age"], bins=10
)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.close()

# Report
total_passengers = len(df)
average_age = df["age"].mean()
survival_rate = df["survived"].mean()
class_survival = df.groupby("pclass")["survived"].mean()

with open("eda_report.txt", "w") as file:
    
    file.write("Titanic Exploratory Data Analysis Report\n")
    file.write("==========================================\n\n")
     
     #Dataset
    file.write(f"Total Passengers: {total_passengers}\n")
    file.write(f"Average Age: {average_age:.2f}\n")
    file.write(f"Overall Survival Rate: {survival_rate:.2%} \n\n")
    
    #Missing values
    file.write("Missing Values Summary:\n")
    file.write(str(df.isnull().sum()))
    file.write("\n\n")
    
    #correlation
    file.write("Correlation Iinsights:\n")
    file.write("-Gender had the strongest correlation wih survival\n")
    file.write("- Passenger class negatively correlated with survival.\n")
    file.write("- Higher fares were associated with better survival chances.\n\n")
    
    #survival
    file.write("Survival Rate by Passenger Class:\n")
    
    for passenger_class , rate in class_survival.items():
        file.write(f"Class {passenger_class}: {rate:.2%}\n")
    
    
    file.write("\nKey Finding:\n")
    file.write("- Female passengers had higher survival rate.\n") 
    file.write("- First-class passengers were more likely to survive.\n")
    file.write("- Most passengers were  young adults.\n")
    file.write("- Fare and class were strongly related.\n")
    
print("\nEDA report generated successfully!")
print("SAved as : eda_report.txt")
   
    
# print(df.columns)
# print(df.shape)
# print("\ndAtaset Information:")
# print(df.info())
# print("\nMissing Values in Each Columns:")
# print(df.isnull().sum())
# print("\nStatistical Summary of Numerical Columns:")
# print(df.describe())