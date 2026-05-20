import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



df = pd.read_csv("titanic.csv")

df["age"] = df["age"].fillna(df["age"].median())

#convert sex to numeric values
df["sex"] = df["sex"].map({"male": 0, "female":1})

#select feature 
X = df[["pclass", "sex" , "age", "fare"]]

#select target
y = df["survived"]

#splite a data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,                  #features
    y,                  #Target
    test_size=0.2,      #20% for testing
    random_state=42     #for reproducibility
    
)

#create the Random Forest model

model = RandomForestClassifier(
    n_estimators=100,   #number of trees in the forest
    random_state=42     #for reproducibility
)
#train a model
model.fit(X_train , y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Calculate accuracy
accuracy = accuracy_score(y_test , y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

#create cinfusion MAtric
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

#visualize confusion matrix

plt.figure(figsize=(6 , 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Coinfusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Acutal")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

#create a report
with open("model_report.txt" ,"w") as file:
    file.write("Titanic Survival Prediction Report\n")
    file.write("=================================\n"   )
    
    file.write(f"Model Accuracy: {accuracy:.2%}\n\n")
    file.write(str(cm))
    file.write("\n\n")
    
    file.write("Kehy Observations:\n")
    file.write("- The model was trained using Random Forst.\n")
    file.write("- Features used: pclass , sex, age, far.\n")
    file.write("- The model predicte whether a passenger survived.\n")
    file.write("- Accuracy indicates the percentage of correct predictions.\n")
    file.write("- The confusion matrix provieds detailed classification results.\n")
    


    #Display information
# print("Features (X):")
# print(X.head())

# print("\nTarget (y):")
# print(y.head())
# print("\nData preprocessing completed successfully")
#step 2: Train a machine learning model

# print("Training feature set shape:", X_train.shape)
# print("Testing feature set shape:", X_test.shape)
# print("Training target set shape:", y_train.shape)
# print("Testing target set shape:", y_test.shape)


#step3 : Display sample prediction
# print("First 10 Predictions:")
# print(y_pred[:10])

# step4:
# print("\nConfusion marix saved as confusion_matrix.png")
# print("Model evaluation completed successfully")

#step5
print("Model report saved as model._report.txt")
print("Project completed succesfully")



