import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("retail_sales.csv")

category_sales = df.groupby("Category")["Sales"].sum()
plt.figure(figsize=(6 , 4))
sns.barplot(
    x = category_sales.index,
    y=category_sales.values
)
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig("sales_by_category.png")
plt.close()

print("Sales by Category chart saved successfully")

product_sales = df.groupby("Product")["Sales"].sum()
plt.figure(figsize=(8,5))
sns.barplot(
    x=product_sales.index,
    y=product_sales.values
)
plt.title("Top Products by Sales")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_products.png")
plt.close()
print("Top products chart saved")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])
daily_sales = df.groupby("Order_Date")["Sales"].sum()
plt.figure(figsize=(8 , 5))

plt.plot(
    daily_sales.index,
    daily_sales.values,
    marker="o"
)
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales_trend.png")
plt.close()


# print("First 5 rows")
# print(df.head())

# print("\nColumn Names:")
# print(df.columns)

# print("\nDataset Shape:")
# print(df.shape)

# print("\nStatistical Summary")
# print(df.describe())

# total_sales = df["Sales"].sum()
# print("\nTotal Sales:" , total_sales)

# total_profit = df["Profit"].sum()
# print("Total Profits:",total_profit)

# average_sales = df["Sales"].mean()
# print("Average Sales:", average_sales)

# #Find a row with highest sales value
# top_product = df.loc[df["Sales"].idxmax()]

# print("\nTop Performing Product")
# print(top_product)

# category_sales  = df.groupby("Category")["Sales"].sum()

# print("\nSales by Category:",category_sales)