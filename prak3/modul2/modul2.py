# %%
import pandas as pd

# %%
# reading the database
data = pd.read_csv("DatasetForCoffeeSales2.csv")

# %%
# printing the top 10 rows
display(data.head(10))

# %%
# Filter data only City, Category, Product, Quantity are showed

data.filter(["City", "Category", "Product", "Quantity"]).head()

# %%
# Sorting quantity (most)
data.sort_values("Quantity", axis=0, ascending=False, inplace=True, na_position="last")
display(data.head())

# %%
# Group Total Sales by City
grp1 = data.groupby('City')
result = grp1['Final Sales'].aggregate('sum')
print(result)
