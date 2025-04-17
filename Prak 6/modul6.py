# %%
# Loading library
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# %%
# Load data as a dataframe
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

iris = sklearn_to_df(datasets.load_iris())
iris.rename(columns={'target':'species'},inplace=True)

iris.describe().T

# %%
# %%
# Show the data
iris.head(10)
print(iris)

# %%
# %%
# Visualisasi data dengan Grafik pada data Iris
sns.pairplot(iris,hue='species',palette='Set1')

# %%
# %%
# Split training and testing data
from sklearn.model_selection import train_test_split
x = iris.drop('species', axis = 1)
y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 10)
print(len(x_train)) 
# %%
# %%
# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#%%
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7,7))

sns.set(font_scale=1.4) # for label size
sns.heatmap(cm, ax=ax,annot=True, annot_kws={"size": 16}) # font size

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# %%
#%%
features = ['sepal length (cm)','sepal width (cm)', 'petal length (cm)', 'petal width (cm)' ]

#%%
# Visualize tree
from sklearn import tree
fig, ax = plt.subplots(figsize=(25,20))
tree.plot_tree(model, feature_names=features)
plt.show()

# %%
# Example of creating a single Iris data point as a dictionary
iris_test_data = {
    'sepal length (cm)': 5.1,
    'sepal width (cm)': 3.5,
    'petal length (cm)': 1.4,
    'petal width (cm)': 0.1
}

# Ensure the order of features matches the training data
feature_order = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
prediction_input_df = pd.DataFrame([iris_test_data])
prediction = model.predict(prediction_input_df[feature_order]) # Ensure correct column order
print(prediction)
