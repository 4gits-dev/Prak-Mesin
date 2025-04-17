# %% # Loading libraries
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import requests
from io import StringIO

# %% # Load the Raisin dataset (downloading from a URL)
# The dataset contains measurements of Kecimen and Besni raisin varieties

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00617/Raisin_Dataset.csv"
try:
    response = requests.get(url)
    if response.status_code == 200:
        raisin_data = pd.read_csv(StringIO(response.text))
    else:
        # If download fails, create a simulated dataset
        raise Exception("Could not download the dataset")
except:
    # Create a simulated raisin dataset
    np.random.seed(42)
    n_samples = 900
    
    # Generate features
    area = np.random.normal(45000, 15000, n_samples)
    perimeter = np.random.normal(800, 150, n_samples)
    major_axis_length = np.random.normal(300, 50, n_samples)
    minor_axis_length = np.random.normal(200, 40, n_samples)
    eccentricity = np.random.uniform(0.5, 0.95, n_samples)
    convex_area = area * np.random.uniform(1.0, 1.2, n_samples)
    extent = np.random.uniform(0.6, 0.85, n_samples)
    
    # Create classes (0: Kecimen, 1: Besni)
    class_target = np.zeros(n_samples, dtype=int)
    
    # Make Besni raisins (class 1) generally larger
    besni_indices = np.random.choice(n_samples, n_samples // 2, replace=False)
    class_target[besni_indices] = 1
    
    # Adjust features based on class
    area[besni_indices] += 10000
    perimeter[besni_indices] += 100
    major_axis_length[besni_indices] += 30
    minor_axis_length[besni_indices] += 20
    
    # Create dataframe
    raisin_data = pd.DataFrame({
        'Area': area,
        'Perimeter': perimeter,
        'MajorAxisLength': major_axis_length,
        'MinorAxisLength': minor_axis_length,
        'Eccentricity': eccentricity,
        'ConvexArea': convex_area,
        'Extent': extent,
        'Class': ['Kecimen' if c == 0 else 'Besni' for c in class_target]
    })

# %% # Process the dataset
# If we downloaded the real dataset, it might need some cleaning
if 'Class' not in raisin_data.columns:
    if 'Class_Kecimen' in raisin_data.columns:
        # If the dataset has a different structure than expected
        raisin_data['Class'] = raisin_data['Class_Kecimen'].map({1: 'Kecimen', 0: 'Besni'})
    else:
        # Rename the last column to Class if it exists but with a different name
        last_col = raisin_data.columns[-1]
        raisin_data.rename(columns={last_col: 'Class'}, inplace=True)

# Convert Class to numeric if it's categorical
if raisin_data['Class'].dtype == 'object':
    class_mapping = {name: i for i, name in enumerate(raisin_data['Class'].unique())}
    raisin_data['class_numeric'] = raisin_data['Class'].map(class_mapping)
else:
    raisin_data['class_numeric'] = raisin_data['Class']
    raisin_data['Class'] = raisin_data['class_numeric'].map({i: f'Class_{i}' for i in raisin_data['class_numeric'].unique()})

# %% # Show data description
print(raisin_data.describe().T)

# %% # Show the data
print(raisin_data.head(10))

# %% # Data visualization with pairplot
feature_cols = [col for col in raisin_data.columns if col not in ['Class', 'class_numeric']]
sns.pairplot(raisin_data, hue='Class', palette='Set1', vars=feature_cols[:4])  # Limit to first 4 features for clarity

# %% # Split training and testing data
X = raisin_data[feature_cols]
y = raisin_data['class_numeric']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# %% # Show training data size
print(f"Training data size: {len(X_train)}")

# %% # Train model (Decision Tree)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# %% # Predict test data
y_pred = model.predict(X_test)

# %% # Classification report
print(classification_report(y_test, y_pred))

# %% # Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 7))

sns.set(font_scale=1.4)
sns.heatmap(cm, ax=ax, annot=True, annot_kws={"size": 16})

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

# %% # Visualize tree
features = X_train.columns.tolist()
class_names = [str(cls) for cls in model.classes_]
fig, ax = plt.subplots(figsize=(25, 20))
tree.plot_tree(model, feature_names=features, class_names=class_names, filled=True)
plt.show()

# %% # Print feature names
print(X_train.columns.tolist())

# %% # Create a sample raisin data point for prediction
raisin_test_data = {
    'Area': 50000,
    'Perimeter': 850,
    'MajorAxisLength': 320,
    'MinorAxisLength': 210,
    'Eccentricity': 0.75,
    'ConvexArea': 55000,
    'Extent': 0.70
}

# Ensure correct column order
feature_order = X_train.columns.tolist()
prediction_input_df = pd.DataFrame([raisin_test_data])
prediction = model.predict(prediction_input_df[feature_order])
class_names = {i: name for name, i in class_mapping.items()} if 'class_mapping' in locals() else {0: 'Kecimen', 1: 'Besni'}
predicted_class = class_names.get(prediction[0], f"Class_{prediction[0]}")
print(f"Prediksi kelas: {prediction[0]} ({predicted_class})")
# %%
