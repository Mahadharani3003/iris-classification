# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load dataset
iris = load_iris()

# Step 3: Create DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("First 5 rows:")
print(df.head())

# Step 4: Split data
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Step 5: Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
# Step 6: Predict
y_pred = model.predict(X_test)
# Step 7: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Step 8: Test with custom input
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
print("Predicted class:", iris.target_names[prediction])