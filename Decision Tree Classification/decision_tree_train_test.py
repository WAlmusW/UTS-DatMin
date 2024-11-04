import pandas as pd
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
file_path = 'Decision Tree Classification/decision_tree_data.csv'  # replace with your file name
data = pd.read_csv(file_path)

# Set up independent variables (X) and dependent variable (y)
# Assume your columns are named 'Income', 'LotSize', and 'Ownership'
X = data[['Income', 'Ukuran lot']]
y = data['Pemilik atau bukan']

# print(data['Income'].value_counts())
# print(data['Ukuran lot'].value_counts())
# print(data['Pemilik atau bukan'].value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Decision Tree classifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)

# Print decision rules
tree_rules = export_text(clf, feature_names=['Income', 'Ukuran lot'])
print(tree_rules)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=['Income', 'LotSize'], class_names=['Pemilik', 'Bukan Pemilik'], filled=True)
plt.show()