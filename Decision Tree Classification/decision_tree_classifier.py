import pandas as pd
from sklearn.tree import export_text, DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def train_decision_tree(X, y):
    """Initialize and train the Decision Tree classifier."""
    clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf.fit(X, y)
    return clf

def make_predictions(clf, X):
    """Make predictions on the dataset using the trained classifier."""
    return clf.predict(X)

def print_decision_tree_rules(clf, feature_names):
    """Print the decision rules of the trained decision tree."""
    tree_rules = export_text(clf, feature_names=feature_names)
    print("")
    print("Decision Tree Rules:\n", tree_rules, sep="")

def plot_decision_tree(clf, feature_names, class_names):
    """Plot the decision tree."""
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()

def main():
    # Load data
    file_path = 'Decision Tree Classification/uts_data.csv'
    data = load_data(file_path)

    # Encode categorical features and the target variable
    label_encoder = LabelEncoder()
    data['Cuaca'] = label_encoder.fit_transform(data['Cuaca'])
    data['Angin'] = label_encoder.fit_transform(data['Angin'])
    y = label_encoder.fit_transform(data['Pergi atau Tidak'])
    
    # Set up independent variables (X) and dependent variable (y)
    X = data[['Cuaca', 'Temperatur', 'Kelembaban', 'Angin']]
    
    # Train the Decision Tree classifier
    clf = train_decision_tree(X, y)

    # Make predictions for the entire dataset
    predictions = make_predictions(clf, X)

    # Add predictions to the original DataFrame
    data['Predicted Class'] = label_encoder.inverse_transform(predictions)

    # Display the results
    print("\nDataset with Predictions:")
    print(data)

    # Print decision rules
    print_decision_tree_rules(clf, feature_names=['Cuaca', 'Temperatur', 'Kelembaban', 'Angin'])

    # Plot the decision tree
    plot_decision_tree(clf, feature_names=['Cuaca', 'Temperatur', 'Kelembaban', 'Angin'], class_names=label_encoder.classes_)

if __name__ == "__main__":
    main()