import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load the training and test data
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

# Prepare the training data
def prepare_data(train_data, target_column, exclude_columns=[]):
    X_train = train_data.drop([target_column] + exclude_columns, axis=1)  # Drop target and excluded columns
    y_train = train_data[target_column]  # Labels for training
    return X_train, y_train

# Predict labels for test data
def predict_test_data(knn, test_data, exclude_columns=[]):
    # Drop excluded columns, ignoring errors if columns are not found
    X_test = test_data.drop(exclude_columns, axis=1, errors='ignore')  
    predictions = knn.predict(X_test)
    return predictions

# Main execution
if __name__ == "__main__":
    train_file = "SKLEARN KNN/uts_data.csv"  # Path to the training dataset
    test_file = "SKLEARN KNN/test_data.csv"  # Path to the test dataset
    target_column = "Kelas"                  # The target column for classification
    exclude_columns = ["Nomor"]              # Columns to exclude from training and testing
    k_values = [3, 5, 7]                     # List of k values to test

    # Load datasets
    train_data, test_data = load_data(train_file, test_file)

    # Prepare training data, excluding "Nomor"
    X_train, y_train = prepare_data(train_data, target_column, exclude_columns)
    
    # Loop through different values of k and make predictions
    for k in k_values:
        print(f"\nResults for k = {k}:")
        
        # Initialize and train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Use a copy of the test data without "Nomor" or any previous prediction columns
        test_data_copy = test_data.drop(columns=exclude_columns + [col for col in test_data.columns if 'Predicted_Kelas' in col], errors='ignore')
        
        # Predict for test data, excluding "Nomor"
        predictions = predict_test_data(knn, test_data_copy, exclude_columns)

        # Add predictions to the copy of test data and display results
        test_data_copy[f'Predicted_Kelas_k_{k}'] = predictions
        print(test_data_copy[['A', 'B', 'C', 'D', f'Predicted_Kelas_k_{k}']])
