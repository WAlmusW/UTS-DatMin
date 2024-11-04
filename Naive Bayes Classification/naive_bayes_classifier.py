import pandas as pd
from collections import defaultdict
from math import prod, exp, sqrt, pi

def load_data(file_path):
    """Load CSV data."""
    return pd.read_csv(file_path)

def calculate_prior_probabilities(df, class_column):
    """Calculate prior probabilities for each class."""
    total_count = len(df)
    class_counts = df[class_column].value_counts()
    return {cls: count / total_count for cls, count in class_counts.items()}

def calculate_continuous_likelihood(value, mean, std):
    """Calculate likelihood for a continuous feature using Gaussian distribution."""
    if std == 0:
        return 1 if value == mean else 0
    exponent = exp(-((value - mean) ** 2 / (2 * std ** 2)))
    return (1 / (sqrt(2 * pi) * std)) * exponent

def calculate_likelihoods(df, input_data_list, class_column, laplace_correction=True):
    """Calculate likelihoods for multiple input data with optional Laplacian correction if enabled."""
    likelihoods = defaultdict(lambda: defaultdict(dict))
    classes = df[class_column].unique()
    
    for cls in classes:
        class_df = df[df[class_column] == cls]
        for input_data in input_data_list:
            for feature, feature_value in input_data.items():
                if feature == "IP Semester 1-6":
                    # Treat as numeric feature and use ranges
                    numeric_value = float(feature_value)
                    mean = class_df[feature].astype(float).mean()
                    std = class_df[feature].astype(float).std()
                    smoothed_prob = calculate_continuous_likelihood(numeric_value, mean, std)
                else:
                    # Categorical feature with optional Laplace correction
                    count = class_df[class_df[feature] == feature_value].shape[0]
                    total_class_count = class_df.shape[0]
                    if laplace_correction:
                        smoothed_prob = (count + 1) / (total_class_count + len(df[feature].unique()))
                    else:
                        smoothed_prob = count / total_class_count if total_class_count > 0 else 0
                # Store the likelihood for this class and input data
                likelihoods[cls][feature][tuple(input_data.items())] = smoothed_prob
    return likelihoods

def classify_multiple(input_data_list, priors, likelihoods):
    """Classify multiple input data based on computed priors and likelihoods."""
    predictions = []
    for input_data in input_data_list:
        posteriors = {}
        for cls, prior in priors.items():
            class_likelihood = prod([likelihoods[cls][feature].get(tuple(input_data.items()), 1) for feature in input_data])
            posteriors[cls] = prior * class_likelihood
        predicted_class = max(posteriors, key=posteriors.get)
        predictions.append((predicted_class, posteriors))
    return predictions

def generate_classification_rules(df, class_column):
    """Generate readable classification rules based on likelihoods."""
    rules = {}
    classes = df[class_column].unique()
    for cls in classes:
        rules[cls] = []
        class_df = df[df[class_column] == cls]
        for feature in df.columns:
            if feature == class_column or feature == "No":
                continue
            if feature == "IP Semester 1-6":
                # Numeric feature: use mean and std as a rough rule for this class
                mean = class_df[feature].astype(float).mean()
                std = class_df[feature].astype(float).std()
                rules[cls].append(f"{feature} is around {mean:.2f} ± {std:.2f}")
            else:
                # Categorical feature: find the most common value for this class
                mode_value = class_df[feature].mode()[0]
                rules[cls].append(f"{feature} is likely '{mode_value}'")
    return rules

def main():
    # Load the dataset
    file_path = 'Naive Bayes Classification/uts_data.csv'
    df = load_data(file_path)
    
    # Specify the class column
    class_column = 'Kelulusan'  # Adjust this based on the CSV
    
    # Calculate priors
    priors = calculate_prior_probabilities(df, class_column)
    
    # Define input data for prediction
    input_data_list = [
        {
            'Jenis Kelamin': 'Laki-Laki',
            'Status Mahasiswa': 'Mahasiswa',
            'Status Pernikahan': 'Menikah',
            'IP Semester 1-6': '2.75',
        }
    ]
    
    # Calculate likelihoods for all input data
    likelihoods = calculate_likelihoods(df, input_data_list, class_column, laplace_correction=True)
    
    # Perform classification for multiple input data
    predictions = classify_multiple(input_data_list, priors, likelihoods)
    
    # Generate classification rules
    rules = generate_classification_rules(df, class_column)
    
    # Print results
    print(df)
    for i, (prediction, posteriors) in enumerate(predictions):
        values_of_predictor = list(input_data_list[i].values())
        print("")
        print(f"Prediction for {values_of_predictor}: {prediction}")
        print("Posterior probabilities:")
        for cls, prob in posteriors.items():
            print(f"{cls}: {prob:.5f}")
        print("")
    
    # Print classification rules
    print("\nClassification Rules:")
    for cls, conditions in rules.items():
        print(f"\nClass '{cls}' will happen when:")
        for condition in conditions:
            print(f" - {condition}")

if __name__ == "__main__":
    main()
