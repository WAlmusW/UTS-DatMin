import pandas as pd
from collections import defaultdict
from math import prod

def load_data(file_path):
    """Load CSV data."""
    return pd.read_csv(file_path)

def calculate_prior_probabilities(df, class_column):
    """Calculate prior probabilities for each class."""
    total_count = len(df)
    class_counts = df[class_column].value_counts()
    return {cls: count / total_count for cls, count in class_counts.items()}

def calculate_likelihoods(df, input_data_list, class_column, laplace_smoothing=True):
    """Calculate likelihoods for multiple input data with optional Laplace smoothing if enabled."""
    likelihoods = defaultdict(lambda: defaultdict(dict))
    classes = df[class_column].unique()
    
    for cls in classes:
        class_df = df[df[class_column] == cls]
        for input_data in input_data_list:
            for feature in input_data:
                feature_value = input_data[feature]
                count = class_df[class_df[feature] == feature_value].shape[0]
                total_class_count = class_df.shape[0]
                # Apply Laplace smoothing if enabled
                if laplace_smoothing:
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
            class_likelihood = prod([likelihoods[cls][feature].get(tuple(input_data.items()), 0) for feature in input_data])
            posteriors[cls] = prior * class_likelihood
        predicted_class = max(posteriors, key=posteriors.get)
        predictions.append((predicted_class, posteriors))
    return predictions

def main():
    # Load the dataset
    file_path = 'Naive Bayes Classification/popular_cars.csv'
    df = load_data(file_path)
    
    # Specify the class column
    class_column = 'Terlaris (a4)'  # Adjust this based on the CSV
    
    # Calculate priors
    priors = calculate_prior_probabilities(df, class_column)
    
    # Define input data for prediction
    input_data_list = [
        {
            'Warna (a1)': 'kuning',
            'Tipe (a2)': 'SUV', 
            'Asal (a3)': 'Domestik', 
        },
    ]
    
    # Calculate likelihoods for all input data
    likelihoods = calculate_likelihoods(df, input_data_list, class_column, laplace_smoothing=False)
    
    # Perform classification for multiple input data
    predictions = classify_multiple(input_data_list, priors, likelihoods)
    
    # Print results
    for i, (prediction, posteriors) in enumerate(predictions):
        print("")
        print(f"Prediction for Car {input_data_list[i].get('Warna (a1)', '')}, {input_data_list[i].get('Tipe (a2)', '')}, {input_data_list[i].get('Asal (a3)', '')}: {prediction}")
        print("Posterior probabilities:")
        for cls, prob in posteriors.items():
            print(f"{cls}: {prob:.5f}")
        print("")

if __name__ == "__main__":
    main()
