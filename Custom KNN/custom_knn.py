import pandas as pd

# Define similarity matrices based on given information
similarity_matrices = {
    "Kriteria1": {("<30", "<30"): 1, (">30", ">30"): 1, ("<30", ">30"): 0.4, (">30", "<30"): 0.4},
    "Kriteria2": {("Tinggi", "Tinggi"): 1, ("Rendah", "Rendah"): 1, ("Tinggi", "Rendah"): 0.5, ("Rendah", "Tinggi"): 0.5},
    "Kriteria3": {("Baik", "Baik"): 1, ("Tidak", "Tidak"): 1, ("Baik", "Tidak"): 0.75, ("Tidak", "Baik"): 0.75}
}

# Define attribute weights
weights = {
    "Kriteria1": 0.5,
    "Kriteria2": 0.75,
    "Kriteria3": 1
}

def load_training_data(file_path):
    """Load the training data from a CSV file."""
    return pd.read_csv(file_path)

def calculate_similarity(row, new_customer, weights, similarity_matrices):
    """Calculate similarity between new customer and each training instance."""
    total_similarity = 0
    for criterion in weights.keys():
        value_1 = row[criterion]
        value_2 = new_customer[criterion]
        similarity = similarity_matrices[criterion].get((value_1, value_2), 0)
        total_similarity += similarity * weights[criterion]
    return total_similarity / sum(weights.values())

def determine_k(data_length):
    """Determine the value of K based on the dataset length."""
    return max(1, data_length // 5)

def make_recommendations(train_data, new_customers, weights, similarity_matrices):
    """Calculate recommendations for new customers based on similarity."""
    recommendations = []
    
    for new_customer in new_customers:
        # Calculate similarity for each row in the training data
        train_data["Similarity"] = train_data.apply(calculate_similarity, axis=1, args=(new_customer, weights, similarity_matrices))
        
        # Determine K based on the number of training samples
        # k = determine_k(len(train_data))
        k = 3
        top_k_neighbors = train_data.nlargest(k, "Similarity")
        
        # Determine the majority class for top K neighbors
        recommendation = top_k_neighbors["Keterangan"].mode()[0]
        recommendations.append(recommendation)

        print("\nTraining Data with Similarity Scores:\n", train_data)
        print(f"Dynamic K (max(1, Dataset Length // 5)) = {k}\n")
    
    return recommendations

def main():
    # Load the training data
    train_file = "Custom KNN/uts_data.csv"
    train_data = load_training_data(train_file)

    # Define the new customer data
    new_customers = [
        {
            "A": "235.5", 
            "B": "30.5", 
            "C": "2", 
            "D": "1.45"
        },
    ]

    # Make recommendations for new customers
    recommendations = make_recommendations(train_data, new_customers, weights, similarity_matrices)

    # Output the recommendation results
    first_key = list(new_customers[0].keys())[0]
    for i, rec in enumerate(recommendations):
        print(f"Recommendation for {new_customers[i][first_key]}: {rec}\n")

if __name__ == "__main__":
    main()
