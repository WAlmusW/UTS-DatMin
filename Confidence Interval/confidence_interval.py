import pandas as pd
import scipy.stats as stats
import numpy as np

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def calculate_statistics(completion_times):
    """Calculate mean and standard deviation of completion times."""
    mean_time = np.mean(completion_times)
    std_dev = np.std(completion_times, ddof=1)  # Using sample standard deviation
    return mean_time, std_dev

def calculate_confidence_interval(mean_time, std_dev, sample_size, confidence_level=0.95):
    """Calculate the confidence interval for the mean."""
    degrees_freedom = sample_size - 1
    confidence_interval = stats.t.interval(
        confidence_level,
        degrees_freedom,
        loc=mean_time,
        scale=std_dev / np.sqrt(sample_size)
    )
    return confidence_interval

def main():
    # Load the data from CSV
    file_path = 'Confidence Interval/completion_times.csv'
    data = load_data(file_path)

    # Extract completion times
    variable_to_calculate = data['WaktuPenyelesaian']  # Adjust this based on the CSV

    # Calculate mean and standard deviation
    mean_time, std_dev = calculate_statistics(variable_to_calculate)

    # Calculate confidence interval
    confidence_interval = calculate_confidence_interval(mean_time, std_dev, len(variable_to_calculate))

    # Output results
    print("")
    print(f"Mean WaktuPenyelesaian: {mean_time:.2f}")
    print(f"95% Confidence Interval for WaktuPenyelesaian: ({confidence_interval[0]:.5f}, {confidence_interval[1]:.5f})")
    print("")

if __name__ == "__main__":
    main()
