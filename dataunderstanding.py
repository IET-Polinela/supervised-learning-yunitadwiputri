import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Calculate descriptive statistics
    stats = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    stats = stats.rename(columns={
        "50%": "Q2 (Median)",
        "25%": "Q1 (25%)",
        "75%": "Q3 (75%)"
    })
    stats["count"] = df.count()
    
    # Visualize numerical features distribution (boxplot)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df.select_dtypes(include=['number']))
    plt.xticks(rotation=45)
    plt.title("Boxplot of Numerical Features")
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig("boxplot_features.png")
    plt.close()
    
    # Visualize mean, median, and standard deviation
    plt.figure(figsize=(12, 6))
    stats[['mean', 'Q2 (Median)', 'std']].plot(kind='bar')
    plt.title("Descriptive Statistics (Mean, Median, Std Dev)")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("descriptive_statistics.png")
    plt.close()
    
    # Save descriptive statistics to CSV
    stats.to_csv("descriptive_statistics.csv")
    
    print("Visualizations and descriptive statistics have been saved as PNG and CSV files.")

# Main execution
if __name__ == "__main__":
    input_file = "train.csv"
    analyze_data(input_file)
