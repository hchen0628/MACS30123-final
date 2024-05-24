import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns

def compare_datasets(path1, path2):
    """Calculate Wasserstein distance for each feature and return a DataFrame."""
    # Load data
    data_gen = pd.read_csv(path1)
    data_orig = pd.read_csv(path2)

    # Define all features for comparison
    features = ['sex', 'race', 'marst', 'famsize', 'bpl', 'citizen', 'hispan', 'health', 'sect']

    # Create an empty DataFrame to store results
    results = pd.DataFrame(columns=['Feature', 'Wasserstein Distance'])

    # Calculate Wasserstein distance for each feature
    for feature in features:
        # Convert categorical data to numeric codes if necessary
        if data_gen[feature].dtype == 'object' or data_gen[feature].dtype.name == 'category':
            data_gen[feature] = data_gen[feature].astype('category').cat.codes
            data_orig[feature] = data_orig[feature].astype('category').cat.codes

        distance = wasserstein_distance(data_orig[feature], data_gen[feature])
        new_row = {'Feature': feature, 'Wasserstein Distance': format(distance, '.4f')}  # Format the distance to 4 decimal places
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    return results

# File paths
file_gen = r"C:\Users\A1157\Downloads\MACS30123\Xgen_ver2.csv"
file_orig = r"C:\Users\A1157\Downloads\MACS30123\Xorigin.csv"

# Get results
result_table = compare_datasets(file_gen, file_orig)

# Styling with seaborn
sns.set(style="whitegrid")

# Create a figure for the table
fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the figure size
ax.axis('off')  # Hide the axes

# Display table with improved formatting
table = ax.table(cellText=result_table.values, colLabels=result_table.columns, loc='center', cellLoc='center', colWidths=[0.2, 0.2], bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(14)  # Increase the font size
table.scale(1, 1.5)  # Adjust the scaling to increase cell height

# Adjust layout
plt.tight_layout()

# Save the figure as an image
plt.savefig('comparison_table.png', dpi=300)
plt.show()
