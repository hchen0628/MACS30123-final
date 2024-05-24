import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_js_divergence(data1, data2, features, bins=10):
    """Calculate Jensen-Shannon Divergence between two datasets for specified features."""
    js_divergences = []

    for feature in features:
        # Create histograms for each feature
        hist1, bin_edges = np.histogram(data1[feature], bins=bins, density=True)
        hist2, _ = np.histogram(data2[feature], bins=bin_edges, density=True)

        # Calculate Jensen-Shannon Divergence
        js_div = jensenshannon(hist1, hist2, base=2)
        js_divergences.append(js_div)
    
    return pd.DataFrame({'Feature': features, 'Jensen-Shannon Divergence': js_divergences})

# Load data
file_gen = r"C:\Users\A1157\Downloads\MACS30123\Xgen_ver2.csv"
file_orig = r"C:\Users\A1157\Downloads\MACS30123\Xorigin.csv"
data_gen = pd.read_csv(file_gen)
data_orig = pd.read_csv(file_orig)

# Define features for comparison
features = ['sex', 'race', 'marst', 'famsize', 'bpl', 'citizen', 'hispan', 'health', 'sect']

# Calculate JS Divergence
result_table = calculate_js_divergence(data_orig, data_gen, features)
print(result_table)

# Save results to CSV
result_table.to_csv(r"C:\Users\A1157\Downloads\MACS30123\js_divergence_results.csv", index=False)

# Visualize JS Divergence with seaborn
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
plt_table = ax.table(cellText=result_table.values, colLabels=result_table.columns, loc='center', cellLoc='center', colWidths=[0.2, 0.2], bbox=[0, 0, 1, 1])
plt_table.auto_set_font_size(False)
plt_table.set_fontsize(14)
plt_table.scale(1, 1.5)
plt.tight_layout()

# Save the figure as an image
plt.savefig('js_divergence_table.png', dpi=300)
plt.show()
plt.close()
