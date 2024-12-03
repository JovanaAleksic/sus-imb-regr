import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sus import SUS, SUSiter
from sampling import *
from learning import *
from metrics import *
from utils import *

def create_imbalanced_dataset():
    """Create a simple 2D dataset with imbalanced target values"""
    # Create two moons dataset
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    # Generate target values based on x,y position
    y = 10 * np.exp(-((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2))
    
    # Add some rare high values
    rare_points = np.random.uniform(low=-2, high=2, size=(20, 2))
    rare_values = np.random.uniform(low=15, high=20, size=20)
    
    X = np.vstack([X, rare_points])
    y = np.concatenate([y, rare_values])

    # Create DataFrame
    df = pd.DataFrame(data=np.column_stack([y, X]), 
                     columns=['target', 'feature_1', 'feature_2'])
    
    return X, y, df

def visualize_dataset(X, y, title="Dataset"):
    """Visualize the 2D dataset with target values as colors"""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()

# Create and visualize test dataset
X, y, df = create_imbalanced_dataset()


# Test SUS
sus = SUS(k=7, blobtr=0.75, spreadtr=0.5)
(X_rare, y_rare), (X_normal, y_normal) = sus._split_rare_normal(X, y)
X_resampled, y_resampled = sus.fit_resample(X, y)



# create instance of a class
data = DataProcess(df)
data.compute_phi(0.8)

X_sampled, y_sampled = SUSReg(data, 7, 0.75, 0.5).sample()

print("SUS v2:")
print(f"{X_resampled} \n {y_resampled}")

print("SUS v1:")
print(f"{X_sampled} \n {y_sampled}")

visualize_dataset(X, y, "Original Dataset")
visualize_dataset(X_resampled, y_resampled, "SUS v2")
visualize_dataset(X_sampled, y_sampled, "SUS v1")





# visualize_dataset(X_resampled, y_resampled, "After SUS")

# # Print some statistics
# print(f"Original dataset size: {len(X)}")
# print(f"Resampled dataset size: {len(X_resampled)}")

# # Test relevance computation
# relevance = sus._compute_relevance(y)
# plt.figure(figsize=(10, 4))
# plt.scatter(range(len(y)), relevance)
# plt.title("Relevance Scores")
# plt.show()

# # Test rare/normal split
# (X_rare, y_rare), (X_normal, y_normal) = sus._split_rare_normal(X, y)
# print(f"\nRare samples: {len(X_rare)}")
# print(f"Normal samples: {len(X_normal)}")

# # Visualize rare vs normal split
# plt.figure(figsize=(10, 6))
# plt.scatter(X_normal[:, 0], X_normal[:, 1], c='blue', label='Normal', alpha=0.5)
# plt.scatter(X_rare[:, 0], X_rare[:, 1], c='red', label='Rare', alpha=0.5)
# plt.title("Rare vs Normal Samples")
# plt.legend()
# plt.show()

# # Test SUSiter
# susiter = SUSiter(k=7, blobtr=0.75, spreadtr=0.5, replacement_ratio=0.3)
# susiter.fit(X, y)

# # Get 3 different iterations and visualize
# plt.figure(figsize=(15, 5))
# for i in range(3):
#     X_iter, y_iter = susiter.get_iteration_sample()
#     plt.subplot(1, 3, i+1)
#     plt.scatter(X_iter[:, 0], X_iter[:, 1], c=y_iter, cmap='viridis')
#     plt.title(f"Iteration {i+1}")
# plt.tight_layout()
# plt.show()