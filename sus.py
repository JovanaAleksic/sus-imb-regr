import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from phi import *

class SUS:
    """
    Selective Under-Sampling (SUS) for imbalanced regression using phi control points.
    
    Parameters
    ----------
    k : int, default=7
        Number of neighbors in kNN model
    blobtr : float, default=0.75
        Percentile threshold for close neighbors distance (between 0 and 1)
    spreadtr : float, default=0.5
        Threshold for cluster target values dispersion
    relevance_threshold : float, default=0.8
        Threshold for phi values to separate rare from normal samples
    random_state : int or None, default=None
        Random state for reproducibility
    """
    
    def __init__(self, k=7, blobtr=0.75, spreadtr=0.5, relevance_threshold=0.8, random_state=None):
        self.k = k
        self.blobtr = blobtr
        self.spreadtr = spreadtr
        self.relevance_threshold = relevance_threshold
        self.random_state = random_state
        np.random.seed(random_state)
        
    def _compute_relevance(self, y):
        """Compute relevance scores using phi control points."""
        # Use phi control points for relevance computation
        phi_params = phi_ctrl_pts(y=y)
        y_phi = phi(y=y, ctrl_pts=phi_params)
        return np.asarray(y_phi)
        
    def _split_rare_normal(self, X, y):
        """Split data into rare and normal samples based on phi values."""
        relevance = self._compute_relevance(y)
        rare_mask = relevance >= self.relevance_threshold
        
        return (X[rare_mask], y[rare_mask]), (X[~rare_mask], y[~rare_mask])
        
    def _compute_blob_threshold(self, distances):
        """Compute threshold for close neighbors."""
        avg_distances = np.mean(distances, axis=1)
        return np.percentile(avg_distances, self.blobtr * 100)
        
    def _process_cluster(self, indices, distances, X, y):
        """Process a cluster of samples to select representatives."""
        if len(indices) <= 1:
            return indices, []
            
        # Calculate target value spread
        y_cluster = y[indices]
        y_mean = np.mean(y_cluster)
        spread = np.std(y_cluster) / (y_mean + 1e-10)
        
        if spread <= self.spreadtr:
            # If spread is small enough, select sample closest to mean
            mean_dist = np.abs(y_cluster - y_mean)
            selected_idx = indices[np.argmin(mean_dist)]
            pool_indices = [idx for idx in indices if idx != selected_idx]
            return [selected_idx], pool_indices
            
        # If spread is large, keep sample with most distant target and recurse
        max_dist_idx = np.argmax(np.abs(y_cluster - y_mean))
        remaining_indices = np.delete(indices, max_dist_idx)
        
        selected = [indices[max_dist_idx]]
        recursed_selected, recursed_pool = self._process_cluster(remaining_indices, distances, X, y)
        
        return selected + recursed_selected, recursed_pool
        
    def fit_resample(self, X, y):
        """
        Fit SUS and return resampled X, y.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        X_resampled : ndarray of shape (n_samples_new, n_features)
            Resampled features
        y_resampled : ndarray of shape (n_samples_new,)
            Resampled target values
        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        
        # Split into rare and normal samples using phi values
        (X_rare, y_rare), (X_normal, y_normal) = self._split_rare_normal(X, y)
        
        if len(X_normal) == 0:
            return X, y
            
        # Store relevance values for reference
        self.phi_values_ = self._compute_relevance(y)
            
        # Fit kNN on normal samples
        self.knn_ = NearestNeighbors(n_neighbors=self.k)
        self.knn_.fit(X_normal)
        
        # Get distances and indices
        distances, indices = self.knn_.kneighbors()
        
        # Compute threshold for close neighbors
        self.blob_threshold_ = self._compute_blob_threshold(distances)
        
        # Initialize selected indices
        selected_indices = []
        pool_indices = []  # For SUSiter
        visited = np.zeros(len(X_normal), dtype=bool)
        
        # Process each unvisited sample
        for i in range(len(X_normal)):
            if visited[i]:
                continue
                
            # Find close neighbors
            close_mask = distances[i] < self.blob_threshold_
            cluster_indices = indices[i][close_mask]
            
            # If no close neighbors, select the sample
            if not np.any(close_mask[1:]):
                selected_indices.append(i)
                visited[cluster_indices] = True
                continue
                
            # Process cluster
            selected, pool = self._process_cluster(cluster_indices, distances, X_normal, y_normal)
            selected_indices.extend(selected)
            pool_indices.extend(pool)
            visited[cluster_indices] = True
            
        # Store indices for SUSiter
        self.selected_normal_indices_ = selected_indices
        self.pool_normal_indices_ = pool_indices
        self.X_normal_ = X_normal
        self.y_normal_ = y_normal
        self.X_rare_ = X_rare
        self.y_rare_ = y_rare
        
        # Store phi values with corresponding y values
        self.phi_zip = list(zip(self.phi_values_, y))
        
        # Combine selected normal samples with rare samples
        X_resampled = np.vstack([X_rare, X_normal[selected_indices]])
        y_resampled = np.hstack([y_rare, y_normal[selected_indices]])
        
        return X_resampled, y_resampled


class SUSiter(SUS):
    """
    Iterative Selective Under-Sampling (SUSiter) for imbalanced regression.
    
    In addition to SUS parameters, also supports:
    
    Parameters
    ----------
    replacement_ratio : float, default=0.3
        Proportion of samples to replace in each iteration
    """
    
    def __init__(self, k=7, blobtr=0.75, spreadtr=0.5, relevance_threshold=0.8, 
                 replacement_ratio=0.3, random_state=None):
        super().__init__(k=k, blobtr=blobtr, spreadtr=spreadtr, 
                        relevance_threshold=relevance_threshold, 
                        random_state=random_state)
        self.replacement_ratio = replacement_ratio
        
    def fit(self, X, y):
        """
        Fit SUSiter (performs initial SUS selection).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Returns self
        """
        # Perform initial SUS selection
        self.fit_resample(X, y)
        return self
        
    def get_iteration_sample(self, X=None, y=None):
        """
        Get a sample for the current iteration, replacing some cluster representatives.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            If None, uses X from fit
        y : array-like of shape (n_samples,), default=None
            If None, uses y from fit
            
        Returns
        -------
        X_iter : ndarray of shape (n_samples_new, n_features)
            Features for this iteration
        y_iter : ndarray of shape (n_samples_new,)
            Target values for this iteration
        """
        if not hasattr(self, 'selected_normal_indices_'):
            if X is None or y is None:
                raise ValueError("Must call fit first or provide X and y")
            self.fit(X, y)
            
        # Determine number of samples to replace
        n_replace = int(len(self.selected_normal_indices_) * self.replacement_ratio)
        
        # Randomly select indices to replace
        replace_mask = np.random.choice(len(self.selected_normal_indices_), 
                                      size=n_replace, replace=False)
        
        # Create new selection by replacing some samples with their close neighbors
        current_selection = self.selected_normal_indices_.copy()
        for idx in replace_mask:
            original_idx = current_selection[idx]
            # Find available neighbors from the pool
            _, cluster_indices = self.knn_.kneighbors(
                self.X_normal_[original_idx].reshape(1, -1)
            )
            valid_replacements = [i for i in cluster_indices[0] 
                                if i in self.pool_normal_indices_]
            
            if valid_replacements:
                # Replace with a random neighbor from the pool
                current_selection[idx] = np.random.choice(valid_replacements)
        
        # Combine with rare samples
        X_iter = np.vstack([self.X_rare_, self.X_normal_[current_selection]])
        y_iter = np.hstack([self.y_rare_, self.y_normal_[current_selection]])
        
        return X_iter, y_iter