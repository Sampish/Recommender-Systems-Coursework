"""
TimeSVD++ Implementation
Based on Koren (2009) - "Collaborative Filtering with Temporal Dynamics"

This implementation adds temporal dynamics to SVD++:
- Time-varying user biases
- Time-varying item biases  
- User factors that evolve over time
"""

import numpy as np
from collections import defaultdict

class TimeSVDpp:
    def __init__(self, n_factors=20, n_epochs=20, lr=0.005, reg=0.02, 
                 n_time_bins=30, random_state=42):
        """
        TimeSVD++ with temporal dynamics.
        
        Parameters:
        -----------
        n_factors : int
            Number of latent factors
        n_epochs : int
            Number of training epochs
        lr : float
            Learning rate
        reg : float
            Regularization parameter
        n_time_bins : int
            Number of time bins for temporal modeling
        random_state : int
            Random seed
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.n_time_bins = n_time_bins
        self.random_state = random_state
        
        # Will be initialized in fit()
        self.global_mean = 0
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        
        # Temporal components
        self.user_bias_bins = None  # Time-varying user bias
        self.item_bias_bins = None  # Time-varying item bias
        
        # Implicit feedback (SVD++ component)
        self.implicit_feedback = None
        
    def _init_params(self, n_users, n_items):
        """Initialize model parameters."""
        np.random.seed(self.random_state)
        
        # Standard SVD components
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        # Temporal components
        self.user_bias_bins = np.zeros((n_users, self.n_time_bins))
        self.item_bias_bins = np.zeros((n_items, self.n_time_bins))
        
        # Implicit feedback factors (for SVD++)
        self.implicit_feedback = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
    def fit(self, user_ids, item_ids, ratings, timestamps=None):
        """
        Train the model.
        
        Parameters:
        -----------
        user_ids : array-like
            User IDs for each rating
        item_ids : array-like
            Item IDs for each rating
        ratings : array-like
            Rating values
        timestamps : array-like, optional
            Timestamps for temporal modeling (if None, uses index as proxy)
        """
        # Convert to numpy arrays
        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        ratings = np.array(ratings, dtype=np.float64)
        
        # Handle timestamps
        if timestamps is None:
            timestamps = np.arange(len(ratings))
        else:
            timestamps = np.array(timestamps)
            
        # Normalize timestamps to bins
        t_min, t_max = timestamps.min(), timestamps.max()
        if t_max > t_min:
            time_bins = ((timestamps - t_min) / (t_max - t_min) * (self.n_time_bins - 1)).astype(int)
        else:
            time_bins = np.zeros(len(timestamps), dtype=int)
        
        # Build user->items mapping for implicit feedback
        user_items = defaultdict(list)
        for u, i in zip(user_ids, item_ids):
            user_items[u].append(i)
        
        # Initialize parameters
        n_users = user_ids.max() + 1
        n_items = item_ids.max() + 1
        self._init_params(n_users, n_items)
        
        # Calculate global mean
        self.global_mean = ratings.mean()
        
        # Training loop
        print(f"Training TimeSVD++ for {self.n_epochs} epochs...")
        for epoch in range(self.n_epochs):
            # Shuffle training data
            indices = np.random.permutation(len(ratings))
            
            epoch_loss = 0.0
            for idx in indices:
                u = user_ids[idx]
                i = item_ids[idx]
                r = ratings[idx]
                t_bin = time_bins[idx]
                
                # Get implicit feedback for user
                implicit_items = user_items[u]
                n_impl = len(implicit_items)
                
                if n_impl > 0:
                    # Sum of implicit item factors
                    impl_sum = self.implicit_feedback[implicit_items].sum(axis=0) / np.sqrt(n_impl)
                else:
                    impl_sum = np.zeros(self.n_factors)
                
                # Prediction with temporal biases
                user_factor = self.user_factors[u] + impl_sum
                pred = (self.global_mean + 
                       self.user_biases[u] + self.user_bias_bins[u, t_bin] +
                       self.item_biases[i] + self.item_bias_bins[i, t_bin] +
                       np.dot(user_factor, self.item_factors[i]))
                
                # Error
                err = r - pred
                epoch_loss += err ** 2
                
                # Gradient updates
                # Biases
                self.user_biases[u] += self.lr * (err - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (err - self.reg * self.item_biases[i])
                self.user_bias_bins[u, t_bin] += self.lr * (err - self.reg * self.user_bias_bins[u, t_bin])
                self.item_bias_bins[i, t_bin] += self.lr * (err - self.reg * self.item_bias_bins[i, t_bin])
                
                # Factors
                user_f_old = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (err * user_factor - self.reg * self.item_factors[i])
                
                # Implicit feedback
                if n_impl > 0:
                    for impl_i in implicit_items:
                        self.implicit_feedback[impl_i] += self.lr * (
                            err * self.item_factors[i] / np.sqrt(n_impl) - 
                            self.reg * self.implicit_feedback[impl_i]
                        )
            
            rmse = np.sqrt(epoch_loss / len(ratings))
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: RMSE = {rmse:.4f}")
                
        print("✅ TimeSVD++ Training Complete")
        
    def predict(self, user_id, item_id, user_history=None, timestamp=None):
        """
        Predict rating for a user-item pair.
        
        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID
        user_history : list, optional
            List of items the user has interacted with (for implicit feedback)
        timestamp : float, optional
            Timestamp for temporal prediction
        """
        # Handle unknown users/items
        if user_id >= len(self.user_biases) or item_id >= len(self.item_biases):
            return self.global_mean
        
        # Determine time bin
        if timestamp is not None and hasattr(self, 't_min'):
            t_bin = int((timestamp - self.t_min) / (self.t_max - self.t_min) * (self.n_time_bins - 1))
            t_bin = np.clip(t_bin, 0, self.n_time_bins - 1)
        else:
            t_bin = self.n_time_bins // 2  # Use middle bin as default
        
        # Get implicit feedback
        if user_history and len(user_history) > 0:
            impl_sum = self.implicit_feedback[user_history].sum(axis=0) / np.sqrt(len(user_history))
        else:
            impl_sum = np.zeros(self.n_factors)
        
        # Prediction
        user_factor = self.user_factors[user_id] + impl_sum
        pred = (self.global_mean + 
               self.user_biases[user_id] + self.user_bias_bins[user_id, t_bin] +
               self.item_biases[item_id] + self.item_bias_bins[item_id, t_bin] +
               np.dot(user_factor, self.item_factors[item_id]))
        
        return pred
