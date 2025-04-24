
"""
Wrappers for different analysis backends, primarily Scikit-learn.
Provides functions for common machine learning tasks like scaling, PCA, PLS,
and classification, with error handling and checks for library availability.
"""
import logging
import numpy as np
import pandas as pd
import traceback
from typing import Tuple, Optional, Dict, Any

# --- Scikit-learn Imports ---
SKLEARN_AVAILABLE = False
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    # from sklearn.cluster import KMeans # Example if adding clustering later
    SKLEARN_AVAILABLE = True
    logging.info("Scikit-learn library found. ML backends enabled.")
except ImportError:
    logging.warning("Scikit-learn library not found. Machine learning analysis features will be disabled. "
                    "Install with 'pip install scikit-learn'.")
    # Define dummy classes/functions to prevent NameErrors if library is missing
    class StandardScaler: pass
    class PCA: pass
    class PLSRegression: pass
    class RandomForestClassifier: pass
    class GradientBoostingClassifier: pass
    class MLPClassifier: pass
    # class KMeans: pass


# --- Backend Check Function ---
def check_sklearn_availability() -> bool:
    """Checks if scikit-learn is installed and available."""
    if not SKLEARN_AVAILABLE:
         # Avoid logging error every time, just return status. Let caller log error if needed.
         pass
    return SKLEARN_AVAILABLE

# --- Preprocessing ---
def scale_data(X: np.ndarray) -> Optional[np.ndarray]:
    """
    Applies standard scaling (zero mean, unit variance) to the data matrix.

    Args:
        X (np.ndarray): Data matrix (samples x features).

    Returns:
        Optional[np.ndarray]: Scaled data matrix, or None if scaling fails or
                              scikit-learn is unavailable.
    """
    if not check_sklearn_availability(): return None
    if X is None or not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] == 0:
         logging.error("Invalid input data provided for scaling (must be 2D array with features).")
         return None
    # Check for NaNs/Infs before scaling
    if not np.all(np.isfinite(X)):
        logging.error("Input data contains NaN or Inf values. Cannot scale.")
        # Option: Impute NaNs before scaling? For now, fail.
        # X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) # Example imputation
        return None
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logging.debug("Data scaling applied successfully.")
        return X_scaled
    except Exception as e:
        logging.error(f"Error during data scaling: {e}", exc_info=True)
        return None

# --- Dimensionality Reduction ---
def run_pca(X: np.ndarray, n_components: int = 3) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Performs Principal Component Analysis (PCA).

    Args:
        X (np.ndarray): Data matrix (samples x features), ideally scaled.
        n_components (int): Number of principal components to compute.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]:
            - scores (np.ndarray): PCA scores (samples x n_components).
            - explained_variance_ratio (np.ndarray): Variance explained by each component.
            Returns None if PCA fails or scikit-learn is unavailable.
    """
    if not check_sklearn_availability(): return None
    if X is None or not isinstance(X, np.ndarray) or X.ndim != 2:
        logging.error("Invalid input for PCA (must be 2D array).")
        return None
    if not np.all(np.isfinite(X)):
        logging.error("PCA input data contains NaN or Inf values.")
        return None

    n_samples, n_features = X.shape
    # Adjust n_components if it exceeds possible dimensions
    actual_n_components = min(n_components, n_samples, n_features)
    if actual_n_components < 1:
        logging.error(f"Cannot perform PCA with n_components={actual_n_components} (samples={n_samples}, features={n_features}).")
        return None
    # Ensure n_components is int if passed as float from UI maybe
    actual_n_components = int(max(1, actual_n_components))

    try:
        pca = PCA(n_components=actual_n_components)
        scores = pca.fit_transform(X)
        variance_explained = pca.explained_variance_ratio_
        logging.info(f"PCA completed with {actual_n_components} components. Explained variance: {variance_explained}")
        return scores, variance_explained
    except Exception as e:
        logging.error(f"Error during PCA execution: {e}", exc_info=True)
        return None

# --- Regression ---
def run_pls_regression(X: np.ndarray, y: np.ndarray, n_components: int = 5) -> Optional[Tuple[np.ndarray, float]]:
    """
    Performs Partial Least Squares (PLS) Regression.

    Args:
        X (np.ndarray): Predictor data matrix (samples x features), ideally scaled.
        y (np.ndarray): Target variable array (samples x 1 or samples).
        n_components (int): Number of PLS components to compute.

    Returns:
        Optional[Tuple[np.ndarray, float]]:
            - y_pred (np.ndarray): Predicted target values.
            - r2_score (float): R-squared score of the model fit.
            Returns None if PLS fails or scikit-learn is unavailable.
    """
    if not check_sklearn_availability(): return None
    if X is None or y is None or not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray) or X.ndim != 2 or y.ndim < 1 or X.shape[0] != len(y):
         logging.error("Invalid input data provided for PLS Regression.")
         return None
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
         logging.error("PLS input data contains NaN or Inf values.")
         return None

    n_samples, n_features = X.shape
    y_target = y.reshape(-1, 1) if y.ndim == 1 else y # Ensure y is 2D for scikit-learn

    # PLS requires n_components < n_samples (adjust if n_samples is very small)
    actual_n_components = min(n_components, n_samples - 1 if n_samples > 1 else 1, n_features)
    if actual_n_components < 1:
        logging.error(f"Cannot perform PLS with n_components={actual_n_components} (samples={n_samples}, features={n_features}).")
        return None
    actual_n_components = int(max(1, actual_n_components))

    try:
        pls = PLSRegression(n_components=actual_n_components)
        # Ensure y_target is float64 for regression
        pls.fit(X, y_target.astype(np.float64))
        y_pred = pls.predict(X)
        r2 = pls.score(X, y_target.astype(np.float64))
        logging.info(f"PLS Regression completed ({actual_n_components} components). R² = {r2:.4f}")
        # Return flattened predictions consistent with original y shape
        return y_pred.flatten() if y.ndim == 1 else y_pred, r2
    except Exception as e:
        logging.error(f"Error during PLS Regression execution: {e}", exc_info=True)
        return None


# --- Classification ---
def run_classification(X: np.ndarray, y_labels: np.ndarray, method: str = 'RandomForest', **kwargs) -> Optional[Tuple[np.ndarray, float]]:
    """
    Performs classification using a specified scikit-learn method.

    Args:
        X (np.ndarray): Data matrix (samples x features), ideally scaled.
        y_labels (np.ndarray): True class labels for each sample (1D array).
        method (str): Classification method ('RandomForest', 'GBT', 'MLP').
        **kwargs: Additional keyword arguments passed to the classifier.

    Returns:
        Optional[Tuple[np.ndarray, float]]:
            - y_pred (np.ndarray): Predicted class labels.
            - accuracy (float): Accuracy score on the training data (for simplicity).
            Returns None on failure or if scikit-learn is unavailable.
    """
    if not check_sklearn_availability(): return None
    if X is None or y_labels is None or not isinstance(X, np.ndarray) or not isinstance(y_labels, np.ndarray) or X.ndim != 2 or y_labels.ndim != 1 or X.shape[0] != len(y_labels):
         logging.error("Invalid input data provided for classification.")
         return None
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y_labels)):
         logging.error("Classification input data contains NaN or Inf values.")
         return None
    if len(np.unique(y_labels)) < 2:
        logging.error("Classification requires at least two distinct classes in y_labels.")
        return None

    logging.info(f"Running classification using {method}...")
    classifier = None
    # Filter kwargs relevant to the chosen classifier to avoid unexpected errors
    # Note: This requires knowing expected args or careful use of inspect module.
    # Simplified: Pass all kwargs for now, relying on scikit-learn to handle extras if possible.
    constructor_kwargs = kwargs.copy()
    constructor_kwargs.setdefault('random_state', 42) # Ensure reproducibility

    try:
        if method == 'RandomForest':
            constructor_kwargs.setdefault('n_estimators', 100)
            classifier = RandomForestClassifier(**constructor_kwargs)
        elif method == 'GBT':
            constructor_kwargs.setdefault('n_estimators', 100)
            constructor_kwargs.setdefault('learning_rate', 0.1)
            classifier = GradientBoostingClassifier(**constructor_kwargs)
        elif method == 'MLP':
             constructor_kwargs.setdefault('hidden_layer_sizes', (50, 25))
             constructor_kwargs.setdefault('max_iter', 500)
             classifier = MLPClassifier(**constructor_kwargs)
        else:
            logging.error(f"Unsupported classification method: {method}")
            return None

        classifier.fit(X, y_labels)
        y_pred = classifier.predict(X)
        # Note: Accuracy on training data can be misleadingly high. Use cross-validation for real evaluation.
        accuracy = np.mean(y_pred == y_labels)
        logging.info(f"{method} classification complete. Training Accuracy = {accuracy:.4f}")
        return y_pred, accuracy

    except Exception as e:
        logging.error(f"Error during {method} classification: {e}", exc_info=True)
        return None

# --- Clustering (Placeholder) ---
# def run_clustering(X: np.ndarray, method: str = 'KMeans', n_clusters: int = 3, **kwargs) -> Optional[np.ndarray]:
#     if not check_sklearn_availability(): return None
#     logging.warning(f"Clustering method '{method}' NI.")
#     return None

# --- C++/ROOT Backend Placeholders ---
CPP_MODULE_AVAILABLE = False # Keep explicitly False as not implemented
cpp_backend = None
# def run_pca_cpp(X: np.ndarray, n_components: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
#     if not CPP_MODULE_AVAILABLE: logging.error("C++ backend NI."); return None
#     # try: return cpp_backend.perform_pca(X, n_components)
#     # except Exception as e: logging.error(f"C++ PCA error: {e}"); return None
#     return None
