"""
Turbidity Estimation Module

This module contains machine learning models and algorithms for estimating
water turbidity from satellite imagery data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TurbidityEstimator:
    """Machine learning-based turbidity estimation from satellite data."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the turbidity estimator.
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boost', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = None
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def extract_features(self, image_data: np.ndarray, band_names: List[str]) -> np.ndarray:
        """
        Extract features from satellite imagery for turbidity estimation.
        
        Args:
            image_data: Multi-band satellite image data (bands, height, width)
            band_names: List of band names corresponding to image channels
            
        Returns:
            Feature array for each pixel
        """
        features = []
        
        # Basic spectral bands
        for i, band_name in enumerate(band_names):
            band_data = image_data[i].flatten()
            features.append(band_data)
        
        # Spectral indices
        if 'B2' in band_names and 'B3' in band_names and 'B4' in band_names and 'B8' in band_names:
            b2_idx = band_names.index('B2')  # Blue
            b3_idx = band_names.index('B3')  # Green
            b4_idx = band_names.index('B4')  # Red
            b8_idx = band_names.index('B8')  # NIR
            
            blue = image_data[b2_idx].flatten()
            green = image_data[b3_idx].flatten()
            red = image_data[b4_idx].flatten()
            nir = image_data[b8_idx].flatten()
            
            # NDWI (Normalized Difference Water Index)
            ndwi = (green - nir) / (green + nir + 1e-10)
            features.append(ndwi)
            
            # Turbidity-related indices
            # Red/NIR ratio (turbidity proxy)
            red_nir_ratio = red / (nir + 1e-10)
            features.append(red_nir_ratio)
            
            # Blue/Green ratio
            blue_green_ratio = blue / (green + 1e-10)
            features.append(blue_green_ratio)
            
            # Green/Red ratio
            green_red_ratio = green / (red + 1e-10)
            features.append(green_red_ratio)
            
            # Turbidity Suspended Matter Index
            tsmi = red / (blue + 1e-10)
            features.append(tsmi)
        
        # Stack features
        feature_matrix = np.stack(features, axis=1)
        
        # Remove invalid pixels (NaN or inf)
        valid_mask = np.isfinite(feature_matrix).all(axis=1)
        feature_matrix = feature_matrix[valid_mask]
        
        return feature_matrix
    
    def prepare_training_data(self, satellite_features: np.ndarray, 
                            ground_truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data by matching satellite features with ground truth measurements.
        
        Args:
            satellite_features: Extracted features from satellite imagery
            ground_truth: Ground truth turbidity measurements
            
        Returns:
            Tuple of (features, targets) ready for training
        """
        # Ensure same number of samples
        min_samples = min(len(satellite_features), len(ground_truth))
        features = satellite_features[:min_samples]
        targets = ground_truth[:min_samples]
        
        # Remove samples with missing values
        valid_mask = np.isfinite(features).all(axis=1) & np.isfinite(targets)
        features = features[valid_mask]
        targets = targets[valid_mask]
        
        return features, targets
    
    def train(self, features: np.ndarray, targets: np.ndarray, 
              test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the turbidity estimation model.
        
        Args:
            features: Training features
            targets: Training targets (turbidity values)
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                   cv=5, scoring='neg_mean_squared_error')
        metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())
        metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
        
        logger.info(f"Training completed. Test R²: {metrics['test_r2']:.3f}, "
                   f"Test RMSE: {metrics['test_rmse']:.3f}")
        
        return metrics
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict turbidity from satellite features.
        
        Args:
            features: Satellite-derived features
            
        Returns:
            Predicted turbidity values
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_image(self, image_data: np.ndarray, band_names: List[str]) -> np.ndarray:
        """
        Predict turbidity for entire satellite image.
        
        Args:
            image_data: Multi-band satellite image (bands, height, width)
            band_names: List of band names
            
        Returns:
            Turbidity map (height, width)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        original_shape = image_data.shape[1:]  # (height, width)
        
        # Extract features for all pixels
        features = self.extract_features(image_data, band_names)
        
        # Predict turbidity
        turbidity_values = self.predict(features)
        
        # Reshape back to image format
        turbidity_map = np.full(original_shape, np.nan)
        
        # Create valid pixel mask
        valid_pixels = np.isfinite(image_data).all(axis=0)
        valid_indices = np.where(valid_pixels.flatten())[0]
        
        if len(valid_indices) == len(turbidity_values):
            turbidity_map.flat[valid_indices] = turbidity_values
        
        return turbidity_map
    
    def save_model(self, filepath: str):
        """Save trained model and scaler."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and scaler."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data.get('feature_importance')
        logger.info(f"Model loaded from {filepath}")
    
    def plot_feature_importance(self, feature_names: List[str] = None):
        """Plot feature importance for tree-based models."""
        if self.feature_importance is None:
            logger.warning("Feature importance not available for this model type")
            return
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance))]
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(self.feature_importance)[::-1]
        
        plt.bar(range(len(self.feature_importance)), 
                self.feature_importance[indices])
        plt.xticks(range(len(self.feature_importance)), 
                   [feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importance for Turbidity Estimation')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()


def create_synthetic_training_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic training data for testing purposes.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (features, turbidity_values)
    """
    np.random.seed(42)
    
    # Generate synthetic spectral bands (reflectance values 0-1)
    blue = np.random.uniform(0.02, 0.15, n_samples)
    green = np.random.uniform(0.03, 0.25, n_samples)
    red = np.random.uniform(0.02, 0.35, n_samples)
    nir = np.random.uniform(0.15, 0.45, n_samples)
    
    # Calculate indices
    red_nir_ratio = red / (nir + 1e-10)
    blue_green_ratio = blue / (green + 1e-10)
    green_red_ratio = green / (red + 1e-10)
    tsmi = red / (blue + 1e-10)
    ndwi = (green - nir) / (green + nir + 1e-10)
    
    # Stack features
    features = np.column_stack([
        blue, green, red, nir, ndwi, red_nir_ratio, 
        blue_green_ratio, green_red_ratio, tsmi
    ])
    
    # Generate synthetic turbidity values based on spectral characteristics
    # Higher red/NIR ratio and lower NDWI typically indicate higher turbidity
    turbidity = (
        50 * red_nir_ratio +  # Red/NIR ratio influence
        -30 * ndwi +          # NDWI influence (negative for turbid water)
        20 * tsmi +           # Turbidity index influence
        np.random.normal(0, 5, n_samples)  # Random noise
    )
    
    # Ensure positive turbidity values
    turbidity = np.maximum(turbidity, 0.1)
    
    return features, turbidity


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Creating synthetic training data...")
    features, turbidity = create_synthetic_training_data(1000)
    
    # Initialize and train model
    estimator = TurbidityEstimator(model_type='random_forest')
    
    print("Training turbidity estimation model...")
    metrics = estimator.train(features, turbidity)
    
    print("Training Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Test prediction
    test_features, test_turbidity = create_synthetic_training_data(100)
    predictions = estimator.predict(test_features)
    
    test_rmse = np.sqrt(mean_squared_error(test_turbidity, predictions))
    test_r2 = r2_score(test_turbidity, predictions)
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {test_rmse:.3f}")
    print(f"  R²: {test_r2:.3f}")
    
    # Feature names for visualization
    feature_names = ['Blue', 'Green', 'Red', 'NIR', 'NDWI', 
                    'Red/NIR', 'Blue/Green', 'Green/Red', 'TSMI']
    estimator.plot_feature_importance(feature_names)
