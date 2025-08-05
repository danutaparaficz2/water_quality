"""
Visualization Module for Turbidity Analysis

This module provides functions for visualizing satellite imagery,
turbidity maps, and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import pandas as pd
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


def plot_satellite_image(image_data: np.ndarray, bands: List[int] = [3, 2, 1], 
                        title: str = "Satellite Image", figsize: Tuple[int, int] = (10, 8)):
    """
    Plot RGB composite of satellite image.
    
    Args:
        image_data: Multi-band satellite image (bands, height, width)
        bands: List of band indices for RGB display
        title: Plot title
        figsize: Figure size
    """
    # Extract RGB bands
    rgb_bands = image_data[bands]
    
    # Normalize to 0-1 range
    rgb_image = np.moveaxis(rgb_bands, 0, -1)
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # Apply gamma correction for better visualization
    rgb_image = np.power(rgb_image, 0.8)
    
    plt.figure(figsize=figsize)
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_turbidity_map(turbidity_map: np.ndarray, title: str = "Turbidity Map",
                      vmin: Optional[float] = None, vmax: Optional[float] = None,
                      figsize: Tuple[int, int] = (12, 8)):
    """
    Plot turbidity map with appropriate colormap.
    
    Args:
        turbidity_map: 2D array of turbidity values
        title: Plot title
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        figsize: Figure size
    """
    # Calculate statistics
    valid_data = turbidity_map[np.isfinite(turbidity_map)]
    if len(valid_data) == 0:
        logger.warning("No valid turbidity data to plot")
        return
    
    if vmin is None:
        vmin = np.percentile(valid_data, 2)
    if vmax is None:
        vmax = np.percentile(valid_data, 98)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot turbidity map
    im = ax.imshow(turbidity_map, cmap='YlOrRd', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Turbidity (NTU)', rotation=270, labelpad=20)
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add statistics text
    stats_text = f"Min: {valid_data.min():.2f} NTU\n" \
                f"Max: {valid_data.max():.2f} NTU\n" \
                f"Mean: {valid_data.mean():.2f} NTU\n" \
                f"Std: {valid_data.std():.2f} NTU"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_turbidity_histogram(turbidity_map: np.ndarray, bins: int = 50,
                           title: str = "Turbidity Distribution"):
    """
    Plot histogram of turbidity values.
    
    Args:
        turbidity_map: 2D array of turbidity values
        bins: Number of histogram bins
        title: Plot title
    """
    valid_data = turbidity_map[np.isfinite(turbidity_map)]
    
    if len(valid_data) == 0:
        logger.warning("No valid turbidity data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(valid_data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Turbidity (NTU)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = valid_data.mean()
    median_val = np.median(valid_data)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='orange', linestyle='--', label=f'Median: {median_val:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_water_indices(image_data: np.ndarray, band_names: List[str], 
                      figsize: Tuple[int, int] = (15, 10)):
    """
    Plot various water quality indices.
    
    Args:
        image_data: Multi-band satellite image (bands, height, width)
        band_names: List of band names
        figsize: Figure size
    """
    # Calculate indices
    indices = {}
    
    if all(band in band_names for band in ['B3', 'B8']):
        b3_idx = band_names.index('B3')
        b8_idx = band_names.index('B8')
        green = image_data[b3_idx]
        nir = image_data[b8_idx]
        indices['NDWI'] = (green - nir) / (green + nir + 1e-10)
    
    if all(band in band_names for band in ['B2', 'B3', 'B4', 'B8']):
        b2_idx = band_names.index('B2')
        b3_idx = band_names.index('B3')
        b4_idx = band_names.index('B4')
        b8_idx = band_names.index('B8')
        
        blue = image_data[b2_idx]
        green = image_data[b3_idx]
        red = image_data[b4_idx]
        nir = image_data[b8_idx]
        
        indices['Red/NIR Ratio'] = red / (nir + 1e-10)
        indices['Blue/Green Ratio'] = blue / (green + 1e-10)
        indices['TSMI'] = red / (blue + 1e-10)
    
    # Plot indices
    n_indices = len(indices)
    if n_indices == 0:
        logger.warning("Cannot calculate water indices with available bands")
        return
    
    cols = min(3, n_indices)
    rows = (n_indices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (name, index_data) in enumerate(indices.items()):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        # Calculate percentiles for robust visualization
        valid_data = index_data[np.isfinite(index_data)]
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [2, 98])
        else:
            vmin, vmax = 0, 1
        
        im = ax.imshow(index_data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    # Hide unused subplots
    for i in range(n_indices, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                          title: str = "Model Performance"):
    """
    Plot model performance metrics.
    
    Args:
        y_true: True turbidity values
        y_pred: Predicted turbidity values
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax1.set_xlabel('True Turbidity (NTU)')
    ax1.set_ylabel('Predicted Turbidity (NTU)')
    ax1.set_title('Predicted vs True Turbidity')
    ax1.grid(True, alpha=0.3)
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white'))
    
    # Residuals plot
    ax2 = axes[1]
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Turbidity (NTU)')
    ax2.set_ylabel('Residuals (NTU)')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_time_series_turbidity(dates: List[str], turbidity_values: List[float],
                              location_name: str = "Roehne River"):
    """
    Plot time series of turbidity measurements.
    
    Args:
        dates: List of date strings
        turbidity_values: List of turbidity values
        location_name: Name of the location
    """
    plt.figure(figsize=(12, 6))
    
    # Convert dates to datetime if they're strings
    if isinstance(dates[0], str):
        dates = pd.to_datetime(dates)
    
    plt.plot(dates, turbidity_values, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Date')
    plt.ylabel('Turbidity (NTU)')
    plt.title(f'Turbidity Time Series - {location_name}')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add trend line
    if len(turbidity_values) > 2:
        z = np.polyfit(range(len(turbidity_values)), turbidity_values, 1)
        p = np.poly1d(z)
        plt.plot(dates, p(range(len(turbidity_values))), "r--", alpha=0.8, 
                label=f'Trend (slope: {z[0]:.3f})')
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def create_turbidity_classification_map(turbidity_map: np.ndarray, 
                                       thresholds: Dict[str, float] = None):
    """
    Create a classified turbidity map based on water quality categories.
    
    Args:
        turbidity_map: 2D array of turbidity values
        thresholds: Dictionary of classification thresholds
    """
    if thresholds is None:
        thresholds = {
            'Clear': 5,
            'Slightly Turbid': 25,
            'Turbid': 100,
            'Very Turbid': float('inf')
        }
    
    # Create classification array
    classified = np.full_like(turbidity_map, -1, dtype=int)
    
    class_labels = list(thresholds.keys())
    class_values = list(thresholds.values())
    
    for i, (label, threshold) in enumerate(thresholds.items()):
        if i == 0:
            mask = turbidity_map <= threshold
        else:
            prev_threshold = class_values[i-1]
            mask = (turbidity_map > prev_threshold) & (turbidity_map <= threshold)
        
        classified[mask] = i
    
    # Create custom colormap
    colors_list = ['blue', 'cyan', 'yellow', 'orange', 'red']
    n_classes = len(class_labels)
    cmap = colors.ListedColormap(colors_list[:n_classes])
    
    # Plot
    plt.figure(figsize=(12, 8))
    im = plt.imshow(classified, cmap=cmap, vmin=0, vmax=n_classes-1)
    
    # Create colorbar with proper labels
    cbar = plt.colorbar(im, ticks=range(n_classes))
    cbar.set_ticklabels(class_labels)
    cbar.set_label('Water Quality Category')
    
    plt.title('Turbidity Classification Map')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    valid_pixels = classified >= 0
    total_valid = valid_pixels.sum()
    
    print("Water Quality Classification Statistics:")
    for i, label in enumerate(class_labels):
        count = (classified == i).sum()
        percentage = (count / total_valid) * 100 if total_valid > 0 else 0
        print(f"  {label}: {count} pixels ({percentage:.1f}%)")


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Creating example visualizations...")
    
    # Generate synthetic turbidity map
    np.random.seed(42)
    height, width = 100, 150
    
    # Create synthetic turbidity data with spatial patterns
    x, y = np.meshgrid(np.linspace(0, 10, width), np.linspace(0, 10, height))
    turbidity_synthetic = (
        20 + 10 * np.sin(x) + 5 * np.cos(y) + 
        np.random.normal(0, 3, (height, width))
    )
    turbidity_synthetic = np.maximum(turbidity_synthetic, 0)
    
    # Plot turbidity map
    plot_turbidity_map(turbidity_synthetic, title="Synthetic Turbidity Map")
    
    # Plot histogram
    plot_turbidity_histogram(turbidity_synthetic)
    
    # Create classification map
    create_turbidity_classification_map(turbidity_synthetic)
    
    print("Visualization examples completed!")
