"""
Main Analysis Script for Roehne River Turbidity Estimation

This script orchestrates the complete workflow for estimating turbidity
from satellite images of the Roehne river.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.satellite_data import SatelliteDataManager, preprocess_satellite_image
from src.turbidity_estimation import TurbidityEstimator, create_synthetic_training_data
from src.visualization import (plot_turbidity_map, plot_turbidity_histogram, 
                              plot_water_indices, create_turbidity_classification_map,
                              plot_time_series_turbidity)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('turbidity_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RoehneTurbidityAnalyzer:
    """Complete turbidity analysis workflow for Roehne river."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "output"):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory for storing satellite data
            output_dir: Directory for saving results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.satellite_manager = SatelliteDataManager()
        self.turbidity_estimator = TurbidityEstimator()
        
        # Results storage
        self.analysis_results = {}
    
    def setup_environment(self):
        """Set up the analysis environment and check dependencies."""
        logger.info("Setting up analysis environment...")
        
        # Check if required directories exist
        required_dirs = ['data/satellite', 'data/ground_truth', 'output/models', 
                        'output/maps', 'output/reports']
        
        for dir_path in required_dirs:
            full_path = Path(dir_path)
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
        
        logger.info("Environment setup completed")
    
    def load_ground_truth_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load ground truth turbidity measurements.
        
        Args:
            filepath: Path to ground truth data file
            
        Returns:
            DataFrame with ground truth measurements
        """
        if filepath and os.path.exists(filepath):
            logger.info(f"Loading ground truth data from {filepath}")
            return pd.read_csv(filepath)
        else:
            logger.warning("No ground truth data file found. Using synthetic data for demonstration.")
            # Create synthetic ground truth data
            dates = pd.date_range('2023-01-01', '2023-12-31', freq='7D')
            turbidity_values = np.random.uniform(5, 50, len(dates)) + \
                              10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 52)  # Seasonal pattern
            
            return pd.DataFrame({
                'date': dates,
                'turbidity_ntu': turbidity_values,
                'latitude': 51.75,  # Approximate Roehne river coordinates
                'longitude': 8.25
            })
    
    def acquire_satellite_data(self, start_date: str, end_date: str):
        """
        Acquire satellite data for the specified time period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        logger.info(f"Acquiring satellite data from {start_date} to {end_date}")
        
        try:
            # Search for Sentinel-2 data
            if hasattr(self.satellite_manager, 'sentinel_api') and self.satellite_manager.sentinel_api:
                search_results = self.satellite_manager.search_sentinel2_data(
                    start_date, end_date, max_cloud_cover=20
                )
                logger.info(f"Found {len(search_results)} Sentinel-2 products")
                self.analysis_results['search_results'] = search_results
            else:
                logger.warning("Sentinel API not configured. Using synthetic data.")
                self._create_synthetic_satellite_data()
                
        except Exception as e:
            logger.error(f"Error acquiring satellite data: {e}")
            logger.info("Falling back to synthetic satellite data")
            self._create_synthetic_satellite_data()
    
    def _create_synthetic_satellite_data(self):
        """Create synthetic satellite data for testing."""
        logger.info("Creating synthetic satellite data...")
        
        # Generate synthetic multi-spectral image
        height, width = 200, 300
        n_bands = 4  # Blue, Green, Red, NIR
        
        # Create realistic spectral patterns for water bodies
        np.random.seed(42)
        
        # Base reflectance values typical for water
        bands = []
        for i in range(n_bands):
            if i == 0:  # Blue - higher for clear water
                band = np.random.uniform(0.02, 0.08, (height, width))
            elif i == 1:  # Green - moderate for water
                band = np.random.uniform(0.01, 0.06, (height, width))
            elif i == 2:  # Red - varies with turbidity
                band = np.random.uniform(0.005, 0.05, (height, width))
            else:  # NIR - low for water
                band = np.random.uniform(0.001, 0.02, (height, width))
            
            # Add spatial patterns
            x, y = np.meshgrid(np.linspace(0, 5, width), np.linspace(0, 5, height))
            pattern = 0.01 * np.sin(x) * np.cos(y)
            band += pattern
            
            bands.append(band)
        
        self.synthetic_image = np.stack(bands, axis=0)
        self.band_names = ['B2', 'B3', 'B4', 'B8']  # Sentinel-2 band names
        
        logger.info(f"Created synthetic image with shape: {self.synthetic_image.shape}")
    
    def train_turbidity_model(self, ground_truth_data: pd.DataFrame):
        """
        Train the turbidity estimation model.
        
        Args:
            ground_truth_data: DataFrame with ground truth measurements
        """
        logger.info("Training turbidity estimation model...")
        
        # For demonstration, use synthetic training data
        # In practice, you would extract features from satellite images at ground truth locations
        features, targets = create_synthetic_training_data(1000)
        
        # Train model
        metrics = self.turbidity_estimator.train(features, targets, test_size=0.2)
        
        # Log training results
        logger.info("Training completed with metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.3f}")
        
        # Save model
        model_path = self.output_dir / "models" / "turbidity_model.joblib"
        self.turbidity_estimator.save_model(str(model_path))
        
        self.analysis_results['training_metrics'] = metrics
    
    def estimate_turbidity(self, image_data: np.ndarray = None, band_names: list = None):
        """
        Estimate turbidity from satellite image.
        
        Args:
            image_data: Satellite image data
            band_names: List of band names
        """
        logger.info("Estimating turbidity from satellite imagery...")
        
        # Use synthetic data if no image provided
        if image_data is None:
            if hasattr(self, 'synthetic_image'):
                image_data = self.synthetic_image
                band_names = self.band_names
            else:
                raise ValueError("No satellite image data available")
        
        # Predict turbidity
        turbidity_map = self.turbidity_estimator.predict_image(image_data, band_names)
        
        # Save results
        self.analysis_results['turbidity_map'] = turbidity_map
        
        # Calculate statistics
        valid_turbidity = turbidity_map[np.isfinite(turbidity_map)]
        stats = {
            'mean': float(np.mean(valid_turbidity)),
            'std': float(np.std(valid_turbidity)),
            'min': float(np.min(valid_turbidity)),
            'max': float(np.max(valid_turbidity)),
            'percentile_25': float(np.percentile(valid_turbidity, 25)),
            'percentile_75': float(np.percentile(valid_turbidity, 75))
        }
        
        self.analysis_results['turbidity_stats'] = stats
        
        logger.info(f"Turbidity estimation completed. Mean: {stats['mean']:.2f} NTU")
        
        return turbidity_map
    
    def generate_visualizations(self):
        """Generate all analysis visualizations."""
        logger.info("Generating visualizations...")
        
        if 'turbidity_map' not in self.analysis_results:
            logger.error("No turbidity map available for visualization")
            return
        
        turbidity_map = self.analysis_results['turbidity_map']
        
        # Plot turbidity map
        plot_turbidity_map(turbidity_map, title="Roehne River Turbidity Map")
        
        # Plot histogram
        plot_turbidity_histogram(turbidity_map, title="Turbidity Distribution - Roehne River")
        
        # Plot classification map
        create_turbidity_classification_map(turbidity_map)
        
        # Plot water indices if synthetic image available
        if hasattr(self, 'synthetic_image'):
            plot_water_indices(self.synthetic_image, self.band_names)
        
        logger.info("Visualizations completed")
    
    def generate_report(self) -> str:
        """
        Generate analysis report.
        
        Returns:
            Report text
        """
        logger.info("Generating analysis report...")
        
        report = []
        report.append("ROEHNE RIVER TURBIDITY ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Training metrics
        if 'training_metrics' in self.analysis_results:
            report.append("MODEL PERFORMANCE:")
            metrics = self.analysis_results['training_metrics']
            report.append(f"  Test R²: {metrics.get('test_r2', 'N/A'):.3f}")
            report.append(f"  Test RMSE: {metrics.get('test_rmse', 'N/A'):.3f} NTU")
            report.append(f"  Cross-validation RMSE: {metrics.get('cv_rmse', 'N/A'):.3f} ± {metrics.get('cv_rmse_std', 'N/A'):.3f} NTU")
            report.append("")
        
        # Turbidity statistics
        if 'turbidity_stats' in self.analysis_results:
            report.append("TURBIDITY STATISTICS:")
            stats = self.analysis_results['turbidity_stats']
            report.append(f"  Mean: {stats['mean']:.2f} NTU")
            report.append(f"  Standard Deviation: {stats['std']:.2f} NTU")
            report.append(f"  Range: {stats['min']:.2f} - {stats['max']:.2f} NTU")
            report.append(f"  25th Percentile: {stats['percentile_25']:.2f} NTU")
            report.append(f"  75th Percentile: {stats['percentile_75']:.2f} NTU")
            report.append("")
        
        # Water quality assessment
        if 'turbidity_stats' in self.analysis_results:
            mean_turbidity = self.analysis_results['turbidity_stats']['mean']
            report.append("WATER QUALITY ASSESSMENT:")
            if mean_turbidity < 5:
                assessment = "Clear water"
            elif mean_turbidity < 25:
                assessment = "Slightly turbid"
            elif mean_turbidity < 100:
                assessment = "Turbid"
            else:
                assessment = "Very turbid"
            
            report.append(f"  Overall Classification: {assessment}")
            report.append("")
        
        report.append("METHODOLOGY:")
        report.append("- Satellite imagery from Sentinel-2/Landsat")
        report.append("- Machine learning-based turbidity estimation")
        report.append("- Spectral indices analysis")
        report.append("- Statistical validation")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / "reports" / f"turbidity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_path}")
        
        return report_text
    
    def run_complete_analysis(self, start_date: str = None, end_date: str = None):
        """
        Run the complete turbidity analysis workflow.
        
        Args:
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
        """
        logger.info("Starting complete turbidity analysis workflow...")
        
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Step 1: Setup environment
            self.setup_environment()
            
            # Step 2: Load ground truth data
            ground_truth = self.load_ground_truth_data()
            
            # Step 3: Acquire satellite data
            self.acquire_satellite_data(start_date, end_date)
            
            # Step 4: Train model
            self.train_turbidity_model(ground_truth)
            
            # Step 5: Estimate turbidity
            turbidity_map = self.estimate_turbidity()
            
            # Step 6: Generate visualizations
            self.generate_visualizations()
            
            # Step 7: Generate report
            report = self.generate_report()
            
            logger.info("Complete analysis workflow finished successfully!")
            print("\nANALYSIS SUMMARY:")
            print(report)
            
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Error in analysis workflow: {e}")
            raise


def main():
    """Main function to run the analysis."""
    print("Roehne River Turbidity Analysis")
    print("================================")
    print()
    
    # Initialize analyzer
    analyzer = RoehneTurbidityAnalyzer()
    
    # Run analysis
    try:
        results = analyzer.run_complete_analysis()
        print("\n✓ Analysis completed successfully!")
        print(f"✓ Results saved to: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        logger.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()
