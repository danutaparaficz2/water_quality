# 🌊 Rhône River Turbidity Analysis from Satellite Images

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive Python project for estimating water turbidity in the Rhône River using Sentinel-2 satellite imagery and multi-spectral analysis techniques.

## 🎯 Overview

This project provides a complete workflow for satellite-based water quality monitoring:
- **Multi-temporal analysis** using Sentinel-2 L2A imagery
- **Advanced spectral indices** for turbidity estimation (NDTI, Red/Green ratio, TSM proxy)
- **Automated water body detection** using NIR and SWIR bands
- **Quantitative turbidity estimation** in NTU (Nephelometric Turbidity Units)
- **Interactive visualizations** ready for scientific presentations
- **Temporal trend analysis** across multiple acquisition dates

## 🚀 Key Features

### 🛰️ Satellite Data Processing
- **Sentinel-2 L2A**: Atmospheric corrected, analysis-ready data
- **Multi-band analysis**: Blue, Green, Red, NIR, SWIR bands
- **Flexible data loading**: Support for multiple dates and missing bands
- **Quality assessment**: Automated band availability checking

### 🔬 Water Quality Analysis
- **Turbidity indices**: NDTI, Red/Green ratio, Total Suspended Matter proxy
- **Water detection**: Advanced NIR+SWIR masking algorithms
- **NTU estimation**: Empirical conversion from spectral reflectance
- **Quality classification**: Excellent, Good, Fair, Poor water quality ratings

### 📊 Visualization & Reporting
- **Multi-temporal maps**: Side-by-side comparison across dates
- **Trend analysis**: Temporal evolution of water quality parameters
- **Presentation-ready outputs**: High-quality figures for scientific communication
- **Export capabilities**: HTML and PDF report generation

## 📁 Project Structure

```
Water/
├── 📁 data/                       # Data storage
│   └── 📁 satellite/             # Sentinel-2 satellite imagery
├── 📁 src/                       # Source code modules
│   ├── 🐍 satellite_data.py      # Data acquisition and preprocessing
│   ├── 🐍 turbidity_estimation.py # Turbidity analysis algorithms
│   └── 🐍 visualization.py       # Plotting and visualization
├── 📁 notebooks/                 # Jupyter notebooks for analysis
│   ├── 📓 rhone_turbidity_clean.ipynb # Main analysis notebook
│   └── 📁 data/                  # Notebook-specific data
├── 📁 output/                    # Analysis results
│   ├── 📁 models/                # Analysis models and parameters
│   ├── 📁 maps/                  # Generated turbidity maps
│   ├── 📁 reports/               # Analysis reports
│   └── 📁 figures/               # Visualization outputs
├── 🐍 main.py                    # Main analysis script
├── 📄 requirements.txt           # Python dependencies
└── 📖 README.md                  # This documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- Git (for cloning the repository)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/rhone-river-turbidity-analysis.git
cd rhone-river-turbidity-analysis
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook notebooks/rhone_turbidity_clean.ipynb
```

## 📊 Usage

### Running the Analysis

The main analysis is contained in the Jupyter notebook `notebooks/rhone_turbidity_clean.ipynb`. This notebook provides:

1. **Data Loading**: Automated loading of Sentinel-2 L2A data for multiple dates
2. **Spectral Analysis**: Calculation of turbidity indices (NDTI, Red/Green ratio, TSM)
3. **Water Detection**: Automated water body identification
4. **Turbidity Estimation**: Quantitative NTU calculations
5. **Visualization**: Multi-temporal maps and trend analysis
6. **Reporting**: Summary statistics and water quality assessment

### Sample Data

The repository includes sample Sentinel-2 data for three dates:
- **June 29, 2025**: Complete dataset (5 bands)
- **July 9, 2025**: Partial dataset (4 bands)
- **July 17, 2025**: Complete dataset (5 bands)

### Adding Your Own Data

To analyze your own Sentinel-2 data:

1. Place TIFF files in `notebooks/data/satellite/new1/`
2. Follow the naming convention: `YYYY-MM-DD-HH:MM_YYYY-MM-DD-HH:MM_Sentinel-2_L2A_BXX_(Raw).tiff`
3. Update the date list in the notebook's data loading section
- **Feature engineering**: Multi-band ratios and transformations
- **Model validation**: Cross-validation and performance metrics

### Visualization
- **Turbidity maps**: Spatial distribution visualization
- **Time series analysis**: Temporal trends
- **Classification maps**: Water quality categories
- **Statistical plots**: Histograms, scatter plots, correlation matrices

## Installation

### Prerequisites

- Python 3.8+ 
- Git
- GDAL (for geospatial libraries)

### Quick Start

1. **Clone and navigate to the repository**:
   ```bash
   cd /Users/danuta.paraficz/PyProjects/Water
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv turbidity_env
   source turbidity_env/bin/activate  # On macOS/Linux
   # turbidity_env\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env with your API credentials
   ```

5. **Run the analysis**:
   ```bash
   python main.py
   ```

### Full Installation (with geospatial libraries)

For complete functionality, install geospatial dependencies:

```bash
# Using conda (recommended for geospatial packages)
conda create -n turbidity python=3.9
conda activate turbidity
conda install -c conda-forge rasterio geopandas earthengine-api sentinelsat
pip install -r requirements.txt
```

## Usage

### Interactive Analysis

Use the Jupyter notebook for interactive analysis:

```bash
jupyter notebook notebooks/roehne_turbidity_analysis.ipynb
```

### Command Line

Run the complete analysis workflow:

```bash
python main.py
```

### Programmatic Usage

```python
from src.satellite_data import SatelliteDataManager
from src.turbidity_estimation import TurbidityEstimator
from src.visualization import plot_turbidity_map

# Initialize components
satellite_manager = SatelliteDataManager()
estimator = TurbidityEstimator()

# Acquire and process data
search_results = satellite_manager.search_sentinel2_data('2023-01-01', '2023-12-31')

# Train model and estimate turbidity
# ... (see examples in notebooks/)
```

## Data Sources and APIs

### Satellite Data Access

1. **Sentinel Hub** (recommended):
   - Register at: https://services.sentinel-hub.com/
   - Free tier available
   - High-quality, preprocessed data

2. **Google Earth Engine**:
   - Register at: https://earthengine.google.com/
   - Cloud-based processing
   - Extensive data catalog

3. **NASA Earthdata**:
   - Register at: https://urs.earthdata.nasa.gov/
   - Landsat and MODIS data
   - Free access

### Ground Truth Data

For model training and validation, you'll need in-situ turbidity measurements. Sources include:
- Local water quality monitoring stations
- Field sampling campaigns
- Water authority databases
- Research collaborations

## Methodology

### Turbidity Estimation Approach

1. **Data Preprocessing**:
   - Atmospheric correction
   - Cloud masking
   - Geometric correction
   - Temporal compositing

2. **Feature Extraction**:
   - Spectral band reflectances
   - Water quality indices (NDWI, MNDWI)
   - Band ratios and transformations
   - Texture features

3. **Model Training**:
   - Random Forest regression
   - Gradient boosting
   - Cross-validation
   - Hyperparameter tuning

4. **Validation**:
   - Independent test set
   - Statistical metrics (R², RMSE, MAE)
   - Spatial and temporal validation

### Roehne River Specifics

- **Location**: Germany (approximate coordinates: 51.5-52.0°N, 8.0-8.5°E)
- **Water type**: River system
- **Typical turbidity range**: 5-100 NTU
- **Seasonal patterns**: Spring snowmelt, summer low flow, autumn storms

## Results and Outputs

### Generated Files

- **Turbidity maps**: GeoTIFF format with spatial projections
- **Time series data**: CSV files with temporal analysis
- **Model files**: Trained models saved as joblib files
- **Reports**: Automated analysis summaries
- **Visualizations**: PNG/PDF plots and maps

### Performance Metrics

Typical model performance on synthetic data:
- **R²**: 0.85-0.95
- **RMSE**: 3-8 NTU
- **MAE**: 2-5 NTU

## Configuration

### Environment Variables

Key settings in `.env` file:
```bash
# API Credentials
SENTINEL_USERNAME=your_username
SENTINEL_PASSWORD=your_password
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Study Area (Roehne River)
ROEHNE_MIN_LON=8.0
ROEHNE_MAX_LON=8.5
ROEHNE_MIN_LAT=51.5
ROEHNE_MAX_LAT=52.0

# Analysis Parameters
DEFAULT_CLOUD_COVER_THRESHOLD=20
```

### Model Parameters

Adjust model settings in `src/turbidity_estimation.py`:
- Number of estimators
- Cross-validation folds
- Feature selection methods
- Validation metrics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## Troubleshooting

### Common Issues

1. **GDAL installation errors**:
   ```bash
   # Use conda for easier GDAL installation
   conda install -c conda-forge gdal
   ```

2. **API authentication errors**:
   - Verify credentials in `.env` file
   - Check API quota limits
   - Ensure proper account registration

3. **Memory issues with large images**:
   - Process smaller tiles
   - Use image pyramids
   - Increase system memory

4. **Missing ground truth data**:
   - Use synthetic data for testing
   - Contact local water authorities
   - Check research databases

### Performance Optimization

- Use cloud computing for large-scale analysis
- Implement parallel processing for multiple images
- Cache preprocessed data
- Use image pyramids for visualization

## References

### Scientific Literature

1. Dörnhöfer, K., & Oppelt, N. (2016). Remote sensing for lake research and monitoring–Recent advances. *Ecological Indicators*, 64, 105-122.

2. Olmanson, L. G., et al. (2008). A 20-year Landsat water clarity census of Minnesota's 10,000 lakes. *Remote Sensing of Environment*, 112(11), 4086-4097.

3. Nechad, B., et al. (2010). Calibration and validation of a generic multisensor algorithm for mapping of total suspended matter in turbid waters. *Remote Sensing of Environment*, 114(4), 854-866.

### Technical Resources

- [Sentinel-2 User Handbook](https://sentinel.esa.int/documents/247904/685211/Sentinel-2_User_Handbook)
- [Google Earth Engine Guides](https://developers.google.com/earth-engine/guides)
- [GDAL Documentation](https://gdal.org/index.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact the maintainers for collaboration opportunities

## Acknowledgments

- European Space Agency (ESA) for Sentinel-2 data
- NASA for Landsat data
- Google Earth Engine team
- Open source geospatial community

---

**Note**: This project is for research and educational purposes. For operational water quality monitoring, consult with domain experts and validate results with ground truth measurements.
