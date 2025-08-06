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

## 📈 Example Results

The analysis provides comprehensive insights including:

- **Water Coverage**: Percentage of study area covered by water
- **Turbidity Levels**: Quantitative measurements in NTU
- **Water Quality Classification**: Automated quality assessment
- **Temporal Trends**: Changes over time across multiple dates
- **Spatial Distribution**: Maps showing turbidity hotspots

## 🔬 Scientific Methodology

### Spectral Indices Used

- **NDTI (Normalized Difference Turbidity Index)**: `(Red - Green) / (Red + Green)`
- **Red/Green Ratio**: Direct ratio for turbidity indication
- **Total Suspended Matter Proxy**: SWIR band reflectance

### Water Detection Algorithm

- **Primary**: NIR < 0.12 AND SWIR < 0.15
- **Fallback**: NIR < 0.12 (when SWIR unavailable)

### Turbidity Estimation

- **Empirical conversion**: Red reflectance × 200 = NTU estimate
- **Quality thresholds**: 
  - Excellent: < 5 NTU
  - Good: 5-25 NTU
  - Fair: 25-50 NTU
  - Poor: > 50 NTU

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{rhone_turbidity_analysis,
  title={Rhône River Turbidity Analysis from Satellite Images},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/rhone-river-turbidity-analysis}
}
```

## 🙏 Acknowledgments

- **Sentinel-2 Data**: European Space Agency (ESA) Copernicus Programme
- **Python Libraries**: NumPy, Matplotlib, Rasterio, SciPy
- **Inspiration**: Environmental monitoring and water quality research community

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **Institution**: Your Institution
- **Project Link**: [https://github.com/YOUR_USERNAME/rhone-river-turbidity-analysis](https://github.com/YOUR_USERNAME/rhone-river-turbidity-analysis)

---

⭐ **Star this repository if you find it useful!** ⭐
