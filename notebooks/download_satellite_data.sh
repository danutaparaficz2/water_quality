#!/bin/bash
# Download helper script for Rhône river satellite data

# Coordinates for Valais, Switzerland
MIN_LON=7.0
MAX_LON=7.8
MIN_LAT=46.0
MAX_LAT=46.5

echo "Rhône River Satellite Data Download Helper"
echo "Study area: $MIN_LAT,$MIN_LON to $MAX_LAT,$MAX_LON"
echo ""
echo "Manual download options:"
echo "1. Copernicus Hub: https://scihub.copernicus.eu/dhus/"
echo "2. USGS Explorer: https://earthexplorer.usgs.gov/"
echo "3. NASA Earthdata: https://search.earthdata.nasa.gov/"
echo ""
echo "Search parameters:"
echo "  - Location: Sion, Valais, Switzerland"
echo "  - Coordinates: 46.25°N, 7.40°E"
echo "  - Satellites: Sentinel-2, Landsat 8/9"
echo "  - Date range: Last 3-6 months"
echo "  - Cloud cover: <20%"
