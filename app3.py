import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TensorFlow info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add these lines to silence deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
# Configure TensorFlow to use compatibility mode for v1 functions
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv
import datetime
import requests
from functools import lru_cache
import asyncio
import httpx
import folium
from streamlit_folium import st_folium
import sentinelhub
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from deep_translator import GoogleTranslator
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
import json
import calendar
import matplotlib as mpl
import cv2
from io import BytesIO

def configure_multilingual_fonts():
    """Configure matplotlib to handle multiple languages including Devanagari script"""
    # Try to find a font that supports Devanagari
    font_candidates = ['Nirmala UI', 'Arial Unicode MS', 'NOTO Sans', 'Mangal', 'Lohit Devanagari']
    
    # Check if any of these fonts are available
    font_found = False
    for font in font_candidates:
        try:
            mpl.font_manager.findfont(font)
            mpl.rcParams['font.family'] = font
            font_found = True
            break
        except:
            continue
    
    if not font_found:
        # Fall back to a generic solution
        mpl.rcParams['font.sans-serif'] = ['Nirmala UI', 'Arial Unicode MS', 'DejaVu Sans', 
                                          'Bitstream Vera Sans', 'sans-serif']
        mpl.rcParams['axes.unicode_minus'] = False  # Fix minus symbol

# Call this function when your app starts
configure_multilingual_fonts()

# Modified translation function with proper async handling
@lru_cache(maxsize=1000)
def translate_text(text, target_lang):
    """Synchronous translation with caching and chunking for long texts"""
    if not text or target_lang == 'en':
        return text
    
    try:
        # Maximum chunk size that works reliably with translation APIs
        MAX_CHUNK_SIZE = 4000
        
        # If text is short enough, translate directly
        if len(text) <= MAX_CHUNK_SIZE:
            translator = GoogleTranslator(source='auto', target=target_lang)
            translation = translator.translate(text)
            return translation if translation else text
        
        # For longer texts, split by paragraphs and translate in chunks
        paragraphs = text.split('\n')
        translated_paragraphs = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, translate current chunk
            if len(current_chunk) + len(paragraph) > MAX_CHUNK_SIZE:
                if current_chunk:
                    translator = GoogleTranslator(source='auto', target=target_lang)
                    translated_chunk = translator.translate(current_chunk)
                    translated_paragraphs.append(translated_chunk)
                    current_chunk = paragraph + "\n"
                else:
                    # Handle case where a single paragraph is too long
                    current_chunk = paragraph + "\n"
            else:
                current_chunk += paragraph + "\n"
        
        # Translate any remaining text
        if current_chunk:
            translator = GoogleTranslator(source='auto', target=target_lang)
            translated_chunk = translator.translate(current_chunk)
            translated_paragraphs.append(translated_chunk)
        
        return "".join(translated_paragraphs)
        
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        # Return original text when translation fails, with a small notice
        return f"{text}\n\n[Translation unavailable]"

# Dictionary of text content in English
TEXT_CONTENT = {
    "app_title": "Plant Disease Detection System for Sustainable Agriculture",
    "page_title": "Agri Crop Advisor",
    "select_page": "Select Page",
    "home": "HOME",
    "disease_recognition": "DISEASE RECOGNITION",
    "features_title": "Features:",
    "features": [
        "Disease Detection using AI",
        "Plant disease Analysis",
        "Smart Planting Strategies",
        "AI-Powered Recommendations",
        "Location-based Weather Insights"
    ],
    "location_settings": "Location Details",
    "location_type": "Location Type",
    "select_location": "Select Rural Location",
    "custom_location": "Custom Location",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "choose_image": "Choose an Image:",
    "show_image": "Show Image",
    "predict_disease": "Predict Disease",
    "disease_results": "Disease Detection Results",
    "model_predicting": "Model is Predicting it's a",
    "get_analysis": "Get AI Analysis",
    "generating_analysis": "Generating comprehensive analysis...",
    "translating": "Translating to {}...",
    "current_weather": "Current Weather Conditions",
    "detailed_analysis": "Detailed Analysis",
    "download_report": "Download Analysis Report",
    "predict_first": "Please predict the disease first before getting AI analysis.",
    "soil_analysis": "Soil Analysis",
    "soil_parameters": "Soil Parameters",
    "soil_optional": "Optional: Add soil test results for better recommendations",
    "soil_ph": "Soil pH",
    "soil_nitrogen": "Nitrogen (N) ppm",
    "soil_phosphorus": "Phosphorus (P) ppm",
    "soil_potassium": "Potassium (K) ppm",
    "soil_organic": "Organic Matter %",
    "soil_texture": "Soil Texture",
    "soil_analysis_results": "Soil Analysis Results",
    "no_soil_data": "No soil data provided. Add soil test results for more comprehensive recommendations.",
    "include_soil": "Include Soil Data in Analysis",
    "satellite_monitoring": "SATELLITE MONITORING",
    "satellite_title": "Agricultural Area Satellite Monitoring",
    "select_area": "Select Area of Interest",
    "area_size": "Area Size (km²)",
    "fetch_imagery": "Fetch Satellite Imagery",
    "select_index": "Select Vegetation Index",
    "ndvi": "NDVI (Vegetation Health)",
    "evi": "EVI (Enhanced Vegetation)",
    "moisture": "NDMI (Moisture Index)",
    "time_period": "Time Period",
    "last_month": "Last Month",
    "last_3_months": "Last 3 Months",
    "last_year": "Last Year",
    "custom_date": "Custom Date Range",
    "start_date": "Start Date",
    "end_date": "End Date",
    "analyzing_satellite": "Analyzing satellite imagery...",
    "satellite_results": "Satellite Monitoring Results",
    "satellite_analysis": "Satellite Analysis",
    "healthy_vegetation": "Healthy Vegetation",
    "stressed_vegetation": "Stressed Vegetation",
    "bare_soil": "Bare Soil/Low Vegetation",
    "download_satellite": "Download Satellite Report",
    "vegetation_analysis": "Vegetation Health Analysis",
    "no_satellite_data": "No satellite data available for the selected region or time period."
}

TEXT_CONTENT.update({
    "climate_analysis": "HISTORICAL CLIMATE ANALYSIS",
    "climate_title": "Historical Climate Data Analysis",
    "climate_subtitle": "Analyze long-term weather patterns and their impact on agriculture",
    "climate_period": "Analysis Period",
    "last_5_years": "Last 5 Years",
    "last_10_years": "Last 10 Years", 
    "last_20_years": "Last 20 Years",
    "data_type": "Climate Data Type",
    "temperature": "Temperature",
    "precipitation": "Precipitation",
    "humidity": "Humidity",
    "analyze_climate": "Analyze Climate Data",
    "analyzing_climate": "Analyzing climate patterns...",
    "climate_results": "Climate Analysis Results",
    "temperature_trends": "Temperature Trends",
    "precipitation_trends": "Precipitation Trends",
    "seasonal_patterns": "Seasonal Patterns",
    "climate_anomalies": "Climate Anomalies",
    "disease_correlation": "Climate-Disease Correlation",
    "download_climate": "Download Climate Report",
    "no_climate_data": "No climate data available for the selected region or time period.",
    "monthly_avg": "Monthly Averages",
    "yearly_trends": "Yearly Trends",
    "extreme_events": "Extreme Weather Events",
    "annual_comparison": "Annual Comparison"
})

TEXT_CONTENT.update({
    "yield_impact": "YIELD IMPACT PREDICTION",
    "disease_severity": "Disease Severity Assessment",
    "crop_loss": "Estimated Crop Loss",
    "severity_low": "Low",
    "severity_moderate": "Moderate", 
    "severity_high": "High",
    "severity_very_high": "Very High",
    "spread_rate": "Disease Spread Rate",
    "spread_slow": "Slow",
    "spread_moderate": "Moderate",
    "spread_fast": "Fast",
    "spread_very_fast": "Very Fast",
    "yield_loss": "Potential Yield Loss",
    "intervention_impact": "Impact of Early Intervention",
    "yield_recovery": "Potential Yield Recovery with Treatment",
    "analyzing_severity": "Analyzing disease severity...",
    "affected_area": "Affected Area",
    "recovery_potential": "Recovery Potential",
    "economic_impact": "Economic Impact",
    "analysis_confidence": "Analysis Confidence",
    "confidence_factors": "Confidence Factors",
    "disease_detection": "Disease Detection Visualization"
})

# Predefined rural locations in India with their coordinates
RURAL_LOCATIONS = {
    "Select Location": (0.0, 0.0),
    "Anantapur, Andhra Pradesh": (14.6833, 77.6000),
    "Bargarh, Odisha": (21.3447, 83.6219),
    "Bhind, Madhya Pradesh": (26.5667, 78.7667),
    "Chittorgarh, Rajasthan": (24.8833, 74.4667),
    "Dharwad, Karnataka": (15.4589, 75.0226),
    "Faizabad, Uttar Pradesh": (26.7833, 82.1500),
    "Gandhinagar, Gujarat": (23.2156, 72.6839),
    "Haridwar, Uttarakhand": (29.9457, 78.1642),
    "Imphal, Manipur": (24.8170, 93.9368),
    "Jaipur, Rajasthan": (26.9124, 75.7873),
    "Karnal, Haryana": (29.6900, 76.9900),
    "Malappuram, Kerala": (11.0568, 76.0745),
    "Nagpur, Maharashtra": (21.1458, 79.0882),
    "Patna, Bihar": (25.5941, 85.1376),
    "Raipur, Chhattisgarh": (21.2514, 81.6296),
    "Shimla, Himachal Pradesh": (31.1048, 77.1734),
    "Thrissur, Kerala": (10.5167, 76.2167),
    "Udaipur, Rajasthan": (24.5854, 73.6535),
    "Warangal, Telangana": (17.9750, 80.6167),
    "Custom Location": None  # Placeholder for custom location
}

INDIAN_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Urdu": "ur",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Odia": "or",
    "Sanskrit": "sa"
}

# Add after your INDIAN_LANGUAGES dictionary
SOIL_TEXTURES = [
    "Unknown/Not Tested",
    "Sandy",
    "Sandy Loam",
    "Loam",
    "Clay Loam",
    "Clay",
    "Silt Loam",
    "Silty Clay"
]

# Load environment variables and configure APIs
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Create a single SHConfig instance and configure it
sh_config = sentinelhub.SHConfig()
sh_config.instance_id = os.getenv('SENTINEL_INSTANCE_ID')
sh_config.sh_client_id = os.getenv('SENTINEL_CLIENT_ID')
sh_config.sh_client_secret = os.getenv('SENTINEL_CLIENT_SECRET')
sh_config.save()

def get_weather_data(latitude, longitude):
    """Fetch weather data for given coordinates"""
    try:
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': latitude,
            'lon': longitude,
            'appid': OPENWEATHERMAP_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            weather_data = response.json()
            return {
                'temperature': weather_data['main']['temp'],
                'humidity': weather_data['main']['humidity'],
                'description': weather_data['weather'][0]['description'],
                'wind_speed': weather_data['wind']['speed']
            }
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def model_prediction(test_image):
    """Predict plant disease from image"""
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

def get_gemini_analysis(plant_disease, weather_data=None, soil_data=None, target_lang='en'):
    """Get AI analysis with translation including soil data"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        weather_context = ""
        if weather_data:
            weather_context = f"""
            Current Weather Conditions:
            - Temperature: {weather_data['temperature']}°C
            - Humidity: {weather_data['humidity']}%
            - Weather Description: {weather_data['description']}
            - Wind Speed: {weather_data['wind_speed']} m/s
            """
            
        soil_context = ""
        if soil_data:
            soil_context = f"""
            Soil Analysis Results:
            - Soil pH: {soil_data['pH']}
            - Nitrogen (N): {soil_data['nitrogen']} ppm
            - Phosphorus (P): {soil_data['phosphorus']} ppm
            - Potassium (K): {soil_data['potassium']} ppm
            - Organic Matter: {soil_data['organic_matter']}%
            - Soil Texture: {soil_data['texture']}
            """
        
        prompt = f"""
        Act as a plant disease expert and provide recommendations for sustainable agriculture.
        Analyze the following plant disease and provide detailed recommendations:
        Disease: {plant_disease}
        
        {weather_context}
        
        {soil_context}
        
        Please provide a comprehensive analysis including:
        1. Treatment Recommendations:
           - Natural treatment methods
           - Preventive measures
        2. Pest Resistance Strategies
        3. Optimal Planting Strategy
        4. Disease Management
        5. Soil Management Recommendations:
           {"- Specific to the provided soil test results" if soil_data else "- General recommendations (no soil data provided)"}
        6. Additional Recommendations
        
        Format the response in a clear, structured manner with bullet points.
        """
        
        response = model.generate_content(prompt)
        analysis = response.text
        
        # Translate analysis if not English
        if target_lang != 'en':
            analysis = translate_text(analysis, target_lang)
        
        return analysis
    except Exception as e:
        return f"Error in getting AI analysis: {str(e)}"

def collect_soil_data(language='en'):
    """Collect soil testing parameters from user"""
    soil_data = {}
    
    # Create expander for soil data
    with st.sidebar.expander(translate_text(TEXT_CONTENT["soil_optional"], language)):
        # Soil pH (typical range 3.5-10)
        soil_data["pH"] = st.slider(
            translate_text(TEXT_CONTENT["soil_ph"], language),
            min_value=3.5, max_value=10.0, value=7.0, step=0.1
        )
        
        # Soil NPK values
        col1, col2 = st.columns(2)
        with col1:
            soil_data["nitrogen"] = st.number_input(
                translate_text(TEXT_CONTENT["soil_nitrogen"], language),
                min_value=0, value=0
            )
        with col2:
            soil_data["phosphorus"] = st.number_input(
                translate_text(TEXT_CONTENT["soil_phosphorus"], language),
                min_value=0, value=0
            )
        
        soil_data["potassium"] = st.number_input(
            translate_text(TEXT_CONTENT["soil_potassium"], language),
            min_value=0, value=0
        )
        
        # Organic matter (0-100%)
        soil_data["organic_matter"] = st.slider(
            translate_text(TEXT_CONTENT["soil_organic"], language),
            min_value=0.0, max_value=15.0, value=2.0, step=0.1
        )
        
        # Soil texture
        soil_data["texture"] = st.selectbox(
            translate_text(TEXT_CONTENT["soil_texture"], language),
            SOIL_TEXTURES
        )
        
        # Flag to include soil data
        include_soil = st.checkbox(
            translate_text(TEXT_CONTENT["include_soil"], language),
            value=True
        )
        
        if not include_soil:
            return None
            
    return soil_data

def fetch_satellite_data(latitude, longitude, area_size=1.0, index_type="NDVI", start_date=None, end_date=None):
    """
    Fetch satellite imagery data for a specific location with specified parameters
    """
    try:
        # Validate configuration
        if not sh_config.instance_id or not sh_config.sh_client_id or not sh_config.sh_client_secret:
            return {
                'has_data': False,
                'message': "Sentinel Hub API credentials not properly configured."
            }
        
        # Calculate bounding box based on approximate conversion (1km ~ 0.009 degrees)
        half_side = (area_size ** 0.5) * 0.0045  # Approximate conversion to degrees
        bbox = sentinelhub.BBox(
            (longitude - half_side, latitude - half_side, 
             longitude + half_side, latitude + half_side), 
            crs=sentinelhub.CRS.WGS84
        )
        
        # Set up time interval
        if not start_date:
            end_date = datetime.datetime.now()
            start_date = end_date - timedelta(days=30)  # Default to last 30 days
            
        time_interval = (start_date, end_date)
        
        # Configure evalscript based on index type
        if index_type == "NDVI":
            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B08", "dataMask"],
                    output: { bands: 4 }
                };
            }
            
            function evaluatePixel(sample) {
                let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
                
                // NDVI color scheme: red (low) to green (high)
                let red = 0;
                let green = 0;
                let blue = 0;
                
                if (ndvi < 0) {
                    red = 1; green = 0; blue = 0;  // Water/non-vegetation
                } else if (ndvi < 0.3) {
                    red = 1 - ndvi; green = ndvi; blue = 0;  // Sparse vegetation
                } else {
                    red = 0; green = 1; blue = 0;  // Dense vegetation
                }
                
                return [red, green, blue, sample.dataMask];
            }
            """
        elif index_type == "EVI":
            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B02", "B04", "B08", "dataMask"],
                    output: { bands: 4 }
                };
            }
            
            function evaluatePixel(sample) {
                // Enhanced Vegetation Index
                let G = 2.5;
                let C1 = 6;
                let C2 = 7.5;
                let L = 1;
                
                let evi = G * ((sample.B08 - sample.B04) / (sample.B08 + C1 * sample.B04 - C2 * sample.B02 + L));
                
                // Normalize EVI values (typically from -1 to 1) to color range
                let nevi = Math.max(0, Math.min(1, (evi + 0.2) / 1.2));
                
                return [1-nevi, nevi, 0, sample.dataMask];
            }
            """
        elif index_type == "NDMI":
            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B08", "B11", "dataMask"],
                    output: { bands: 4 }
                };
            }
            
            function evaluatePixel(sample) {
                // Normalized Difference Moisture Index
                let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11);
                
                // Convert to color: blue (high moisture) to yellow (low moisture)
                let moisture = Math.max(-1, Math.min(1, ndmi));
                let norm_moisture = (moisture + 1) / 2; // Scale from -1,1 to 0,1
                
                return [1-norm_moisture, 1-norm_moisture, norm_moisture, sample.dataMask];
            }
            """
                
        # Create request and get data, using the proper configuration
        request = sentinelhub.SentinelHubRequest(
            data_folder="sentinel_data",
            evalscript=evalscript,
            input_data=[
                sentinelhub.SentinelHubRequest.input_data(
                    data_collection=sentinelhub.DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order='leastCC'  # Least cloud coverage
                )
            ],
            responses=[
                sentinelhub.SentinelHubRequest.output_response('default', sentinelhub.MimeType.PNG)
            ],
            bbox=bbox,
            size=(512, 512),  # Image size in pixels
            config=sh_config  # Use the saved configuration
        )
        
        # Get data with proper error handling
        try:
            data = request.get_data()
            
            # Check if we have valid data
            if len(data) > 0 and data[0] is not None:
                # Check for empty image (all zeros)
                if np.max(data[0]) == 0:
                    return {
                        'has_data': False,
                        'message': "No valid satellite imagery found for the selected location and time period."
                    }
                
                # Return the image data and some basic analysis
                result = {
                    'image': data[0],
                    'index_type': index_type,
                    'bbox': bbox,
                    'time_interval': time_interval,
                    'has_data': True
                }
                
                return result
            else:
                return {
                    'has_data': False,
                    'message': "No satellite data available for the selected parameters."
                }
        except sentinelhub.SHRuntimeError as e:
            return {
                'has_data': False,
                'message': f"Sentinel Hub runtime error: {str(e)}"
            }
            
    except Exception as e:
        return {
            'has_data': False,
            'message': f"Error fetching satellite data: {str(e)}"
        }

def visualize_satellite_data(satellite_data, language='en'):
    """Create visual representation of satellite data with analysis"""
    if not satellite_data or not satellite_data.get('has_data'):
        st.warning(translate_text(TEXT_CONTENT["no_satellite_data"], language))
        return

    # Create figure with satellite image
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the satellite image
    ax.imshow(satellite_data['image'])
    ax.set_title(translate_text(f"{satellite_data['index_type']} Index", language), fontsize=14)
    ax.axis('off')
    
    # Add legend based on index type
    if satellite_data['index_type'] == "NDVI":
        legend_elements = [
            mpatches.Patch(color='darkgreen', label=translate_text(TEXT_CONTENT["healthy_vegetation"], language)),
            mpatches.Patch(color='lightgreen', label=translate_text(TEXT_CONTENT["stressed_vegetation"], language)),
            mpatches.Patch(color='red', label=translate_text(TEXT_CONTENT["bare_soil"], language))
        ]
    elif satellite_data['index_type'] == "EVI":
        legend_elements = [
            mpatches.Patch(color='green', label=translate_text(TEXT_CONTENT["healthy_vegetation"], language)),
            mpatches.Patch(color='yellow', label=translate_text(TEXT_CONTENT["stressed_vegetation"], language)),
            mpatches.Patch(color='red', label=translate_text(TEXT_CONTENT["bare_soil"], language))
        ]
    elif satellite_data['index_type'] == "NDMI":
        legend_elements = [
            mpatches.Patch(color='blue', label="High Moisture"),
            mpatches.Patch(color='cyan', label="Medium Moisture"),
            mpatches.Patch(color='yellow', label="Low Moisture")
        ]
    
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Display the figure
    st.pyplot(fig)
    
    # Display time information
    start_date = satellite_data['time_interval'][0].strftime('%Y-%m-%d')
    end_date = satellite_data['time_interval'][1].strftime('%Y-%m-%d')
    st.write(f"**{translate_text('Time Period', language)}:** {start_date} to {end_date}")

def fetch_historical_climate_data(latitude, longitude, years_back=5):
    """
    Fetch historical climate data for a given location
    """
    try:
        # Calculate date range
        end_date = datetime.datetime.now()  # Fix: Use datetime.datetime.now() instead of datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        # Format dates for API
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Using Open-Meteo Historical API which is free and doesn't require API key
        base_url = "https://archive-api.open-meteo.com/v1/era5"
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_str,
            'end_date': end_str,
            'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,relative_humidity_2m_max,relative_humidity_2m_min',
            'timezone': 'auto'
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if we have valid data
            if 'daily' in data:
                # Convert to pandas DataFrame
                df = pd.DataFrame({
                    'date': pd.to_datetime(data['daily']['time']),
                    'temp_max': data['daily']['temperature_2m_max'],
                    'temp_min': data['daily']['temperature_2m_min'],
                    'temp_mean': data['daily']['temperature_2m_mean'],
                    'precipitation': data['daily']['precipitation_sum'],
                    'humidity_max': data['daily']['relative_humidity_2m_max'],
                    'humidity_min': data['daily']['relative_humidity_2m_min']
                })
                
                # Set date as index
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Add month, year, season columns for easier analysis
                df['month'] = df.index.month
                df['year'] = df.index.year
                df['season'] = df.index.month.map(lambda m: 'Winter' if m in [12, 1, 2] else
                                                'Spring' if m in [3, 4, 5] else
                                                'Summer' if m in [6, 7, 8] else 'Fall')
                
                return df
            else:
                return None
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching historical climate data: {str(e)}")
        return None

def analyze_climate_data(climate_data, data_type="temperature", language="en"):
    """
    Analyze and visualize historical climate data
    
    Args:
        climate_data (pandas.DataFrame): Historical climate data
        data_type (str): Type of data to analyze (temperature, precipitation, humidity)
        language (str): Language code for translations
        
    Returns:
        dict: Analysis results and figures
    """
    if climate_data is None or climate_data.empty:
        return None
    
    results = {
        'has_data': True,
        'insights': {},
        'figures': []
    }
    
    try:
        # Create a figure with subplots
        fig = plt.figure(figsize=(14, 16))
        
        # Plot 1: Time series of the selected data type
        ax1 = plt.subplot(3, 1, 1)
        
        if data_type == "temperature":
            climate_data['temp_mean'].plot(ax=ax1, color='red', label=translate_text('Mean Temperature', language))
            climate_data['temp_max'].plot(ax=ax1, color='orangered', alpha=0.7, label=translate_text('Max Temperature', language))
            climate_data['temp_min'].plot(ax=ax1, color='darkred', alpha=0.7, label=translate_text('Min Temperature', language))
            ax1.set_ylabel('°C')
            ax1.set_title(translate_text(TEXT_CONTENT['temperature_trends'], language))
            
            # Calculate temperature insights
            avg_temp = climate_data['temp_mean'].mean()
            max_temp = climate_data['temp_max'].max()
            min_temp = climate_data['temp_min'].min()
            temp_trend = climate_data.groupby('year')['temp_mean'].mean().diff().mean()
            
            results['insights']['avg_temp'] = avg_temp
            results['insights']['max_temp'] = max_temp
            results['insights']['min_temp'] = min_temp
            results['insights']['temp_trend'] = temp_trend
            
        elif data_type == "precipitation":
            climate_data['precipitation'].plot(ax=ax1, color='blue', label=translate_text('Precipitation', language))
            ax1.set_ylabel('mm')
            ax1.set_title(translate_text(TEXT_CONTENT['precipitation_trends'], language))
            
            # Calculate precipitation insights
            total_precip = climate_data['precipitation'].sum()
            max_precip_day = climate_data['precipitation'].idxmax()
            max_precip = climate_data.loc[max_precip_day, 'precipitation']
            avg_precip = climate_data['precipitation'].mean()
            
            results['insights']['total_precip'] = total_precip
            results['insights']['max_precip'] = max_precip
            results['insights']['max_precip_day'] = max_precip_day.strftime('%Y-%m-%d')
            results['insights']['avg_precip'] = avg_precip
            
        elif data_type == "humidity":
            climate_data['humidity_max'].plot(ax=ax1, color='teal', alpha=0.7, label=translate_text('Max Humidity', language))
            climate_data['humidity_min'].plot(ax=ax1, color='darkturquoise', alpha=0.7, label=translate_text('Min Humidity', language))
            ax1.set_ylabel('%')
            ax1.set_title(translate_text('Humidity Trends', language))
            
            # Calculate humidity insights
            avg_humidity_max = climate_data['humidity_max'].mean()
            avg_humidity_min = climate_data['humidity_min'].mean()
            
            results['insights']['avg_humidity_max'] = avg_humidity_max
            results['insights']['avg_humidity_min'] = avg_humidity_min
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Monthly averages (seasonal patterns)
        ax2 = plt.subplot(3, 1, 2)
        
        if data_type == "temperature":
            monthly_temp = climate_data.groupby('month')['temp_mean'].mean()
            monthly_temp.plot(kind='bar', ax=ax2, color='indianred')
            ax2.set_ylabel('°C')
            
        elif data_type == "precipitation":
            monthly_precip = climate_data.groupby('month')['precipitation'].mean()
            monthly_precip.plot(kind='bar', ax=ax2, color='royalblue')
            ax2.set_ylabel('mm')
            
        elif data_type == "humidity":
            monthly_humidity = climate_data.groupby('month')['humidity_max'].mean()
            monthly_humidity.plot(kind='bar', ax=ax2, color='teal')
            ax2.set_ylabel('%')
        
        # Replace month numbers with month names
        month_names = [calendar.month_name[i] for i in range(1, 13)]
        ax2.set_xticklabels([translate_text(m[:3], language) for m in month_names], rotation=45)
        ax2.set_title(translate_text(TEXT_CONTENT['monthly_avg'], language))
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Year-over-year comparison
        ax3 = plt.subplot(3, 1, 3)
        
        # Group data by year and calculate annual statistics
        yearly_data = climate_data.groupby('year')
        
        if data_type == "temperature":
            yearly_temp = yearly_data['temp_mean'].mean()
            yearly_temp_max = yearly_data['temp_max'].max()
            yearly_temp_min = yearly_data['temp_min'].min()
            
            x = range(len(yearly_temp))
            width = 0.3
            
            ax3.bar([i-width for i in x], yearly_temp_min, width=width, color='darkblue', label=translate_text('Min', language))
            ax3.bar(x, yearly_temp, width=width, color='royalblue', label=translate_text('Avg', language))
            ax3.bar([i+width for i in x], yearly_temp_max, width=width, color='lightblue', label=translate_text('Max', language))
            
            ax3.set_ylabel('°C')
            
        elif data_type == "precipitation":
            yearly_precip = yearly_data['precipitation'].sum()
            yearly_precip.plot(kind='bar', ax=ax3, color='dodgerblue')
            ax3.set_ylabel('mm')
            
        elif data_type == "humidity":
            yearly_humidity_max = yearly_data['humidity_max'].mean()
            yearly_humidity_min = yearly_data['humidity_min'].mean()
            
            yearly_humidity_max.plot(kind='bar', ax=ax3, color='teal', alpha=0.7, label=translate_text('Max', language))
            yearly_humidity_min.plot(kind='bar', ax=ax3, color='darkturquoise', alpha=0.7, label=translate_text('Min', language))
            ax3.set_ylabel('%')
        
        ax3.set_title(translate_text(TEXT_CONTENT['yearly_trends'], language))
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Save figure to results
        results['figures'].append(fig)
        
        return results
    except Exception as e:
        st.error(f"Error analyzing climate data: {str(e)}")
        return None

def display_climate_analysis(analysis_results, data_type, language='en'):
    """Display climate analysis results with insights and visualizations"""
    if not analysis_results or not analysis_results.get('has_data'):
        st.warning(translate_text(TEXT_CONTENT["no_climate_data"], language))
        return
    
    # Display figures
    for fig in analysis_results.get('figures', []):
        st.pyplot(fig)
    
    # Display key insights based on data type
    st.subheader(translate_text("Climate Insights", language))
    
    if data_type == "temperature":
        st.write(f"**{translate_text('Average Temperature', language)}:** {analysis_results['insights'].get('avg_temp', 0):.1f}°C")
        st.write(f"**{translate_text('Maximum Temperature', language)}:** {analysis_results['insights'].get('max_temp', 0):.1f}°C")
        st.write(f"**{translate_text('Minimum Temperature', language)}:** {analysis_results['insights'].get('min_temp', 0):.1f}°C")
        
        temp_trend = analysis_results['insights'].get('temp_trend', 0)
        trend_description = translate_text("warming", language) if temp_trend > 0.01 else (
            translate_text("cooling", language) if temp_trend < -0.01 else translate_text("stable", language))
        
        st.write(f"**{translate_text('Temperature Trend', language)}:** {trend_description} ({temp_trend:.2f}°C/year)")
        
    elif data_type == "precipitation":
        st.write(f"**{translate_text('Total Precipitation', language)}:** {analysis_results['insights'].get('total_precip', 0):.1f} mm")
        st.write(f"**{translate_text('Average Daily Precipitation', language)}:** {analysis_results['insights'].get('avg_precip', 0):.2f} mm")
        st.write(f"**{translate_text('Maximum Daily Precipitation', language)}:** {analysis_results['insights'].get('max_precip', 0):.1f} mm on {analysis_results['insights'].get('max_precip_day', '')}")
    
    elif data_type == "humidity":
        st.write(f"**{translate_text('Average Maximum Humidity', language)}:** {analysis_results['insights'].get('avg_humidity_max', 0):.1f}%")
        st.write(f"**{translate_text('Average Minimum Humidity', language)}:** {analysis_results['insights'].get('avg_humidity_min', 0):.1f}%")

def analyze_disease_severity(image_file, disease_name):
    """
    Analyzes disease severity from the plant image
    
    Args:
        image_file: Image file
        disease_name: Detected disease name
        
    Returns:
        dict: Disease severity analysis results
    """
    # Load and convert image for processing
    image_bytes = image_file.getvalue() if hasattr(image_file, 'getvalue') else image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV color space for better disease detection
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Detect disease areas based on color variations and disease type
    if "rust" in disease_name.lower():
        # Detect orange/brown rust spots
        lower_bound = np.array([10, 60, 60])
        upper_bound = np.array([30, 255, 255])
    elif "blight" in disease_name.lower():
        # Detect brown/black blight spots
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([180, 255, 100])
    elif "powdery" in disease_name.lower() or "mildew" in disease_name.lower():
        # Detect whitish powdery mildew
        lower_bound = np.array([0, 0, 180])
        upper_bound = np.array([180, 30, 255])
    elif "mosaic" in disease_name.lower() or "virus" in disease_name.lower():
        # Detect yellowing from viral diseases
        lower_bound = np.array([20, 100, 100])
        upper_bound = np.array([35, 255, 255])
    elif "scab" in disease_name.lower():
        # Detect dark scab spots
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([180, 255, 80])
    elif "bacterial" in disease_name.lower():
        # Detect water-soaked lesions of bacterial infections
        lower_bound = np.array([40, 0, 0])
        upper_bound = np.array([80, 255, 255])
    else:
        # Generic disease detection (yellowish/brownish areas)
        lower_bound = np.array([15, 30, 30])
        upper_bound = np.array([35, 255, 255])
    
    # Create mask of diseased areas
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    
    # Calculate percentage of affected area
    total_pixels = mask.size
    affected_pixels = cv2.countNonZero(mask)
    disease_percentage = (affected_pixels / total_pixels) * 100
    
    # Determine severity based on percentage affected
    if disease_percentage < 5:
        severity = "severity_low"
        severity_score = 1
    elif disease_percentage < 15:
        severity = "severity_moderate"
        severity_score = 2
    elif disease_percentage < 30:
        severity = "severity_high"
        severity_score = 3
    else:
        severity = "severity_very_high"
        severity_score = 4
    
    # Determine disease spread pattern
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    # Average size of disease spots
    if num_labels > 1:
        avg_spot_size = np.mean(stats[1:, cv2.CC_STAT_AREA])
        spot_density = (num_labels - 1) / total_pixels * 10000  # Spots per 10,000 pixels
    else:
        avg_spot_size = 0
        spot_density = 0
    
    # Determine spread rate based on spot characteristics
    if spot_density < 0.5:
        spread_rate = "spread_slow"
        spread_score = 1
    elif spot_density < 1.5:
        spread_rate = "spread_moderate"
        spread_score = 2
    elif spot_density < 3:
        spread_rate = "spread_fast"
        spread_score = 3
    else:
        spread_rate = "spread_very_fast"
        spread_score = 4
    
    # Create visualization of detected disease areas
    disease_highlight = img_array.copy()
    disease_highlight[mask > 0] = [255, 0, 0]  # Highlight diseased areas in red
    
    # Estimated crop loss based on severity and spread
    base_loss = {
        1: 5,    # Low severity: ~5% loss
        2: 15,   # Moderate severity: ~15% loss
        3: 30,   # High severity: ~30% loss
        4: 50    # Very high severity: ~50% loss
    }
    
    spread_multiplier = {
        1: 0.8,  # Slow spread: 0.8x base loss
        2: 1.0,  # Moderate spread: 1.0x base loss
        3: 1.5,  # Fast spread: 1.5x base loss
        4: 2.0   # Very fast spread: 2.0x base loss
    }
    
    estimated_loss = base_loss[severity_score] * spread_multiplier[spread_score]
    estimated_loss = min(estimated_loss, 95)  # Cap at 95% loss
    
    # Estimate impact of early intervention
    if severity_score <= 2 and spread_score <= 2:
        recovery_potential = "High"
        recovery_percentage = 80
    elif severity_score <= 3 and spread_score <= 3:
        recovery_potential = "Moderate"
        recovery_percentage = 50
    else:
        recovery_potential = "Low"
        recovery_percentage = 20
    
    return {
        "disease_percentage": disease_percentage,
        "severity": severity,
        "severity_score": severity_score,
        "spread_rate": spread_rate,
        "spread_score": spread_score,
        "estimated_loss": estimated_loss,
        "recovery_potential": recovery_potential,
        "recovery_percentage": recovery_percentage,
        "disease_mask": mask,
        "disease_highlight": disease_highlight
    }

def predict_yield_impact(disease_name, disease_analysis, weather_data=None, soil_data=None):
    """
    Predicts crop yield impact based on disease and environmental factors
    
    Args:
        disease_name: Name of the detected disease
        disease_analysis: Disease severity analysis results
        weather_data: Weather data (optional)
        soil_data: Soil data (optional)
        
    Returns:
        dict: Yield impact predictions
    """
    # Base yield impact from disease analysis
    base_loss_percentage = disease_analysis["estimated_loss"]
    
    # Adjust based on weather conditions if available
    weather_multiplier = 1.0
    if weather_data:
        humidity = weather_data.get('humidity', 60)
        temperature = weather_data.get('temperature', 25)
        
        # High humidity increases disease spread for most fungal diseases
        if humidity > 80 and ("rust" in disease_name.lower() or "blight" in disease_name.lower() or "mildew" in disease_name.lower()):
            weather_multiplier *= 1.2
        
        # Temperature effects depend on the disease
        if "blight" in disease_name.lower() and temperature > 25:
            weather_multiplier *= 1.15  # Blight thrives in warm temperatures
        elif "powdery mildew" in disease_name.lower() and temperature > 28:
            weather_multiplier *= 0.9  # Powdery mildew is inhibited by very high temperatures
    
    # Adjust based on soil conditions if available
    soil_multiplier = 1.0
    if soil_data:
        ph = soil_data.get('pH', 7.0)
        
        # Soil pH can influence disease severity
        if "scab" in disease_name.lower() and ph > 6.5:
            soil_multiplier *= 1.2  # Potato scab thrives in alkaline soil
        elif "clubroot" in disease_name.lower() and ph < 6.0:
            soil_multiplier *= 1.3  # Clubroot thrives in acidic soil
        
        # Nutrient balance affects plant resilience
        if soil_data.get('potassium', 100) < 50:
            soil_multiplier *= 1.1  # Low potassium reduces disease resistance
    
    # Final yield loss calculation
    adjusted_loss = base_loss_percentage * weather_multiplier * soil_multiplier
    adjusted_loss = min(adjusted_loss, 95)  # Cap at 95%
    
    # Calculate economic impact (simplified example - would need crop pricing data)
    economic_impact_per_hectare = {
        "Apple": 500,
        "Corn": 200,
        "Grape": 800,
        "Potato": 250,
        "Tomato": 300,
        "Pepper": 280,
        "Strawberry": 900,
        "Peach": 600,
        "Cherry": 750,
        "Blueberry": 850,
        "Soybean": 180,
        "Squash": 220,
        "Orange": 450,
        "Raspberry": 800
    }
    
    # Extract crop type from disease name
    crop_type = disease_name.split("___")[0].split(",")[0]
    
    # Calculate economic impact
    base_value = economic_impact_per_hectare.get(crop_type, 300)  # Default $300 per hectare
    economic_impact = (adjusted_loss / 100) * base_value
    
    # Treatment recommendations and timeline
    if disease_analysis["severity_score"] <= 2:
        treatment_timeline = "Early intervention recommended within 3-5 days"
        untreated_outcome = "If untreated, disease may progress to moderate/severe within 2-3 weeks"
    elif disease_analysis["severity_score"] == 3:
        treatment_timeline = "Urgent treatment required within 1-2 days"
        untreated_outcome = "If untreated, significant crop damage likely within 1-2 weeks"
    else:
        treatment_timeline = "Immediate treatment required"
        untreated_outcome = "If untreated, major crop failure likely within days"
    
    # Prediction confidence
    confidence_factors = []
    confidence_score = 70  # Base confidence
    
    if weather_data:
        confidence_score += 10
        confidence_factors.append("Weather data available")
    
    if soil_data:
        confidence_score += 10
        confidence_factors.append("Soil data available")
    
    # Return results
    return {
        "base_loss_percentage": base_loss_percentage,
        "adjusted_loss": adjusted_loss,
        "economic_impact": economic_impact,
        "crop_type": crop_type,
        "treatment_timeline": treatment_timeline,
        "untreated_outcome": untreated_outcome,
        "confidence": confidence_score,
        "confidence_factors": confidence_factors,
        "recovery_potential": disease_analysis["recovery_potential"],
        "recovery_percentage": disease_analysis["recovery_percentage"]
    }

def display_yield_impact(disease_name, original_image, disease_analysis, yield_prediction, language='en'):
    """
    Displays yield impact prediction results
    
    Args:
        disease_name: Name of the detected disease
        original_image: Original uploaded image
        disease_analysis: Disease severity analysis
        yield_prediction: Yield impact prediction
        language: Language for translation
    """
    # Display disease severity section
    st.subheader(translate_text(TEXT_CONTENT["yield_impact"], language))
    
    # Display disease highlights
    col1, col2 = st.columns(2)
    
    with col1:
        # For original image, we already have the file object
        st.image(original_image, caption=translate_text("Original Image", language), use_container_width=True)
    
    with col2:
        # Convert numpy array to image for display
        highlight_img = Image.fromarray(disease_analysis["disease_highlight"])
        st.image(highlight_img, caption=translate_text(TEXT_CONTENT["disease_detection"], language), use_container_width=True)
    
    # Disease severity metrics
    st.write("### " + translate_text(TEXT_CONTENT["disease_severity"], language))
    
    # Create 4 columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Display severity
    severity_text = translate_text(TEXT_CONTENT[disease_analysis["severity"]], language)
    with col1:
        severity_color = {
            "severity_low": "green",
            "severity_moderate": "blue",
            "severity_high": "orange",
            "severity_very_high": "red"
        }.get(disease_analysis["severity"], "blue")
        
        st.markdown(
            f"<h3 style='text-align: center; color: {severity_color};'>"
            f"{disease_analysis['severity_score']}/4</h3>"
            f"<p style='text-align: center;'>{translate_text('Severity', language)}</p>",
            unsafe_allow_html=True
        )
    
    # Display spread rate
    spread_text = translate_text(TEXT_CONTENT[disease_analysis["spread_rate"]], language)
    with col2:
        spread_color = {
            "spread_slow": "green",
            "spread_moderate": "blue",
            "spread_fast": "orange",
            "spread_very_fast": "red"
        }.get(disease_analysis["spread_rate"], "blue")
        
        st.markdown(
            f"<h3 style='text-align: center; color: {spread_color};'>"
            f"{disease_analysis['spread_score']}/4</h3>"
            f"<p style='text-align: center;'>{translate_text(TEXT_CONTENT['spread_rate'], language)}</p>",
            unsafe_allow_html=True
        )
    
    # Display affected area percentage
    with col3:
        st.markdown(
            f"<h3 style='text-align: center;'>"
            f"{disease_analysis['disease_percentage']:.1f}%</h3>"
            f"<p style='text-align: center;'>{translate_text(TEXT_CONTENT['affected_area'], language)}</p>",
            unsafe_allow_html=True
        )
    
    # Display recovery potential
    with col4:
        recovery_color = {
            "High": "green",
            "Moderate": "orange",
            "Low": "red"
        }.get(disease_analysis["recovery_potential"], "blue")
        
        st.markdown(
            f"<h3 style='text-align: center; color: {recovery_color};'>"
            f"{disease_analysis['recovery_potential']}</h3>"
            f"<p style='text-align: center;'>{translate_text(TEXT_CONTENT['recovery_potential'], language)}</p>",
            unsafe_allow_html=True
        )
    
    # Yield impact prediction
    st.write("### " + translate_text(TEXT_CONTENT["crop_loss"], language))
    
    # Create 3 columns for yield metrics
    col1, col2, col3 = st.columns(3)
    
    # Display estimated crop loss
    with col1:
        loss_color = "red" if yield_prediction["adjusted_loss"] > 30 else ("orange" if yield_prediction["adjusted_loss"] > 15 else "blue")
        
        st.markdown(
            f"<h3 style='text-align: center; color: {loss_color};'>"
            f"{yield_prediction['adjusted_loss']:.1f}%</h3>"
            f"<p style='text-align: center;'>{translate_text(TEXT_CONTENT['yield_loss'], language)}</p>",
            unsafe_allow_html=True
        )
    
    # Display economic impact
    with col2:
        st.markdown(
            f"<h3 style='text-align: center;'>"
            f"${yield_prediction['economic_impact']:.0f}/ha</h3>"
            f"<p style='text-align: center;'>{translate_text(TEXT_CONTENT['economic_impact'], language)}</p>",
            unsafe_allow_html=True
        )
    
    # Display potential recovery with treatment
    with col3:
        st.markdown(
            f"<h3 style='text-align: center; color: green;'>"
            f"{yield_prediction['recovery_percentage']}%</h3>"
            f"<p style='text-align: center;'>{translate_text(TEXT_CONTENT['yield_recovery'], language)}</p>",
            unsafe_allow_html=True
        )
    
    # Timeline and recommendations
    st.info(yield_prediction["treatment_timeline"])
    st.warning(yield_prediction["untreated_outcome"])
    
    # Confidence level
    st.write(f"**{translate_text(TEXT_CONTENT['analysis_confidence'], language)}:** {yield_prediction['confidence']}%")
    if yield_prediction["confidence_factors"]:
        st.write(f"**{translate_text(TEXT_CONTENT['confidence_factors'], language)}:** {', '.join(yield_prediction['confidence_factors'])}")

def main():
    try:
        # Get selected language from session state or set default
        if 'language' not in st.session_state:
            st.session_state.language = 'en'
        
        # Page config with translated title (with error handling)
        try:
            page_title = translate_text(TEXT_CONTENT["page_title"], st.session_state.language)
        except Exception:
            page_title = TEXT_CONTENT["page_title"]  # Fallback to English
        
        st.set_page_config(page_title=page_title, page_icon="🌾", layout="wide")

        # Sidebar with error handling
        try:
            # Add language selection to sidebar
            st.sidebar.title("Settings")
            selected_language = st.sidebar.selectbox(
                "Language / भाषा / ভাষা",
                options=list(INDIAN_LANGUAGES.keys()),
                format_func=lambda x: x
            )
            
            # Set session state language
            st.session_state.language = INDIAN_LANGUAGES[selected_language]
            
            # Page selection with translated options
            app_mode = st.sidebar.selectbox(
                translate_text(TEXT_CONTENT["select_page"], st.session_state.language),
                [
                    translate_text(TEXT_CONTENT["home"], st.session_state.language),
                    translate_text(TEXT_CONTENT["disease_recognition"], st.session_state.language),
                    translate_text(TEXT_CONTENT["satellite_monitoring"], st.session_state.language),
                    translate_text(TEXT_CONTENT["climate_analysis"], st.session_state.language)
                ]
            )
        except Exception as e:
            st.error(f"Translation error in page selection: {e}")
            app_mode = "HOME"  # Fallback to English

        # Display header image
        img = Image.open("Diseases.png")
        st.image(img)

        # Main Page
        if app_mode == translate_text(TEXT_CONTENT["home"], st.session_state.language):
            st.markdown(
                f"<h1 style='text-align: center;'>{translate_text(TEXT_CONTENT['app_title'], st.session_state.language)}</h1>",
                unsafe_allow_html=True
            )
            
            features_title = translate_text(TEXT_CONTENT["features_title"], st.session_state.language)
            st.write(f"### {features_title}")
            
            for feature in TEXT_CONTENT["features"]:
                translated_feature = translate_text(feature, st.session_state.language)
                st.write(f"- {translated_feature}")
            

        # Disease Recognition Page
        elif app_mode == translate_text(TEXT_CONTENT["disease_recognition"], st.session_state.language):
            st.header(translate_text(TEXT_CONTENT["app_title"], st.session_state.language))
            
            # Location selection
            st.sidebar.header(translate_text(TEXT_CONTENT["location_settings"], st.session_state.language))
            location_type = st.sidebar.radio(
                translate_text(TEXT_CONTENT["location_type"], st.session_state.language),
                ["Predefined", "Custom"]
            )
            
            if location_type == "Predefined":
                selected_location = st.sidebar.selectbox(
                    translate_text(TEXT_CONTENT["select_location"], st.session_state.language),
                    list(RURAL_LOCATIONS.keys())[:-1]
                )
                latitude, longitude = RURAL_LOCATIONS[selected_location]
            else:
                selected_location = translate_text(TEXT_CONTENT["custom_location"], st.session_state.language)
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    latitude = st.number_input(
                        translate_text(TEXT_CONTENT["latitude"], st.session_state.language),
                        min_value=-90.0, max_value=90.0, value=0.0
                    )
                with col2:
                    longitude = st.number_input(
                        translate_text(TEXT_CONTENT["longitude"], st.session_state.language),
                        min_value=-180.0, max_value=180.0, value=0.0
                    )

            # Collect soil data
            soil_data = collect_soil_data(st.session_state.language)

            # File uploader
            test_image = st.file_uploader(
                translate_text(TEXT_CONTENT["choose_image"], st.session_state.language)
            )

            if test_image:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(translate_text(TEXT_CONTENT["show_image"], st.session_state.language)):
                        st.image(test_image, use_container_width=True)
                
                with col2:
                    predict_button = st.button(
                        translate_text(TEXT_CONTENT["predict_disease"], st.session_state.language)
                    )

                # Initialize session state
                if 'prediction' not in st.session_state:
                    st.session_state.prediction = None

                if predict_button:
                    st.snow()
                    st.write(f"### {translate_text(TEXT_CONTENT["disease_results"], st.session_state.language)}")
                    result_index = model_prediction(test_image)
                    
                    # Your existing class names list
                    class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                                'Tomato___healthy']
                    
                    st.session_state.prediction = class_name[result_index]
                    prediction_text = translate_text(
                        TEXT_CONTENT["model_predicting"],
                        st.session_state.language
                    )
                    st.success(f"{prediction_text} {st.session_state.prediction}")
                    
                    # Add yield impact prediction if the detected plant is diseased (not healthy)
                    if not st.session_state.prediction.endswith("___healthy"):
                        # Display a separator
                        st.markdown("---")
                        
                        # Reset the file pointer to the beginning of the file for reprocessing
                        test_image.seek(0)
                        
                        # Analyze disease severity
                        with st.spinner(translate_text(TEXT_CONTENT["analyzing_severity"], st.session_state.language)):
                            # Save the original image for display
                            test_image.seek(0)
                            
                            # Analyze disease severity
                            disease_analysis = analyze_disease_severity(test_image, st.session_state.prediction)
                            
                            # Reset the file pointer again for future use
                            test_image.seek(0)
                            
                            # Get weather data for the location (already available)
                            weather_data = get_weather_data(latitude, longitude)
                            
                            # Predict yield impact
                            yield_prediction = predict_yield_impact(
                                st.session_state.prediction, 
                                disease_analysis, 
                                weather_data, 
                                soil_data
                            )
                            
                            # Display yield impact prediction
                            display_yield_impact(
                                st.session_state.prediction,
                                test_image,
                                disease_analysis,
                                yield_prediction,
                                st.session_state.language
                            )

                if st.button(
                    translate_text(TEXT_CONTENT["get_analysis"], st.session_state.language),
                    use_container_width=True
                ):
                    if st.session_state.prediction:
                        with st.spinner(translate_text(TEXT_CONTENT["generating_analysis"], st.session_state.language)):
                            # Get weather and analysis
                            weather_data = get_weather_data(latitude, longitude)
                            
                            # Pass soil data to analysis function
                            analysis = get_gemini_analysis(
                                st.session_state.prediction,
                                weather_data,
                                soil_data,
                                st.session_state.language
                            )
                            
                            # Display location
                            location_header = translate_text("Location", st.session_state.language)
                            st.write(f"### {location_header}: {selected_location}")
                            
                            # Display weather information
                            if weather_data:
                                weather_header = translate_text(TEXT_CONTENT["current_weather"], st.session_state.language)
                                st.write(f"### {weather_header}")
                                
                                weather_info = {
                                    "Temperature": f"{weather_data['temperature']}°C",
                                    "Humidity": f"{weather_data['humidity']}%",
                                    "Description": weather_data['description'],
                                    "Wind Speed": f"{weather_data['wind_speed']} m/s"
                                }
                                
                                for key, value in weather_info.items():
                                    translated_key = translate_text(key, st.session_state.language)
                                    st.write(f"{translated_key}: {value}")
                            
                            # Display soil information if available
                            if soil_data:
                                soil_header = translate_text(TEXT_CONTENT["soil_analysis_results"], st.session_state.language)
                                st.write(f"### {soil_header}")
                                
                                soil_info = {
                                    "Soil pH": soil_data['pH'],
                                    "Nitrogen (N)": f"{soil_data['nitrogen']} ppm",
                                    "Phosphorus (P)": f"{soil_data['phosphorus']} ppm",
                                    "Potassium (K)": f"{soil_data['potassium']} ppm",
                                    "Organic Matter": f"{soil_data['organic_matter']}%",
                                    "Soil Texture": soil_data['texture']
                                }
                                
                                for key, value in soil_info.items():
                                    translated_key = translate_text(key, st.session_state.language)
                                    st.write(f"{translated_key}: {value}")
                            else:
                                st.info(translate_text(TEXT_CONTENT["no_soil_data"], st.session_state.language))
                            
                            # Display analysis
                            analysis_header = translate_text(TEXT_CONTENT["detailed_analysis"], st.session_state.language)
                            st.write(f"### {analysis_header}")
                            st.write(analysis)
                            
                            # Download button
                            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                            download_label = translate_text(TEXT_CONTENT["download_report"], st.session_state.language)
                            
                            st.download_button(
                                label=download_label,
                                data=analysis,
                                file_name=f"plant_disease_analysis_{st.session_state.language}_{current_time}.txt",
                                mime="text/plain"
                            )
                    else:
                        st.warning(translate_text(TEXT_CONTENT["predict_first"], st.session_state.language))
                    
        # Satellite Monitoring Page
        elif app_mode == translate_text(TEXT_CONTENT["satellite_monitoring"], st.session_state.language):
            st.header(translate_text(TEXT_CONTENT["satellite_title"], st.session_state.language))
            
            # Location selection
            st.sidebar.header(translate_text(TEXT_CONTENT["location_settings"], st.session_state.language))
            location_type = st.sidebar.radio(
                translate_text(TEXT_CONTENT["location_type"], st.session_state.language),
                ["Predefined", "Custom"]
            )
            
            if location_type == "Predefined":
                selected_location = st.sidebar.selectbox(
                    translate_text(TEXT_CONTENT["select_location"], st.session_state.language),
                    list(RURAL_LOCATIONS.keys())[:-1]
                )
                latitude, longitude = RURAL_LOCATIONS[selected_location]
            else:
                selected_location = translate_text(TEXT_CONTENT["custom_location"], st.session_state.language)
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    latitude = st.number_input(
                        translate_text(TEXT_CONTENT["latitude"], st.session_state.language),
                        min_value=-90.0, max_value=90.0, value=20.0
                    )
                with col2:
                    longitude = st.number_input(
                        translate_text(TEXT_CONTENT["longitude"], st.session_state.language),
                        min_value=-180.0, max_value=180.0, value=78.0
                    )
            
            # Display map for reference
            m = folium.Map(location=[latitude, longitude], zoom_start=12)
            folium.Marker(
                [latitude, longitude],
                popup=selected_location,
                tooltip=selected_location
            ).add_to(m)
            
            # Add a circle to represent the area of interest
            area_size = st.sidebar.slider(
                translate_text(TEXT_CONTENT["area_size"], st.session_state.language),
                min_value=1.0, max_value=25.0, value=4.0, step=0.5
            )
            
            # Rough conversion from km² to meters for circle radius (assuming square area)
            radius = (area_size ** 0.5) * 500  # Half the side length in meters
            folium.Circle(
                radius=radius,
                location=[latitude, longitude],
                color="red",
                fill=True,
            ).add_to(m)
            
            st_folium(m, width=800, height=500)
            
            # Select vegetation index
            index_type = st.sidebar.selectbox(
                translate_text(TEXT_CONTENT["select_index"], st.session_state.language),
                ["NDVI", "EVI", "NDMI"],
                format_func=lambda x: {
                    "NDVI": translate_text(TEXT_CONTENT["ndvi"], st.session_state.language),
                    "EVI": translate_text(TEXT_CONTENT["evi"], st.session_state.language),
                    "NDMI": translate_text(TEXT_CONTENT["moisture"], st.session_state.language)
                }.get(x, x)
            )
            
            # Time period selection
            time_period = st.sidebar.selectbox(
                translate_text(TEXT_CONTENT["time_period"], st.session_state.language),
                ["last_month", "last_3_months", "last_year", "custom_date"],
                format_func=lambda x: {
                    "last_month": translate_text(TEXT_CONTENT["last_month"], st.session_state.language),
                    "last_3_months": translate_text(TEXT_CONTENT["last_3_months"], st.session_state.language),
                    "last_year": translate_text(TEXT_CONTENT["last_year"], st.session_state.language),
                    "custom_date": translate_text(TEXT_CONTENT["custom_date"], st.session_state.language)
                }.get(x, x)
            )
            
            # Handle date selection based on time period
            end_date = datetime.datetime.now()
            if time_period == "last_month":
                start_date = end_date - timedelta(days=30)
            elif time_period == "last_3_months":
                start_date = end_date - timedelta(days=90)
            elif time_period == "last_year":
                start_date = end_date - timedelta(days=365)
            else:  # Custom date
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    start_date = st.date_input(
                        translate_text(TEXT_CONTENT["start_date"], st.session_state.language),
                        value=end_date - timedelta(days=30)
                    )
                with col2:
                    end_date = st.date_input(
                        translate_text(TEXT_CONTENT["end_date"], st.session_state.language),
                        value=end_date
                    )
                
                # Convert to datetime
                start_date = datetime.datetime.combine(start_date, datetime.time.min)
                end_date = datetime.datetime.combine(end_date, datetime.time.max)
            
            # Fetch satellite data button
            if st.sidebar.button(translate_text(TEXT_CONTENT["fetch_imagery"], st.session_state.language)):
                with st.spinner(translate_text(TEXT_CONTENT["analyzing_satellite"], st.session_state.language)):
                    # Store data in session state
                    st.session_state.satellite_data = fetch_satellite_data(
                        latitude, 
                        longitude, 
                        area_size=area_size,
                        index_type=index_type,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Display results header
                    st.subheader(translate_text(TEXT_CONTENT["satellite_results"], st.session_state.language))
                    
                    # Display location
                    location_header = translate_text("Location", st.session_state.language)
                    st.write(f"### {location_header}: {selected_location}")
                    
                    # Visualize the data
                    if 'satellite_data' in st.session_state and st.session_state.satellite_data:
                        visualize_satellite_data(st.session_state.satellite_data, st.session_state.language)
                        
                        # Get AI analysis of satellite data
                        if st.session_state.satellite_data.get('has_data'):
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            
                            prompt = f"""
                            Analyze satellite imagery ({index_type} index) for agricultural land with the following information:
                            
                            Location: {selected_location}
                            Area Size: {area_size} square kilometers
                            Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
                            
                            Provide a brief agricultural analysis focusing on:
                            1. Vegetation health assessment based on {index_type} index
                            2. Recommendations for agricultural management
                            3. Potential concerns visible from satellite imagery
                            4. Comparison of this area with typical agricultural land
                            
                            Format the response with bullet points for each section.
                            """
                            
                            try:
                                response = model.generate_content(prompt)
                                analysis = response.text
                                
                                # Translate analysis if not English
                                if st.session_state.language != 'en':
                                    analysis = translate_text(analysis, st.session_state.language)
                                
                                analysis_header = translate_text(TEXT_CONTENT["satellite_analysis"], st.session_state.language)
                                st.write(f"### {analysis_header}")
                                st.write(analysis)
                                
                                # Download button
                                current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                                download_label = translate_text(TEXT_CONTENT["download_satellite"], st.session_state.language)
                                
                                report_content = f"""
                                Satellite Analysis Report
                                ------------------------
                                Location: {selected_location}
                                Coordinates: {latitude}, {longitude}
                                Area Size: {area_size} km²
                                Index Type: {index_type}
                                Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
                                
                                {analysis}
                                """
                                
                                st.download_button(
                                    label=download_label,
                                    data=report_content,
                                    file_name=f"satellite_analysis_{st.session_state.language}_{current_time}.txt",
                                    mime="text/plain"
                                )
                                
                            except Exception as e:
                                st.error(f"Error generating analysis: {str(e)}")
                    
        # Climate Data Analysis Page
        elif app_mode == translate_text(TEXT_CONTENT["climate_analysis"], st.session_state.language):
            st.header(translate_text(TEXT_CONTENT["climate_title"], st.session_state.language))
            st.write(translate_text(TEXT_CONTENT["climate_subtitle"], st.session_state.language))
            
            # Location selection
            st.sidebar.header(translate_text(TEXT_CONTENT["location_settings"], st.session_state.language))
            location_type = st.sidebar.radio(
                translate_text(TEXT_CONTENT["location_type"], st.session_state.language),
                ["Predefined", "Custom"]
            )
            
            if location_type == "Predefined":
                selected_location = st.sidebar.selectbox(
                    translate_text(TEXT_CONTENT["select_location"], st.session_state.language),
                    list(RURAL_LOCATIONS.keys())[:-1]
                )
                latitude, longitude = RURAL_LOCATIONS[selected_location]
            else:
                selected_location = translate_text(TEXT_CONTENT["custom_location"], st.session_state.language)
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    latitude = st.number_input(
                        translate_text(TEXT_CONTENT["latitude"], st.session_state.language),
                        min_value=-90.0, max_value=90.0, value=20.0
                    )
                with col2:
                    longitude = st.number_input(
                        translate_text(TEXT_CONTENT["longitude"], st.session_state.language),
                        min_value=-180.0, max_value=180.0, value=78.0
                    )
            
            # Display map for location reference
            m = folium.Map(location=[latitude, longitude], zoom_start=10)
            folium.Marker(
                [latitude, longitude],
                popup=selected_location,
                tooltip=selected_location
            ).add_to(m)
            
            st_folium(m, width=800, height=400)
            
            # Analysis parameters
            st.sidebar.header(translate_text(TEXT_CONTENT["climate_period"], st.session_state.language))
            
            years_back = st.sidebar.radio(
                translate_text("Time Period", st.session_state.language),
                [5, 10, 20],
                format_func=lambda x: {
                    5: translate_text(TEXT_CONTENT["last_5_years"], st.session_state.language),
                    10: translate_text(TEXT_CONTENT["last_10_years"], st.session_state.language),
                    20: translate_text(TEXT_CONTENT["last_20_years"], st.session_state.language)
                }.get(x, str(x))
            )
            
            data_type = st.sidebar.radio(
                translate_text(TEXT_CONTENT["data_type"], st.session_state.language),
                ["temperature", "precipitation", "humidity"],
                format_func=lambda x: {
                    "temperature": translate_text(TEXT_CONTENT["temperature"], st.session_state.language),
                    "precipitation": translate_text(TEXT_CONTENT["precipitation"], st.session_state.language),
                    "humidity": translate_text(TEXT_CONTENT["humidity"], st.session_state.language)
                }.get(x, x)
            )
            
            # Analysis button
            if st.sidebar.button(translate_text(TEXT_CONTENT["analyze_climate"], st.session_state.language)):
                with st.spinner(translate_text(TEXT_CONTENT["analyzing_climate"], st.session_state.language)):
                    # Fetch climate data
                    climate_data = fetch_historical_climate_data(latitude, longitude, years_back=years_back)
                    
                    if climate_data is not None and not climate_data.empty:
                        # Store in session state
                        st.session_state.climate_data = climate_data
                        
                        # Analyze data
                        analysis_results = analyze_climate_data(
                            climate_data,
                            data_type=data_type,
                            language=st.session_state.language
                        )
                        
                        # Store analysis in session state
                        st.session_state.climate_analysis = analysis_results
                        
                        # Display results header
                        st.subheader(translate_text(TEXT_CONTENT["climate_results"], st.session_state.language))
                        
                        # Display location
                        location_header = translate_text("Location", st.session_state.language)
                        st.write(f"### {location_header}: {selected_location}")
                        
                        # Display period
                        period_text = translate_text("Analysis Period", st.session_state.language)
                        period_value = translate_text(f"Last {years_back} Years", st.session_state.language)
                        st.write(f"**{period_text}:** {period_value}")
                        
                        # Display analysis
                        display_climate_analysis(analysis_results, data_type, st.session_state.language)
                        
                        # Generate and provide report for download
                        if analysis_results and analysis_results.get('has_data'):
                            # Create report content
                            report_content = f"""
                            Historical Climate Analysis Report
                            ---------------------------------
                            Location: {selected_location}
                            Coordinates: {latitude}, {longitude}
                            Analysis Period: Last {years_back} years
                            Data Type: {data_type}
                            
                            Key Insights:
                            """
                            
                            if data_type == "temperature":
                                report_content += f"""
                                Average Temperature: {analysis_results['insights'].get('avg_temp', 0):.1f}°C
                                Maximum Temperature: {analysis_results['insights'].get('max_temp', 0):.1f}°C
                                Minimum Temperature: {analysis_results['insights'].get('min_temp', 0):.1f}°C
                                Temperature Trend: {analysis_results['insights'].get('temp_trend', 0):.2f}°C/year
                                """
                                
                                if 'extreme_hot_days' in analysis_results:
                                    report_content += f"""
                                    Extreme Hot Days: {analysis_results['extreme_hot_days']}
                                    Extreme Cold Days: {analysis_results['extreme_cold_days']}
                                    """
                            
                            elif data_type == "precipitation":
                                report_content += f"""
                                Total Precipitation: {analysis_results['insights'].get('total_precip', 0):.1f} mm
                                Average Daily Precipitation: {analysis_results['insights'].get('avg_precip', 0):.2f} mm
                                Maximum Daily Precipitation: {analysis_results['insights'].get('max_precip', 0):.1f} mm on {analysis_results['insights'].get('max_precip_day', '')}
                                """
                            
                            elif data_type == "humidity":
                                report_content += f"""
                                Average Maximum Humidity: {analysis_results['insights'].get('avg_humidity_max', 0):.1f}%
                                Average Minimum Humidity: {analysis_results['insights'].get('avg_humidity_min', 0):.1f}%
                                """
                            
                            if 'disease_risk' in analysis_results:
                                high_risk = analysis_results['disease_risk']['high_risk_percent']
                                moderate_risk = analysis_results['disease_risk']['moderate_risk_percent']
                                
                                report_content += f"""
                                
                                Plant Disease Risk Assessment:
                                High Disease Risk Days: {analysis_results['disease_risk']['high_risk_days']} ({high_risk:.1f}%)
                                Moderate Disease Risk Days: {analysis_results['disease_risk']['moderate_risk_days']} ({moderate_risk:.1f}%)
                                """
                                
                                if high_risk > 30:
                                    report_content += "\nThe climate in this region shows frequent high-risk conditions for plant diseases."
                                elif high_risk > 15:
                                    report_content += "\nThe climate shows moderate disease pressure with seasonal high-risk periods."
                                else:
                                    report_content += "\nThe climate generally has low disease pressure, with few high-risk days."
                            
                            # Add recommendations based on climate data
                            report_content += """
                            
                            Agricultural Recommendations:
                            """
                            
                            if data_type == "temperature":
                                if analysis_results['insights'].get('temp_trend', 0) > 0.03:
                                    report_content += "\n- Consider heat-tolerant crop varieties as temperatures are rising significantly."
                                elif analysis_results['insights'].get('temp_trend', 0) < -0.03:
                                    report_content += "\n- Consider cold-tolerant crop varieties as temperatures are decreasing."
                                
                                if 'extreme_hot_days' in analysis_results and analysis_results['extreme_hot_days'] > 10:
                                    report_content += "\n- Implement shade structures or row covers to protect crops during extreme heat events."
                            
                            elif data_type == "precipitation":
                                if analysis_results['insights'].get('avg_precip', 0) < 1.0:
                                    report_content += "\n- Consider drought-resistant crops or implement irrigation systems due to low rainfall."
                                elif analysis_results['insights'].get('avg_precip', 0) > 5.0:
                                    report_content += "\n- Improve drainage systems and consider raised beds due to high rainfall."
                            
                            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Fix: Use datetime.datetime.now()
                            download_label = translate_text(TEXT_CONTENT["download_climate"], st.session_state.language)
                            
                            st.download_button(
                                label=download_label,
                                data=report_content,
                                file_name=f"climate_analysis_{st.session_state.language}_{current_time}.txt",
                                mime="text/plain"
                            )
                    else:
                        st.error(translate_text(TEXT_CONTENT["no_climate_data"], st.session_state.language))
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Falling back to English interface")
        st.session_state.language = 'en'
                
if __name__ == "__main__":
    main()