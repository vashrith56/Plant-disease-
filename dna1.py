import streamlit as st
import random
import json
from datasets import load_dataset
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import altair as alt
import requests
from PIL import Image
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# Try to import streamlit_lottie, use fallback if not available
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False
    # Define a placeholder function
    def st_lottie(*args, **kwargs):
        st.warning("Lottie animations unavailable. Install with: pip install streamlit-lottie")
        st.image("https://via.placeholder.com/200x100?text=DNA+Animation", width=200)

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="DNA Encryption Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'current_view' not in st.session_state:
    st.session_state.current_view = "encrypt"
if 'animation_displayed' not in st.session_state:
    st.session_state.animation_displayed = False

# DNA encoding dictionary
DNA_TO_BINARY = {'A': '00', 'T': '01', 'C': '10', 'G': '11'}
FUNCTIONAL_MAPPING = {
    'Mutated': '101',
    'Crossover': '110',
    'Reshape': '111'
}

# Reverse mapping for decryption
BINARY_TO_DNA = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}

# Color themes
THEMES = {
    "light": {
        "bg_color": "#f0f2f6",
        "card_bg": "#ffffff",
        "primary": "#4C8BF5",
        "secondary": "#FF6B6B",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "danger": "#F44336",
        "text": "#2c3e50",
        "subtitle": "#34495e",
        "mutated": "#FF6B6B",
        "crossover": "#4ECDC4",
        "reshape": "#45B7D1"
    },
    "dark": {
        "bg_color": "#121212",
        "card_bg": "#1E1E1E",
        "primary": "#BB86FC",
        "secondary": "#03DAC6",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "danger": "#CF6679",
        "text": "#FFFFFF",
        "subtitle": "#BBBBBB",
        "mutated": "#CF6679",
        "crossover": "#03DAC6",
        "reshape": "#BB86FC"
    }
}

# Helper functions for UI
def get_theme():
    return THEMES[st.session_state.theme]

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def load_lottie_url(url: str):
    """Load a Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def get_dna_animation():
    """Get DNA animation URL based on theme"""
    if st.session_state.theme == "light":
        return "https://assets5.lottiefiles.com/packages/lf20_UgZWvP.json"
    else:
        return "https://assets1.lottiefiles.com/packages/lf20_Ab9KPl.json"

def get_img_as_base64(file):
    """Convert image to base64 string"""
    img = Image.open(file)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Apply custom CSS based on theme
def apply_custom_css():
    theme = get_theme()
    
    # Base64 encoded background patterns
    bg_pattern = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH4wEDEwMyVi2wQwAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAC0klEQVRo3u2aPW/UQBCGn12fAykCUiAUFJFIJCpS8BGQEpIiogkaJCoKxC8gHT8BfgdFGrq0SBQUFEgUFJFIJAoQj3TJ3t5uMRuyh8/ni7N3jpGsXT/7zszOzr53jvwfJrkY4/rFc9X+jeuXipjHjVUZZYn1xsBQVZdUtc30elDPryW5Mjs8kD8z21W1AbwGLsesW0ddebGqTYDvQFtVP03Lu9BmgMxFVZuBwABUVZCg4eKzYub1WdWHwfezwPWZn1rgC3AbeA+8BT6KyJdpBkhK9wPJP1TVy6p6EWioahf4Abz0OvsI2JWRA5Hp5nuquge29oEVEdlrlJTGNLIJrIf2DeBGrQCp6grQcvBaFZE3tQNE5BXwycH6YlvVk1F1DdFI7XJ6FrEs69ewrnlZQ1UngTmgH7iCvrfdBLpAG9hW1Q0R2TlJIPPAHaADLABzHlaxCozieDbKKhkHQCbCdKOCdkPtZ4GHodJbwFKZA7lS3lHVoaq+A24B91T1UcnGzYx4fHd/ZWaWQFpAT0QeA70wjkeq+jBw/Dsi8uJvgXhZqTFE90XkSRRFhbYQXMPSuHKBeJIewFZGm0cZbZuhn/UygQTD0jgQP/9aMJlG1RK1tL7JMVV9Er7PiMjXQwMRke+xB7RqRcAsIHulA6lajVHlQB4DD4BrwPJJTNFP1EZG9xXQqCMQcfAXpbr2rEP5RzQR9TyBq5NhDo5fHKB9x1+WXAERkaGIPDvSIeeCNlS1AZw7hGOvgOfAKbvGy5X4U6KqOrKRiOwCD4Hd2BYcRjbw99Gt03wDl3NPAsR9CpnLCVznnccGrj6NLjsQOCag2rXJIHBNiMhBLYGIyOfI7I3VDjIJ3IXBMlrbnJkDxdK5Jxm4Oq7I1hMErh7D8gx1jFxJWU+Jq2CRuNzANe+BL3mNK7wnPT09f2y2P7aqXNkv5EpcmZgAAAAASUVORK5CYII="
    
    st.markdown(f"""
        <style>
        /* Base styles */
        .main {{
            background-color: {theme["bg_color"]};
            background-image: url("{bg_pattern}");
            background-repeat: repeat;
            padding: 20px;
            color: {theme["text"]};
        }}
        
        .stButton>button {{
            background-color: {theme["primary"]};
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .stTextInput>div>div>input {{
            border-radius: 8px;
            padding: 12px;
            border: 1px solid {theme["primary"] if st.session_state.theme == "light" else "#333"};
            background-color: {theme["card_bg"]};
            color: {theme["text"]};
        }}
        
        .stRadio>div, .stCheckbox>div {{
            background-color: {theme["card_bg"]};
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        
        .stRadio>div:hover, .stCheckbox>div:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        
        /* Cards and containers */
        .card {{
            background-color: {theme["card_bg"]};
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,{0.15 if st.session_state.theme == "light" else 0.3});
            margin-bottom: 24px;
            transition: all 0.3s ease;
        }}
        
        .card:hover {{
            box-shadow: 0 6px 15px rgba(0,0,0,{0.2 if st.session_state.theme == "light" else 0.4});
        }}
        
        /* Cards with different accents */
        .card-primary {{
            border-top: 4px solid {theme["primary"]};
        }}
        
        .card-secondary {{
            border-top: 4px solid {theme["secondary"]};
        }}
        
        .card-success {{
            border-top: 4px solid {theme["success"]};
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            color: {theme["text"]};
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            background: linear-gradient(90deg, {theme["primary"]}, {theme["secondary"]});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }}
        
        h2 {{
            font-size: 1.8rem;
            color: {theme["text"]};
            margin-bottom: 1rem;
        }}
        
        h3 {{
            font-size: 1.4rem;
            color: {theme["subtitle"]};
            font-weight: 500;
        }}
        
        /* DNA sequence display */
        .dna-sequence {{
            font-family: 'Courier New', monospace;
            background-color: {theme["bg_color"]};
            color: {theme["text"]};
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: nowrap;
            border: 1px solid {theme["primary"] if st.session_state.theme == "light" else "#333"};
        }}
        
        .dna-sequence .A {{ color: #FF5733; }}
        .dna-sequence .T {{ color: #33FF57; }}
        .dna-sequence .C {{ color: #3357FF; }}
        .dna-sequence .G {{ color: #F033FF; }}
        
        /* Sidebar styling */
        .css-1d391kg, .css-1lcbmhc {{
            background-color: {theme["card_bg"]};
        }}
        
        /* Status indicators */
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-success {{
            background-color: {theme["success"]};
        }}
        
        .status-warning {{
            background-color: {theme["warning"]};
        }}
        
        .status-error {{
            background-color: {theme["danger"]};
        }}
        
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 8px;
        }}
        
        .badge-primary {{
            background-color: {theme["primary"]};
            color: white;
        }}
        
        .badge-secondary {{
            background-color: {theme["secondary"]};
            color: white;
        }}
        
        .badge-mutated {{
            background-color: {theme["mutated"]};
            color: white;
        }}
        
        .badge-crossover {{
            background-color: {theme["crossover"]};
            color: white;
        }}
        
        .badge-reshape {{
            background-color: {theme["reshape"]};
            color: white;
        }}
        
        /* Navigation tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: {theme["card_bg"]};
            border-radius: 8px 8px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {theme["primary"]};
            color: white;
        }}
        
        /* Dataframe styling */
        .dataframe {{
            font-family: 'Roboto', sans-serif;
            width: 100%;
            border-collapse: collapse;
        }}
        
        .dataframe th {{
            background-color: {theme["primary"]};
            color: white;
            padding: 12px;
            text-align: left;
        }}
        
        .dataframe td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        
        .dataframe tr:hover {{
            background-color: {theme["bg_color"]};
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid {theme["primary"] if st.session_state.theme == "light" else "#333"};
            font-size: 0.9rem;
            color: {theme["subtitle"]};
        }}
        
        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .animate-fade-in {{
            animation: fadeIn 0.6s ease-in;
        }}
        
        @keyframes slideInRight {{
            from {{ transform: translateX(30px); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        
        .animate-slide-in {{
            animation: slideInRight 0.5s ease-out;
        }}
        
        /* Loading spinner */
        .loading-spinner {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }}
        
        /* Switch container - for theme toggle */
        .switch-container {{
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .switch-label {{
            margin-right: 10px;
            font-weight: 500;
        }}
        
        /* Highlight sequences */
        .highlight-A {{
            color: #FF5733;
            font-weight: bold;
        }}
        
        .highlight-T {{
            color: #33FF57;
            font-weight: bold;
        }}
        
        .highlight-C {{
            color: #3357FF;
            font-weight: bold;
        }}
        
        .highlight-G {{
            color: #F033FF;
            font-weight: bold;
        }}
        
        /* Dashboard metrics */
        .metric-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-top: 20px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            flex: 1;
            min-width: 200px;
            padding: 16px;
            background-color: {theme["card_bg"]};
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 600;
            color: {theme["primary"]};
            margin: 10px 0;
        }}
        
        .metric-title {{
            font-size: 1rem;
            color: {theme["subtitle"]};
            font-weight: 500;
        }}
        
        /* Progress bar */
        .progress-container {{
            width: 100%;
            height: 12px;
            background-color: {theme["bg_color"]};
            border-radius: 6px;
            margin: 10px 0;
            overflow: hidden;
        }}
        
        .progress-bar {{
            height: 100%;
            background: linear-gradient(90deg, {theme["primary"]}, {theme["secondary"]});
            border-radius: 6px;
            transition: width 0.5s ease;
        }}
        </style>
    """, unsafe_allow_html=True)

# Functions to get available datasets and load data
def get_available_datasets():
    """Returns a list of available genomic datasets from Hugging Face"""
    return [
        # Featured datasets first
        "LysandreJik/canonical-human-genes-sequences", 
        "bigcode/fastaDNA",
        # Other datasets
        "dnagpt/human_genome_GCF_009914755.1",
        "dnagpt/human_genome_GCF_000001405.40",
        "soleimanian/dna_test"
    ]

def validate_dataset(dataset_name):
    """Verify that a dataset exists and is accessible"""
    try:
        # Just attempt to load dataset info without downloading full dataset
        from datasets import get_dataset_config_names
        get_dataset_config_names(dataset_name)
        return True
    except Exception:
        return False

def get_verified_datasets():
    """Returns only datasets that can be verified as accessible"""
    all_datasets = get_available_datasets()
    verified = []
    
    # Make sure our featured datasets are always included regardless of validation
    featured_datasets = ["LysandreJik/canonical-human-genes-sequences", "bigcode/fastaDNA"]
    
    for dataset in all_datasets:
        if dataset in featured_datasets or validate_dataset(dataset):
            verified.append(dataset)
    
    # Always provide at least one fallback option
    if not verified:
        verified = ["dnagpt/human_genome_GCF_009914755.1"]
        
    return verified

def load_genome_sample(dataset_name):
    """Load a sample from the selected dataset"""
    try:
        with st.spinner(f"Loading dataset {dataset_name}..."):
            # First check if dataset exists
            if not validate_dataset(dataset_name):
                st.warning(f"Dataset {dataset_name} is not accessible. Using fallback sequence.")
                return "ATCGATCGATCGATCGATCG"
            
            try:
                dataset = load_dataset(dataset_name, split="train")
                st.info(f"Dataset loaded successfully.")
            except Exception as e:
                st.warning(f"Error loading dataset: {str(e)}. Using fallback sequence.")
                return "ATCGATCGATCGATCGATCG"
            
            # Dataset-specific handling
            if dataset_name == "LysandreJik/canonical-human-genes-sequences":
                # This dataset has sequences in the 'sequence' column
                if 'sequence' in dataset[0]:
                    return dataset[0]['sequence'][:100]
                else:
                    return "ATCGATCGATCGATCGATCG"
            elif dataset_name == "bigcode/fastaDNA":
                # This dataset has sequences in the 'content' column
                if 'content' in dataset[0]:
                    return ''.join(c for c in dataset[0]['content'].upper() if c in 'ATCG')[:100]
                else:
                    return "ATCGATCGATCGATCGATCG"
            
            # General handling for other datasets
            if 'sequence' in dataset.column_names:
                return dataset[0]['sequence'][:100]
            elif 'dna' in dataset.column_names:
                return dataset[0]['dna'][:100]
            elif 'genome' in dataset.column_names:
                return dataset[0]['genome'][:100]
            else:
                # Try to find any string column that looks like DNA
                for col in dataset.column_names:
                    # Check if it's a string and contains DNA bases
                    if isinstance(dataset[0][col], str) and any(base in dataset[0][col].upper() for base in 'ATCG'):
                        sample = dataset[0][col][:100]
                        # Filter to keep only ATCG characters
                        sample = ''.join(c for c in sample.upper() if c in 'ATCG')
                        if sample:
                            return sample
                
                # If we still can't find it, try to look inside nested structures
                if len(dataset.column_names) == 1:
                    col = dataset.column_names[0]
                    if isinstance(dataset[0][col], (str, list, dict)):
                        if isinstance(dataset[0][col], str):
                            sample = dataset[0][col][:100]
                            sample = ''.join(c for c in sample.upper() if c in 'ATCG')
                            if sample:
                                return sample
                        elif isinstance(dataset[0][col], list) and len(dataset[0][col]) > 0:
                            sample = str(dataset[0][col][0])[:100]
                            sample = ''.join(c for c in sample.upper() if c in 'ATCG')
                            if sample:
                                return sample
                        elif isinstance(dataset[0][col], dict) and 'sequence' in dataset[0][col]:
                            sample = dataset[0][col]['sequence'][:100]
                            sample = ''.join(c for c in sample.upper() if c in 'ATCG')
                            if sample:
                                return sample
                
                st.warning(f"Could not find DNA sequence in dataset {dataset_name}. Using fallback sequence.")
                return "ATCGATCGATCGATCGATCG"
    except Exception as e:
        st.error(f"Error loading dataset {dataset_name}: {e}")
        return "ATCGATCGATCGATCGATCG"

def load_full_dataset(dataset_name, max_samples=50):
    """Load and return multiple sequences from the dataset"""
    try:
        with st.spinner(f"Loading dataset {dataset_name}..."):
            # Check if dataset exists
            if not validate_dataset(dataset_name):
                st.warning(f"Dataset {dataset_name} is not accessible. Using fallback sequence.")
                return [{"id": 0, "preview": "ATCGATCGATCGATCGATCG", "full_sequence": "ATCGATCGATCGATCGATCG", "length": 20}]
                
            dataset = load_dataset(dataset_name, split="train")
            
            # Limit to a reasonable number of samples
            num_samples = min(len(dataset), max_samples)
            
            sequences = []
            for i in range(num_samples):
                sample = dataset[i]
                sequence = None
                
                # Try to find sequence in different standard fields
                if 'sequence' in dataset.column_names:
                    sequence = sample['sequence']
                elif 'dna' in dataset.column_names:
                    sequence = sample['dna']
                elif 'genome' in dataset.column_names:
                    sequence = sample['genome']
                else:
                    # Try to find any string column that looks like DNA
                    for col in dataset.column_names:
                        if isinstance(sample[col], str) and any(base in sample[col].upper() for base in 'ATCG'):
                            sequence = sample[col]
                            break
                    
                    # If still not found, check nested structures
                    if sequence is None and len(dataset.column_names) == 1:
                        col = dataset.column_names[0]
                        if isinstance(sample[col], (str, list, dict)):
                            if isinstance(sample[col], str):
                                sequence = sample[col]
                            elif isinstance(sample[col], list) and len(sample[col]) > 0:
                                sequence = str(sample[col][0])
                            elif isinstance(sample[col], dict) and 'sequence' in sample[col]:
                                sequence = sample[col]['sequence']
                
                if sequence:
                    # Clean the sequence to only contain ATCG
                    cleaned_seq = ''.join(c for c in sequence.upper() if c in 'ATCG')
                    if cleaned_seq:
                        # Calculate complexity metrics
                        a_count = cleaned_seq.count('A')
                        t_count = cleaned_seq.count('T')
                        c_count = cleaned_seq.count('C')
                        g_count = cleaned_seq.count('G')
                        gc_content = ((g_count + c_count) / len(cleaned_seq)) * 100 if len(cleaned_seq) > 0 else 0
                        
                        sequences.append({
                            "id": i,
                            "preview": cleaned_seq[:50] + "..." if len(cleaned_seq) > 50 else cleaned_seq,
                            "full_sequence": cleaned_seq,
                            "length": len(cleaned_seq),
                            "a_count": a_count,
                            "t_count": t_count,
                            "c_count": c_count,
                            "g_count": g_count,
                            "gc_content": gc_content
                        })
            
            if not sequences:
                st.warning(f"Could not find any valid DNA sequences in dataset {dataset_name}.")
                return [{"id": 0, "preview": "ATCGATCGATCGATCGATCG", "full_sequence": "ATCGATCGATCGATCGATCG", "length": 20}]
                
            return sequences
    except Exception as e:
        st.error(f"Error loading dataset {dataset_name}: {e}")
        return [{"id": 0, "preview": "ATCGATCGATCGATCGATCG", "full_sequence": "ATCGATCGATCGATCGATCG", "length": 20}]

def get_dataset_info(dataset_name):
    """Returns information about specific datasets"""
    dataset_info = {
        "LysandreJik/canonical-human-genes-sequences": {
            "description": "A comprehensive collection of canonical human gene sequences",
            "size": "~20,000 sequences",
            "format": "Sequences are provided in the 'sequence' column",
            "source": "Derived from human genome annotation data",
            "usage": "Ideal for gene-level DNA encryption and pattern analysis"
        },
        "bigcode/fastaDNA": {
            "description": "Various DNA sequences in FASTA format",
            "size": "Multiple sequences of varying lengths",
            "format": "Sequences are provided in the 'content' column with FASTA headers in the 'name' column",
            "source": "Compiled from various genomic databases",
            "usage": "Good for testing encryption on diverse sequence types"
        }
    }
    
    return dataset_info.get(dataset_name, {
        "description": "Standard genomic dataset",
        "size": "Varies",
        "format": "Standard format",
        "source": "Hugging Face Hub",
        "usage": "General DNA sequence analysis"
    })

# DNA encoding and decoding functions
def encode_dna_to_binary(dna_sequence):
    """Convert DNA sequence to binary representation"""
    return ''.join(DNA_TO_BINARY[base] for base in dna_sequence.upper())

def binary_to_dna(binary_sequence):
    """Convert binary sequence back to DNA"""
    dna = ""
    for i in range(0, len(binary_sequence), 2):
        if i+1 < len(binary_sequence):
            pair = binary_sequence[i:i+2]
            if pair in BINARY_TO_DNA:
                dna += BINARY_TO_DNA[pair]
    return dna

def format_dna_sequence(sequence, highlight=True):
    """Format DNA sequence with colored base pairs"""
    if not highlight:
        return sequence
        
    formatted = ""
    for base in sequence:
        if base == 'A':
            formatted += f'<span class="highlight-A">{base}</span>'
        elif base == 'T':
            formatted += f'<span class="highlight-T">{base}</span>'
        elif base == 'C':
            formatted += f'<span class="highlight-C">{base}</span>'
        elif base == 'G':
            formatted += f'<span class="highlight-G">{base}</span>'
        else:
            formatted += base
    return formatted

def classify_segment(segment):
    """Classify DNA segment based on patterns"""
    if 'TTT' in segment or 'GGG' in segment:
        return 'Mutated'
    elif 'GT' in segment or 'TG' in segment:
        return 'Crossover'
    else:
        return 'Reshape'

def encrypt_dna(dna_sequence, custom_key=None):
    """Encrypt DNA sequence"""
    segment_length = 5
    segments = [dna_sequence[i:i+segment_length] for i in range(0, len(dna_sequence), segment_length)]
    if len(dna_sequence) % segment_length != 0:
        segments[-1] = dna_sequence[-(len(dna_sequence) % segment_length):]
    
    binary_output = ''
    segment_info = []
    classifications = {'Mutated': 0, 'Crossover': 0, 'Reshape': 0}
    
    # Show encryption progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if custom_key:
        status_text.text("Applying custom key encryption...")
        custom_binary = encode_dna_to_binary(custom_key)
        binary_output = custom_binary
        
        for i, segment in enumerate(segments):
            progress = (i + 1) / len(segments)
            progress_bar.progress(progress)
            status_text.text(f"Processing segment {i+1}/{len(segments)}...")
            time.sleep(0.01)  # Small delay for visual effect
            
            classification = classify_segment(segment)
            classifications[classification] += 1
            segment_info.append({
                'original': segment,
                'classification': classification,
                'binary': encode_dna_to_binary(segment)
            })
    else:
        status_text.text("Applying standard encryption algorithm...")
        for i, segment in enumerate(segments):
            progress = (i + 1) / len(segments)
            progress_bar.progress(progress)
            status_text.text(f"Processing segment {i+1}/{len(segments)}...")
            time.sleep(0.01)  # Small delay for visual effect
            
            binary = encode_dna_to_binary(segment)
            classification = classify_segment(segment)
            classifications[classification] += 1
            encrypted_key = FUNCTIONAL_MAPPING[classification]
            binary_output += encrypted_key
            segment_info.append({
                'original': segment,
                'classification': classification,
                'binary': binary
            })
    
    progress_bar.progress(1.0)
    status_text.text("Encryption complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    total_segments = len(segments)
    percentages = {
        'Mutated': (classifications['Mutated'] / total_segments * 100) if total_segments > 0 else 0,
        'Crossover': (classifications['Crossover'] / total_segments * 100) if total_segments > 0 else 0,
        'Reshape': (classifications['Reshape'] / total_segments * 100) if total_segments > 0 else 0
    }
    
    return binary_output, segment_info, percentages, classifications

def decrypt_dna(binary_output, segment_info):
    """Decrypt DNA sequence"""
    decrypted_sequence = ''
    segment_size = 3
    
    # Show decryption progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if all(char in '01' for char in binary_output):
        status_text.text("Applying decryption algorithm...")
        for i, info in enumerate(segment_info):
            progress = (i + 1) / len(segment_info)
            progress_bar.progress(progress)
            status_text.text(f"Decrypting segment {i+1}/{len(segment_info)}...")
            time.sleep(0.02)  # Small delay for visual effect
            
            start = i * segment_size
            end = start + segment_size
            if binary_output[start:end] == FUNCTIONAL_MAPPING[info['classification']]:
                decrypted_sequence += info['original']
    else:
        status_text.text("Processing custom key...")
        for i, info in enumerate(segment_info):
            progress = (i + 1) / len(segment_info)
            progress_bar.progress(progress)
            status_text.text(f"Recovering segment {i+1}/{len(segment_info)}...")
            time.sleep(0.02)  # Small delay for visual effect
            
            decrypted_sequence += info['original']
    
    progress_bar.progress(1.0)
    status_text.text("Decryption complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return decrypted_sequence

# Visualization functions
def plot_pie_chart(percentages, title):
    """Create pie chart for classification distribution"""
    theme = get_theme()
    labels = ['Mutated', 'Crossover', 'Reshape']
    sizes = [percentages['Mutated'], percentages['Crossover'], percentages['Reshape']]
    colors = [theme["mutated"], theme["crossover"], theme["reshape"]]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=sizes,
        hole=.4,
        marker=dict(colors=colors),
        textinfo='percent+label',
        textfont=dict(color='white', size=14),
        hoverinfo='label+percent+value',
        textposition='inside',
        pull=[0.05, 0.05, 0.05]
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=theme["text"])
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

def plot_nucleotide_distribution(dna_sequence):
    """Create bar chart for nucleotide distribution"""
    theme = get_theme()
    
    # Count nucleotides
    a_count = dna_sequence.count('A')
    t_count = dna_sequence.count('T')
    c_count = dna_sequence.count('C')
    g_count = dna_sequence.count('G')
    
    # Create DataFrame for plotting
    data = pd.DataFrame({
        'Nucleotide': ['A', 'T', 'C', 'G'],
        'Count': [a_count, t_count, c_count, g_count]
    })
    
    # Create Plotly bar chart
    fig = px.bar(
        data, 
        x='Nucleotide', 
        y='Count',
        color='Nucleotide',
        color_discrete_map={
            'A': '#FF5733', 
            'T': '#33FF57',
            'C': '#3357FF',
            'G': '#F033FF'
        },
        text='Count'
    )
    
    fig.update_layout(
        title='Nucleotide Distribution',
        xaxis_title='Nucleotide',
        yaxis_title='Count',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=theme["text"]),
        height=350
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    
    return fig

def plot_segment_classes(segment_info):
    """Create visualization of segment classifications"""
    theme = get_theme()
    
    # Prepare data
    data = []
    for i, info in enumerate(segment_info):
        data.append({
            'Segment': i + 1,
            'Sequence': info['original'],
            'Classification': info['classification']
        })
    
    df = pd.DataFrame(data)
    
    # Create color mapping
    color_map = {
        'Mutated': theme["mutated"],
        'Crossover': theme["crossover"],
        'Reshape': theme["reshape"]
    }
    
    # Create Plotly figure
    fig = px.scatter(
        df, 
        x='Segment', 
        y='Classification',
        color='Classification',
        color_discrete_map=color_map,
        hover_data=['Sequence'],
        size=[len(s) for s in df['Sequence']],
        size_max=20
    )
    
    fig.update_layout(
        title='Segment Classification Map',
        xaxis_title='Segment Position',
        yaxis_title='Classification',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=theme["text"]),
        height=300
    )
    
    return fig

# Display formatted DNA sequence with highlighting
def display_formatted_dna(sequence, title="DNA Sequence"):
    theme = get_theme()
    
    # Create colored HTML for the sequence
    html_sequence = ""
    for base in sequence:
        if base == 'A':
            html_sequence += f'<span class="A">{base}</span>'
        elif base == 'T':
            html_sequence += f'<span class="T">{base}</span>'
        elif base == 'C':
            html_sequence += f'<span class="C">{base}</span>'
        elif base == 'G':
            html_sequence += f'<span class="G">{base}</span>'
        else:
            html_sequence += base
    
    st.markdown(f"**{title}**", unsafe_allow_html=True)
    st.markdown(f'<div class="dna-sequence">{html_sequence}</div>', unsafe_allow_html=True)

# Dashboard components
def show_dna_metrics(dna_sequence):
    """Display metrics for a DNA sequence"""
    theme = get_theme()
    
    # Calculate metrics
    a_count = dna_sequence.count('A')
    t_count = dna_sequence.count('T')
    c_count = dna_sequence.count('C')
    g_count = dna_sequence.count('G')
    gc_content = ((g_count + c_count) / len(dna_sequence)) * 100 if len(dna_sequence) > 0 else 0
    at_to_gc_ratio = (a_count + t_count) / (g_count + c_count) if (g_count + c_count) > 0 else 0
    
    # Display metrics in a nice grid
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    
    # Length metric
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-title">Sequence Length</div>
        <div class="metric-value">{len(dna_sequence)}</div>
        <div>base pairs</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # GC Content metric
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-title">GC Content</div>
        <div class="metric-value">{gc_content:.1f}%</div>
        <div>of sequence</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # AT/GC Ratio metric
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-title">AT/GC Ratio</div>
        <div class="metric-value">{at_to_gc_ratio:.2f}</div>
        <div>balance factor</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Complexity metric (Shannon entropy)
    base_frequencies = [a_count/len(dna_sequence), t_count/len(dna_sequence), 
                       c_count/len(dna_sequence), g_count/len(dna_sequence)]
    entropy = -sum([freq * np.log2(freq) if freq > 0 else 0 for freq in base_frequencies])
    max_entropy = np.log2(4)  # Max possible entropy with 4 bases
    complexity = (entropy / max_entropy) * 100
    
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-title">Complexity Score</div>
        <div class="metric-value">{complexity:.1f}%</div>
        <div>entropy-based</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Base distribution with progress bars
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'''
        <div style="margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span><span class="highlight-A">A</span> Adenine</span>
                <span>{a_count} ({(a_count/len(dna_sequence)*100 if len(dna_sequence) > 0 else 0):.1f}%)</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {(a_count/len(dna_sequence)*100 if len(dna_sequence) > 0 else 0)}%;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div style="margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span><span class="highlight-C">C</span> Cytosine</span>
                <span>{c_count} ({(c_count/len(dna_sequence)*100 if len(dna_sequence) > 0 else 0):.1f}%)</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {(c_count/len(dna_sequence)*100 if len(dna_sequence) > 0 else 0)}%;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

def main():
    # Apply custom CSS based on current theme
    apply_custom_css()
    
    # Sidebar
    with st.sidebar:
        # Theme toggle
        st.markdown('<div class="switch-container">', unsafe_allow_html=True)
        st.markdown(f'<span class="switch-label">🌙 Dark Mode</span>', unsafe_allow_html=True)
        theme_toggle = st.checkbox('Dark Mode', value=(st.session_state.theme == "dark"), key="theme_toggle", label_visibility="hidden")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Update theme based on toggle
        if theme_toggle and st.session_state.theme == "light":
            toggle_theme()
        elif not theme_toggle and st.session_state.theme == "dark":
            toggle_theme()
        
        # Navigation
        st.header("Navigation")
        nav_encrypt = st.button("🔒 Encrypt DNA", use_container_width=True)
        nav_decrypt = st.button("🔓 Decrypt DNA", use_container_width=True)
        nav_dashboard = st.button("📊 Analytics Dashboard", use_container_width=True)
        
        # Update navigation state
        if nav_encrypt:
            st.session_state.current_view = "encrypt"
        elif nav_decrypt:
            st.session_state.current_view = "decrypt"
        elif nav_dashboard:
            st.session_state.current_view = "dashboard"
        
        # Add DNA animation
        if not st.session_state.animation_displayed:
            if LOTTIE_AVAILABLE:
                lottie_dna = load_lottie_url(get_dna_animation())
                if lottie_dna:
                    st_lottie(lottie_dna, height=200, key="dna_animation")
                    st.session_state.animation_displayed = True
            else:
                # Fallback without Lottie animations
                st.image("https://cdn.pixabay.com/photo/2020/02/24/04/25/web-4875183_1280.png", width=200)
                st.session_state.animation_displayed = True
        
        # Information section
        st.markdown("---")
        st.markdown("### About DNA Encryption")
        st.markdown("""
        This tool allows you to:
        - Encrypt DNA sequences using genetic patterns
        - Analyze nucleotide distribution and patterns
        - Visualize genetic data from Hugging Face datasets
        - Apply custom encryption keys
        
        For detailed documentation, visit the Analytics Dashboard.
        """)
        
        # Footer
        st.markdown("---")
        st.markdown("**DNA Encryption Studio**")
        st.markdown("v2.0 | © 2025")
    
    # Main section
    if st.session_state.current_view == "encrypt":
        display_encryption_interface()
    elif st.session_state.current_view == "decrypt":
        display_decryption_interface()
    elif st.session_state.current_view == "dashboard":
        display_dashboard()

def display_encryption_interface():
    """Display the encryption interface"""
    theme = get_theme()
    
    st.title("🧬 DNA Encryption Studio")
    st.markdown("""
    <div class="animate-fade-in">
        <p>Encrypt your DNA sequences using advanced genetic pattern recognition and binary transformation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main container
    with st.container():
        st.markdown('<div class="card card-primary animate-slide-in">', unsafe_allow_html=True)
        st.subheader("🔬 Sequence Input")
        
        # Input method selection
        input_method = st.radio("**Select Input Method**", 
                               ("Enter manually", "Use sample from dataset", "Generate random"),
                               key="input_method",
                               horizontal=True)
        
        dna_sequence = ""
        if input_method == "Enter manually":
            dna_sequence = st.text_input("**DNA Sequence (A, T, C, G only)**", "").upper()
        
        elif input_method == "Use sample from dataset":
            # Dataset selection with enhanced UI and featured datasets section
            available_datasets = get_verified_datasets()
            
            st.markdown("#### 🌟 Featured Datasets")
            featured_cols = st.columns(2)
            
            with featured_cols[0]:
                st.markdown("""
                **Human Genes**
                
                Collection of canonical human gene sequences
                """)
                human_genes_btn = st.button("Use Human Genes", key="human_genes_btn")
                if human_genes_btn:
                    selected_dataset = "LysandreJik/canonical-human-genes-sequences"
                    st.session_state.dataset_select = "LysandreJik/canonical-human-genes-sequences"
            
            with featured_cols[1]:
                st.markdown("""
                **DNA Sequences**
                
                Various DNA sequences in FASTA format
                """)
                dna_seq_btn = st.button("Use DNA Sequences", key="dna_seq_btn")
                if dna_seq_btn:
                    selected_dataset = "bigcode/fastaDNA"
                    st.session_state.dataset_select = "bigcode/fastaDNA"
            
            st.markdown("#### Or Select Another Dataset")
            
            # Dataset selection with enhanced UI
            available_datasets = get_verified_datasets()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_dataset = st.selectbox(
                    "**Select Dataset**",
                    available_datasets,
                    key="dataset_select",
                    help="Choose a DNA dataset from Hugging Face"
                )
            
            with col2:
                browse_btn = st.button("🔍 Browse Dataset", key="browse_dataset", use_container_width=True)
            
            if browse_btn:
                with st.spinner("Loading dataset samples..."):
                    sequences = load_full_dataset(selected_dataset)
                    st.session_state['dataset_sequences'] = sequences
            
            # Display sequence selection interface with improved UI
            if 'dataset_sequences' in st.session_state and st.session_state['dataset_sequences']:
                sequences = st.session_state['dataset_sequences']
                
                # Create tabs for different ways to browse the data
                browse_tab1, browse_tab2 = st.tabs(["📋 Table View", "📊 Visual Explorer"])
                
                with browse_tab1:
                    # Enhanced table with more metrics
                    sequence_df = pd.DataFrame([{
                        "ID": seq["id"],
                        "Length": seq["length"],
                        "GC Content (%)": seq.get("gc_content", 0),
                        "Preview": seq["preview"]
                    } for seq in sequences])
                    
                    st.dataframe(
                        sequence_df, 
                        height=200,
                        column_config={
                            "ID": st.column_config.NumberColumn("ID", format="%d"),
                            "Length": st.column_config.NumberColumn("Length", format="%d bp"),
                            "GC Content (%)": st.column_config.NumberColumn("GC Content (%)", format="%.1f%%"),
                            "Preview": st.column_config.TextColumn("Preview")
                        },
                        hide_index=True
                    )
                
                with browse_tab2:
                    # Visual explorer with charts
                    if len(sequences) > 1:
                        # Create scatter plot of sequence length vs GC content
                        fig = px.scatter(
                            x=[seq["length"] for seq in sequences],
                            y=[seq.get("gc_content", 0) for seq in sequences],
                            color=[seq["id"] for seq in sequences],
                            labels={"x": "Sequence Length (bp)", "y": "GC Content (%)"},
                            title="Sequence Properties",
                            hover_name=[f"Sequence {seq['id']}" for seq in sequences]
                        )
                        
                        fig.update_layout(
                            xaxis_title="Sequence Length (bp)",
                            yaxis_title="GC Content (%)",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough sequences to create a visualization")
                
                # Add a selection dropdown with enhanced styling
                st.markdown('<div class="card" style="padding: 10px;">', unsafe_allow_html=True)
                selected_index = st.selectbox(
                    "**Select a DNA sequence to use**", 
                    range(len(sequences)),
                    format_func=lambda i: f"Sequence {sequences[i]['id']} (Length: {sequences[i]['length']} bp, GC: {sequences[i].get('gc_content', 0):.1f}%)"
                )
                
                if st.button("Use Selected Sequence", key="use_selected", type="primary"):
                    dna_sequence = sequences[selected_index]['full_sequence']
                    st.session_state['dna_sequence'] = dna_sequence
                    
                    # Show preview of the selected sequence
                    st.markdown(f'<div class="dna-sequence" style="max-height: 100px; overflow-y: auto;">{format_dna_sequence(dna_sequence[:100])}{"..." if len(dna_sequence) > 100 else ""}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Get the current sequence from session state
            dna_sequence = st.session_state.get('dna_sequence', '')
            
            if dna_sequence:
                st.success(f"Loaded a DNA sequence with {len(dna_sequence)} bases")
        
        else:  # Generate random
            st.markdown('<div style="background-color: rgba(0,0,0,0.05); padding: 15px; border-radius: 8px;">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                seq_length = st.slider("**Sequence Length**", 10, 500, 50, key="seq_length")
            
            with col2:
                gc_bias = st.slider("**GC Content Bias**", 0.0, 1.0, 0.5, 0.01, key="gc_bias", 
                                  help="Higher values generate sequences with more G and C nucleotides")
            
            with col3:
                if st.button("Generate", key="gen_random", type="primary", use_container_width=True):
                    # Generate with specified GC bias
                    bases = []
                    for _ in range(seq_length):
                        if random.random() < gc_bias:
                            bases.append(random.choice('GC'))
                        else:
                            bases.append(random.choice('AT'))
                    
                    dna_sequence = ''.join(bases)
                    st.session_state['dna_sequence'] = dna_sequence
            
            st.markdown('</div>', unsafe_allow_html=True)
            dna_sequence = st.session_state.get('dna_sequence', '')
        
        # Custom encryption options
        st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.subheader("🔐 Encryption Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            use_custom_key = st.checkbox("**Use Custom Key**", key="use_custom_key")
            custom_key = None
            if use_custom_key:
                custom_key = st.text_input("**Custom Key (ATGC)**", "").upper()
                if custom_key and not all(base in 'ATCG' for base in custom_key):
                    st.error("Invalid key! Use A, T, C, G only.")
                    custom_key = None
        
        with col2:
            segment_visualization = st.checkbox("**Show Segment Visualization**", value=True, key="seg_viz")
            encryption_details = st.checkbox("**Show Detailed Classification**", value=True, key="enc_details")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # End of input card
        
        # Process and display results
        if dna_sequence:
            if not all(base in 'ATCG' for base in dna_sequence.upper()):
                st.error("Invalid sequence! Use A, T, C, G only.")
            else:
                st.markdown('<div class="card card-secondary animate-slide-in" style="margin-top: 20px;">', unsafe_allow_html=True)
                st.subheader("📈 Analysis Results")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Display formatted DNA
                    display_formatted_dna(dna_sequence, "Original Sequence")
                    
                    # Binary encoding
                    binary_encoded = encode_dna_to_binary(dna_sequence)
                    st.markdown(f"**Binary Encoding:**")
                    st.code(binary_encoded, language="")
                
                with col2:
                    # Show nucleotide distribution
                    fig = plot_nucleotide_distribution(dna_sequence)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Encrypt the DNA
                encrypted_binary, segment_info, percentages, classifications = encrypt_dna(dna_sequence, custom_key)
                
                # Display metrics and visualizations
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("### Encryption Result")
                    st.markdown(f"**Encrypted Output:**")
                    st.code(encrypted_binary, language="")
                    
                    # Show encryption metrics
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    
                    # Number of segments
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Segments</div>
                        <div class="metric-value">{len(segment_info)}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Mutated count
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Mutated</div>
                        <div class="metric-value">{classifications['Mutated']}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Crossover count
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Crossover</div>
                        <div class="metric-value">{classifications['Crossover']}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Reshape count
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Reshape</div>
                        <div class="metric-value">{classifications['Reshape']}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    # Show classification pie chart
                    fig = plot_pie_chart(percentages, "Segment Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show segment visualization if enabled
                if segment_visualization:
                    st.markdown("### Segment Visualization")
                    fig = plot_segment_classes(segment_info)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed segment breakdown if enabled
                if encryption_details:
                    st.markdown("### Segment Breakdown")
                    
                    # Create a clean table to display segments
                    segment_data = []
                    for i, info in enumerate(segment_info):
                        segment_data.append({
                            "Position": i + 1,
                            "Segment": info['original'],
                            "Classification": info['classification'],
                            "Binary": info['binary']
                        })
                    
                    segment_df = pd.DataFrame(segment_data)
                    
                    # Display as a styled dataframe
                    st.dataframe(
                        segment_df,
                        column_config={
                            "Position": st.column_config.NumberColumn("Position", width="small"),
                            "Segment": st.column_config.TextColumn("Segment", width="medium"),
                            "Classification": st.column_config.TextColumn("Classification", width="medium"),
                            "Binary": st.column_config.TextColumn("Binary", width="medium")
                        },
                        hide_index=True
                    )
                
                # Prepare download data
                json_data = {
                    'original_sequence': dna_sequence,
                    'binary_encoded': binary_encoded,
                    'encrypted_output': encrypted_binary,
                    'segments': segment_info,
                    'percentages': percentages,
                    'custom_key': custom_key if custom_key else None,
                    'dataset_source': st.session_state.get('dataset_select', None) if input_method == "Use sample from dataset" else None,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                    'encryption_version': '2.0'
                }
                
                # Save or download options
                st.markdown("### Save Your Results")
                col5, col6 = st.columns(2)
                
                with col5:
                    # Download as JSON
                    json_str = json.dumps(json_data, indent=2)
                    st.download_button(
                        label="📥 Download Data (JSON)",
                        data=json_str,
                        file_name="encrypted_dna.json",
                        mime="application/json",
                        key="download_encrypt"
                    )
                
                with col6:
                    # Save to session state for later use
                    if st.button("💾 Save for Decryption", type="primary", key="save_result"):
                        st.session_state['encrypted_data'] = json_data
                        st.success("Data saved! You can now switch to the Decrypt tab.")
                
                st.markdown('</div>', unsafe_allow_html=True)  # End of results card

def display_decryption_interface():
    """Display the decryption interface"""
    theme = get_theme()
    
    st.title("🧬 DNA Decryption Lab")
    st.markdown("""
    <div class="animate-fade-in">
        <p>Decrypt DNA sequences that were encrypted using the DNA Encryption Studio.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card card-primary animate-slide-in">', unsafe_allow_html=True)
        st.subheader("🔑 Decryption Input")
        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📤 Upload File", "💾 Use Saved Data"])
        
        uploaded_file = None
        json_data = None
        
        with tab1:
            uploaded_file = st.file_uploader("**Upload JSON File**", type=['json'], key="upload_file")
            
            if uploaded_file:
                try:
                    json_data = json.load(uploaded_file)
                    st.success(f"File loaded successfully! Contains a DNA sequence with {len(json_data['original_sequence'])} base pairs.")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        with tab2:
            if 'encrypted_data' in st.session_state:
                json_data = st.session_state['encrypted_data']
                st.success(f"Using saved encryption data with {len(json_data['original_sequence'])} base pairs.")
                
                # Show a preview
                if 'original_sequence' in json_data:
                    st.markdown(f"**Preview of encrypted data:**")
                    st.code(json_data['encrypted_output'][:50] + "..." if len(json_data['encrypted_output']) > 50 else json_data['encrypted_output'])
            else:
                st.info("No saved encryption data found. Please use the Encrypt tab first or upload a JSON file.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
                # Process and display decryption results
        if json_data:
            st.markdown('<div class="card card-secondary animate-slide-in" style="margin-top: 20px;">', unsafe_allow_html=True)
            st.subheader("🔍 Decryption Results")
            
            # Display metadata about the encrypted data
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Encryption Metadata")
                metadata_html = f"""
                <div style="background-color: {theme['bg_color']}; padding: 15px; border-radius: 8px;">
                    <p><strong>Encryption Date:</strong> {json_data.get('timestamp', 'Not available')}</p>
                    <p><strong>Version:</strong> {json_data.get('encryption_version', '1.0')}</p>
                """
                
                if json_data.get('dataset_source'):
                    metadata_html += f"<p><strong>Source Dataset:</strong> {json_data['dataset_source']}</p>"
                    
                if json_data.get('custom_key'):
                    metadata_html += f"<p><strong>Custom Key Used:</strong> Yes</p>"
                else:
                    metadata_html += f"<p><strong>Custom Key Used:</strong> No</p>"
                    
                metadata_html += "</div>"
                st.markdown(metadata_html, unsafe_allow_html=True)
            
            with col2:
                # Show encryption statistics
                st.markdown("### Encryption Statistics")
                
                # Create pie chart of segment types
                fig = plot_pie_chart(json_data['percentages'], "Segment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Decrypt the sequence
            decrypted_sequence = decrypt_dna(json_data['encrypted_output'], json_data['segments'])
            
            # Display decryption results
            st.markdown("### Decryption Output")
            col3, col4 = st.columns([2, 1])
            
            with col3:
                # Show the encrypted and decrypted sequences
                st.markdown("**Encrypted Output:**")
                st.code(json_data['encrypted_output'], language="")
                
                st.markdown("**Decrypted Sequence:**")
                display_formatted_dna(decrypted_sequence, "")
            
            with col4:
                # Validation check
                if decrypted_sequence == json_data['original_sequence']:
                    st.markdown("""
                    <div style="background-color: #4CAF50; color: white; padding: 20px; border-radius: 8px; text-align: center;">
                        <h3 style="margin: 0;">✓ Validation Successful</h3>
                        <p>The decrypted sequence matches the original.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #FF9800; color: white; padding: 20px; border-radius: 8px; text-align: center;">
                        <h3 style="margin: 0;">⚠ Validation Warning</h3>
                        <p>The decrypted sequence does not match the original.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show more details about the mismatch
                    if len(decrypted_sequence) != len(json_data['original_sequence']):
                        st.info(f"Length mismatch: Decrypted ({len(decrypted_sequence)}) vs Original ({len(json_data['original_sequence'])})")
                    
                    # Calculate and show the difference percentage
                    mismatch_count = sum(1 for a, b in zip(decrypted_sequence, json_data['original_sequence']) if a != b)
                    match_percentage = (1 - mismatch_count / max(len(decrypted_sequence), len(json_data['original_sequence']))) * 100
                    
                    st.markdown(f"""
                    <div style="margin-top: 15px;">
                        <p>Match accuracy: {match_percentage:.2f}%</p>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {match_percentage}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show sequence comparison
            st.markdown("### Sequence Comparison")
            
            show_comparison = st.checkbox("Show detailed sequence comparison", value=False)
            if show_comparison:
                # Create a visual comparison
                comparison_html = """
                <div style="font-family: monospace; white-space: pre; overflow-x: auto; margin-top: 10px; padding: 15px; 
                     background-color: {theme['bg_color']}; border-radius: 8px;">
                <div style="font-weight: bold; margin-bottom: 5px;">Original:</div>
                """.format(theme=theme)
                
                for i, char in enumerate(json_data['original_sequence']):
                    char_color = ""
                    if i < len(decrypted_sequence) and char != decrypted_sequence[i]:
                        char_color = f'<span style="color: {theme["danger"]};">{char}</span>'
                    else:
                        char_color = f'<span class="highlight-{char}">{char}</span>'
                    comparison_html += char_color
                
                comparison_html += """
                <div style="font-weight: bold; margin-top: 10px; margin-bottom: 5px;">Decrypted:</div>
                """
                
                for i, char in enumerate(decrypted_sequence):
                    char_color = ""
                    if i < len(json_data['original_sequence']) and char != json_data['original_sequence'][i]:
                        char_color = f'<span style="color: {theme["danger"]};">{char}</span>'
                    else:
                        char_color = f'<span class="highlight-{char}">{char}</span>'
                    comparison_html += char_color
                
                comparison_html += "</div>"
                st.markdown(comparison_html, unsafe_allow_html=True)
            
            # Detailed segment analysis
            st.markdown("### Segment Analysis")
            
            # Create a table of segments
            segment_df = pd.DataFrame([{
                "Position": i+1,
                "Original": segment['original'],
                "Classification": segment['classification'],
                "Binary Pattern": FUNCTIONAL_MAPPING[segment['classification']]
            } for i, segment in enumerate(json_data['segments'])])
            
            st.dataframe(
                segment_df,
                height=300,
                column_config={
                    "Position": st.column_config.NumberColumn("Position"),
                    "Original": st.column_config.TextColumn("Original"),
                    "Classification": st.column_config.TextColumn("Classification"),
                    "Binary Pattern": st.column_config.TextColumn("Binary Pattern")
                },
                hide_index=True
            )
            
            # Add nucleotide analysis
            st.markdown("### Nucleotide Analysis")
            
            col5, col6 = st.columns(2)
            
            with col5:
                # Show original sequence nucleotide distribution
                fig_orig = plot_nucleotide_distribution(json_data['original_sequence'])
                fig_orig.update_layout(title="Original Sequence Nucleotides")
                st.plotly_chart(fig_orig, use_container_width=True)
            
            with col6:
                # Show decrypted sequence nucleotide distribution
                fig_decr = plot_nucleotide_distribution(decrypted_sequence)
                fig_decr.update_layout(title="Decrypted Sequence Nucleotides")
                st.plotly_chart(fig_decr, use_container_width=True)
            
            # Download options
            st.markdown("### Export Results")
            
            # Prepare comparison data for download
            comparison_data = {
                "original_sequence": json_data['original_sequence'],
                "decrypted_sequence": decrypted_sequence,
                "match_percentage": match_percentage if 'match_percentage' in locals() else 100.0,
                "decrypt_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "original_metadata": {
                    "timestamp": json_data.get('timestamp', 'Not available'),
                    "dataset_source": json_data.get('dataset_source', None),
                    "encryption_version": json_data.get('encryption_version', '1.0')
                }
            }
            
            col7, col8 = st.columns(2)
            
            with col7:
                # Download decrypted sequence as FASTA
                fasta_header = f">Decrypted_DNA_{time.strftime('%Y%m%d_%H%M%S')}"
                fasta_content = f"{fasta_header}\n{decrypted_sequence}"
                
                st.download_button(
                    label="📥 Download FASTA",
                    data=fasta_content,
                    file_name="decrypted_dna.fasta",
                    mime="text/plain",
                    key="download_fasta"
                )
            
            with col8:
                # Download comparison results as JSON
                comparison_json = json.dumps(comparison_data, indent=2)
                
                st.download_button(
                    label="📥 Download Comparison",
                    data=comparison_json,
                    file_name="decryption_results.json",
                    mime="application/json",
                    key="download_comparison"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)  # End of decryption results card

def display_dashboard():
    """Display analytics dashboard with reference information"""
    theme = get_theme()
    
    st.title("📊 DNA Encryption Analytics")
    st.markdown("""
    <div class="animate-fade-in">
        <p>Explore DNA sequences, encryption metrics, and learn about the encryption methodology.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard tabs
    dashboard_tab1, dashboard_tab2, dashboard_tab3 = st.tabs([
        "📈 Sequence Analyzer", 
        "🧪 Method Explained", 
        "📖 Documentation"
    ])
    
    # Sequence Analyzer Tab
    with dashboard_tab1:
        st.markdown('<div class="card card-primary animate-slide-in">', unsafe_allow_html=True)
        st.subheader("DNA Sequence Analyzer")
        
        # Input sequence for analysis
        analysis_sequence = st.text_area(
            "Enter a DNA sequence to analyze",
            "",
            height=100,
            help="Input a DNA sequence containing A, T, G, C nucleotides"
        ).upper()
        
        # Or use sample data
        col1, col2 = st.columns([3, 1])
        with col1:
            sample_type = st.selectbox(
                "Or select a sample sequence type",
                ["Random Human DNA", "GC-rich Sequence", "AT-rich Sequence", "Repetitive Pattern"]
            )
        
        with col2:
            if st.button("Load Sample"):
                if sample_type == "Random Human DNA":
                    analysis_sequence = "GCTGCAATGATACCGCGAGACCCACGCTCACCGGCTCCAGATTTATCAGCAATAAACCAGCCAGCCGGAA"
                elif sample_type == "GC-rich Sequence":
                    analysis_sequence = "GCGCGCGCGCCGCGGCCCGCGCGCGCCGCCCGCGCGCCCGCGCGCGCGCCGCGCGCGCGCGCCCGCGCGCGCGC"
                elif sample_type == "AT-rich Sequence":
                    analysis_sequence = "AATATATATTATATATATATTATATATATATATAAATATATATATATATATATATATAATATATATATA"
                else:  # Repetitive
                    analysis_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
                
                st.session_state['analysis_sequence'] = analysis_sequence
        
        if 'analysis_sequence' in st.session_state:
            analysis_sequence = st.session_state['analysis_sequence']
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Featured Datasets Examples")

        featured_dataset_type = st.radio(
            "Choose a featured dataset type",
            ["Human Genes (LysandreJik/canonical-human-genes-sequences)", 
             "DNA Sequences (bigcode/fastaDNA)"],
            horizontal=True
        )

        if st.button("Load Example from Selected Dataset"):
            with st.spinner("Loading dataset sample..."):
                if "Human Genes" in featured_dataset_type:
                    sample = load_genome_sample("LysandreJik/canonical-human-genes-sequences")
                    st.session_state['analysis_sequence'] = sample
                    st.success("Loaded sample from Human Genes dataset!")
                else:
                    sample = load_genome_sample("bigcode/fastaDNA")
                    st.session_state['analysis_sequence'] = sample
                    st.success("Loaded sample from DNA Sequences dataset!")
        
        # Analyze the sequence if provided
        if analysis_sequence and all(base in 'ATCG' for base in analysis_sequence):
            st.markdown('<div class="card card-secondary animate-slide-in" style="margin-top: 20px;">', unsafe_allow_html=True)
            st.subheader("Sequence Analysis")
            
            # Sequence metrics
            show_dna_metrics(analysis_sequence)
            
            # Display more advanced analyses
            st.markdown("### Advanced Analysis")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Show segment classification
                segment_length = 5
                segments = [analysis_sequence[i:i+segment_length] for i in range(0, len(analysis_sequence), segment_length)]
                if len(analysis_sequence) % segment_length != 0:
                    segments[-1] = analysis_sequence[-(len(analysis_sequence) % segment_length):]
                
                segment_classes = [classify_segment(segment) for segment in segments]
                class_counts = {
                    'Mutated': segment_classes.count('Mutated'),
                    'Crossover': segment_classes.count('Crossover'),
                    'Reshape': segment_classes.count('Reshape')
                }
                
                # Plot segment classification
                fig = px.bar(
                    x=['Mutated', 'Crossover', 'Reshape'],
                    y=[class_counts['Mutated'], class_counts['Crossover'], class_counts['Reshape']],
                    color=['Mutated', 'Crossover', 'Reshape'],
                    color_discrete_map={
                        'Mutated': theme["mutated"],
                        'Crossover': theme["crossover"],
                        'Reshape': theme["reshape"]
                    },
                    labels={'x': 'Classification', 'y': 'Count'},
                    title='Segment Classifications'
                )
                
                fig.update_layout(
                    xaxis_title='Classification',
                    yaxis_title='Count',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=theme["text"]),
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                # Show nucleotide distribution
                fig = plot_nucleotide_distribution(analysis_sequence)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show formatted sequence
            st.markdown("### Formatted Sequence")
            display_formatted_dna(analysis_sequence, "Analyzed Sequence")
            
            st.markdown('</div>', unsafe_allow_html=True)  # End of analysis card
    
    # Method Explained Tab
    with dashboard_tab2:
        st.markdown('<div class="card card-primary animate-slide-in">', unsafe_allow_html=True)
        st.subheader("Encryption Methodology")
        
        st.markdown("""
        The DNA Encryption Studio uses a combination of genetic pattern recognition and binary transformation to encrypt DNA sequences.
        
        ### Key Steps:
        1. **Segmentation**: The DNA sequence is divided into segments of fixed length.
        2. **Classification**: Each segment is classified based on specific genetic patterns (e.g., Mutated, Crossover, Reshape).
        3. **Binary Encoding**: Each nucleotide (A, T, C, G) is converted to a binary representation.
        4. **Functional Mapping**: Classified segments are mapped to specific binary keys.
        5. **Custom Key (Optional)**: Users can provide a custom key for additional encryption.
        
        ### Example:
        - Original Sequence: `ATCGATCG`
        - Segments: `ATCGA`, `TCG`
        - Classification: `Reshape`, `Crossover`
        - Binary Encoding: `0001101000110100`
        - Encrypted Output: `111000110100110`
        
        The encrypted output can be decrypted back to the original sequence using the same methodology.
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Documentation Tab
    with dashboard_tab3:
        st.markdown('<div class="card card-primary animate-slide-in">', unsafe_allow_html=True)
        st.subheader("Documentation")
        
        st.markdown("""
        ### DNA Encryption Studio Documentation
        
        **Version**: 2.0
        
        **Features**:
        - Encrypt and decrypt DNA sequences
        - Analyze nucleotide distribution and patterns
        - Visualize genetic data from Hugging Face datasets
        - Apply custom encryption keys
        
        **Usage**:
        1. **Encrypt DNA**: Navigate to the "Encrypt DNA" tab, input a DNA sequence, and configure encryption settings.
        2. **Decrypt DNA**: Navigate to the "Decrypt DNA" tab, upload a JSON file or use saved data, and view decryption results.
        3. **Analytics Dashboard**: Explore DNA sequences, encryption metrics, and learn about the encryption methodology.
        
        **Supported Datasets**:
        - `LysandreJik/canonical-human-genes-sequences`: Canonical human gene sequences
        - `bigcode/fastaDNA`: Various DNA sequences in FASTA format
        - `dnagpt/human_genome_GCF_009914755.1`: Human genome sequences
        - `dnagpt/human_genome_GCF_000001405.40`: Human genome sequences
        - `soleimanian/dna_test`: Test DNA sequences
        
        **Encryption Methodology**:
        - Segmentation, classification, binary encoding, and functional mapping
        - Optional custom key for additional encryption
        
        **Contact**:
        For support or inquiries, please contact the DNA Encryption Studio team at support@dnaencryptionstudio.com.
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()