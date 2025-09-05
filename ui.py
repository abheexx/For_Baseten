"""
Streamlit UI for Whisper Inference Service

A simple web interface for uploading audio files and getting transcriptions.
"""

import streamlit as st
import requests
import json
import time
from typing import Optional, Dict, Any
import io


# Configuration
API_BASE_URL = "http://localhost:8000"
SUPPORTED_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"]


def check_service_health() -> bool:
    """Check if the Whisper service is healthy and ready."""
    try:
        # Check health
        health_response = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
        if health_response.status_code != 200:
            return False
        
        # Check readiness
        ready_response = requests.get(f"{API_BASE_URL}/readyz", timeout=5)
        if ready_response.status_code != 200:
            return False
        
        return True
    except requests.exceptions.RequestException:
        return False


def transcribe_audio(
    audio_file: bytes, 
    filename: str, 
    language: Optional[str] = None,
    task: str = "transcribe"
) -> Dict[str, Any]:
    """Send audio file to the transcription API."""
    
    files = {
        'file': (filename, audio_file, 'audio/wav')
    }
    
    params = {}
    if language:
        params['language'] = language
    if task:
        params['task'] = task
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/transcribe",
            files=files,
            params=params,
            timeout=300  # 5 minutes timeout for transcription
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error {response.status_code}: {response.text}"}
    
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The audio file might be too long."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


def format_transcription_result(result: Dict[str, Any]) -> str:
    """Format transcription result for display."""
    if "error" in result:
        return f"❌ **Error:** {result['error']}"
    
    # Basic info
    output = f"📄 **File:** {result.get('filename', 'Unknown')}\n"
    output += f"🌍 **Language:** {result.get('language', 'Unknown')} "
    output += f"(confidence: {result.get('language_probability', 0):.2%})\n"
    output += f"⏱️ **Duration:** {result.get('duration', 0):.2f} seconds\n\n"
    
    # Transcription
    transcription = result.get('transcription', {})
    full_text = transcription.get('full_text', '')
    
    if full_text:
        output += f"📝 **Transcription:**\n\n{full_text}\n\n"
    
    # Segments with timestamps
    segments = transcription.get('segments', [])
    if segments:
        output += "🕐 **Detailed Segments:**\n\n"
        for i, segment in enumerate(segments, 1):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            # Format time as MM:SS
            start_mm_ss = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
            end_mm_ss = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
            
            output += f"**{i}.** `{start_mm_ss} - {end_mm_ss}` {text}\n"
    
    # Model info
    model_info = result.get('model_info', {})
    if model_info:
        output += f"\n🤖 **Model:** {model_info.get('model_size', 'Unknown')} "
        output += f"({model_info.get('compute_type', 'Unknown')})"
    
    return output


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Whisper Transcription",
        page_icon="🎤",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for elegant styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        color: #2c3e50;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .result-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 2rem 0;
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Whisper</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### Settings")
        
        # Language selection
        language_options = {
            "Auto-detect": None,
            "English": "en",
            "Spanish": "es", 
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Chinese": "zh",
            "Japanese": "ja",
            "Korean": "ko"
        }
        
        selected_language = st.selectbox("Language", options=list(language_options.keys()))
        language_code = language_options[selected_language]
        
        # Task selection
        task = st.selectbox("Task", options=["transcribe", "translate"])
        
        # Service status
        if st.button("Check Status"):
            with st.spinner("Checking..."):
                is_healthy = check_service_health()
                if is_healthy:
                    st.success("✅ Ready")
                else:
                    st.error("❌ Offline")
    
    # Main content
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=[ext[1:] for ext in SUPPORTED_FORMATS],
        help="Supported: MP3, WAV, M4A, FLAC, OGG, WMA, AAC"
    )
    
    if uploaded_file is not None:
        st.info(f"**{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
        
        if st.button("Transcribe", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                audio_content = uploaded_file.read()
                result = transcribe_audio(
                    audio_content=audio_content,
                    filename=uploaded_file.name,
                    language=language_code,
                    task=task
                )
                st.session_state.transcription_result = result
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results section
    if 'transcription_result' in st.session_state:
        result = st.session_state.transcription_result
        
        if "error" not in result:
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Duration", f"{result.get('duration', 0):.1f}s")
            with col2:
                st.metric("Language", result.get('language', 'Unknown'))
            with col3:
                st.metric("Confidence", f"{result.get('language_probability', 0):.1%}")
            with col4:
                segments = result.get('transcription', {}).get('segments', [])
                st.metric("Segments", len(segments))
            
            # Transcription text
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            full_text = result.get('transcription', {}).get('full_text', '')
            if full_text:
                st.markdown("### Transcription")
                st.markdown(f"_{full_text}_")
                
                # Download button
                st.download_button(
                    "Download",
                    full_text,
                    f"transcription_{int(time.time())}.txt",
                    "text/plain"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
