import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from datetime import datetime, timedelta
import time
from threading import Thread
import queue
import json
from pinecone import Pinecone, ServerlessSpec
import hashlib
import requests
# ============================================================================
# CONFIGURATION - API KEYS
# ============================================================================

# ThingSpeak Configuration
DEFAULT_THINGSPEAK_CHANNEL_ID = "3191826"
DEFAULT_THINGSPEAK_READ_API_KEY = "39BXAFZH258E35WX"

# Pinecone Configuration
DEFAULT_PINECONE_API_KEY = "pcsk_5Birbn_DTe6kq71XqxYtWcfi4RnULH4xRo6nMsK9GgyhMeGGzqRenSHRgRxuNAVKeEv5bh"


class ThingSpeakConnector:
    """Handles ThingSpeak API integration for real sensor data"""
    
    def __init__(self, channel_id="3191826", read_api_key="39BXAFZH258E35WX"):
        """
        Initialize ThingSpeak connection
        
        Get your Channel ID and Read API Key from:
        https://thingspeak.com/channels/your_channel
        """
        self.channel_id = channel_id
        self.read_api_key = read_api_key
        self.base_url = "https://api.thingspeak.com"
        
    def test_connection(self):
        """Test if ThingSpeak connection works"""
        if not self.channel_id or not self.read_api_key:
            return False, "Channel ID and API Key required"
        
        try:
            url = f"{self.base_url}/channels/{self.channel_id}/feeds.json"
            params = {
                'api_key': self.read_api_key,
                'results': 1
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'feeds' in data and len(data['feeds']) > 0:
                    return True, "Connection successful!"
                else:
                    return False, "No data available in channel"
            else:
                return False, f"HTTP Error {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"
    
    def get_latest_reading(self):
        """
        Fetch the latest reading from ThingSpeak
        
        Expected ThingSpeak Fields:
        - field1: Flow rate
        - field2: Temperature
        - field3: Turbidity
        - field4: TDS
        - field5: pH
        """
        if not self.channel_id or not self.read_api_key:
            raise ValueError("Channel ID and API Key must be configured")
        
        try:
            url = f"{self.base_url}/channels/{self.channel_id}/feeds.json"
            params = {
                'api_key': self.read_api_key,
                'results': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'feeds' not in data or len(data['feeds']) == 0:
                return None
            
            latest_feed = data['feeds'][0]
            
            # Parse sensor values (handle None/empty values)
            reading = {
                'flow': float(latest_feed.get('field1', 0) or 0),
                'temperature': float(latest_feed.get('field2', 0) or 0),
                'turbidity': float(latest_feed.get('field3', 0) or 0),
                'tds': float(latest_feed.get('field4', 0) or 0),
                'ph': float(latest_feed.get('field5', 7.0) or 7.0),
                'timestamp': latest_feed.get('created_at', datetime.now().isoformat())
            }
            
            return reading
            
        except requests.exceptions.RequestException as e:
            st.error(f"ThingSpeak API error: {e}")
            return None
        except (ValueError, KeyError) as e:
            st.error(f"Data parsing error: {e}")
            return None
    
    def get_channel_info(self):
        """Get information about the ThingSpeak channel"""
        if not self.channel_id or not self.read_api_key:
            return None
        
        try:
            url = f"{self.base_url}/channels/{self.channel_id}/feeds.json"
            params = {
                'api_key': self.read_api_key,
                'results': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            channel = data.get('channel', {})
            
            return {
                'name': channel.get('name', 'Unknown'),
                'description': channel.get('description', ''),
                'field1': channel.get('field1', 'Flow'),
                'field2': channel.get('field2', 'Temperature'),
                'field3': channel.get('field3', 'Turbidity'),
                'field4': channel.get('field4', 'TDS'),
                'field5': channel.get('field5', 'pH'),
                'last_entry_id': channel.get('last_entry_id', 0)
            }
            
        except Exception as e:
            return None

# ============================================================================
# PINECONE DATABASE SETUP
# ============================================================================

class PineconeConnector:
    """Handles all Pinecone vector database operations"""
    
    def __init__(self, api_key="pcsk_5Birbn_DTe6kq71XqxYtWcfi4RnULH4xRo6nMsK9GgyhMeGGzqRenSHRgRxuNAVKeEv5bh", index_name='water-monitoring'):
        """Initialize Pinecone connection"""
        self.api_key = api_key or st.secrets.get("PINECONE_API_KEY", None)
        self.index_name = index_name
        self.pc = None
        self.index = None
        
    def connect(self):
        """Establish connection to Pinecone"""
        try:
            if not self.api_key:
                st.warning("⚠️ Pinecone API key not found. Add it to Streamlit secrets or pass directly.")
                return False
                
            self.pc = Pinecone(api_key=self.api_key)
            
            # Create index if it doesn't exist
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=11,  # 5 parameters + 5 SHAP values + 1 timestamp encoding
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                st.info(f"Created new Pinecone index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            return True
            
        except Exception as e:
            st.error(f"❌ Pinecone connection failed: {e}")
            return False
    
    def _create_vector_id(self, site_id, timestamp):
        """Create unique vector ID"""
        unique_string = f"{site_id}_{timestamp.isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _encode_reading(self, reading_data, shap_values, timestamp):
        """Convert reading into vector embedding"""
        # Normalize parameters for better vector representation
        params = [
            reading_data['flow'] / 250.0,
            reading_data['temperature'] / 60.0,
            reading_data['turbidity'] / 250.0,
            reading_data['tds'] / 7000.0,
            reading_data['ph'] / 14.0
        ]
        
        # Normalize SHAP values
        shaps = [float(s) for s in shap_values[:5]]
        
        # Encode time as cyclical feature
        hour = timestamp.hour
        time_encoding = np.sin(2 * np.pi * hour / 24)
        
        # Combine into single vector
        vector = params + shaps + [time_encoding]
        return vector
    
    def insert_reading(self, site_id, category, reading_data, prediction_data):
        """Insert a new water quality reading as vector"""
        if not self.index:
            return False
            
        timestamp = datetime.now()
        vector_id = self._create_vector_id(site_id, timestamp)
        
        # Create embedding vector
        vector = self._encode_reading(
            reading_data,
            prediction_data['shap_values'],
            timestamp
        )
        
        # Metadata for filtering and display
        metadata = {
            'site_id': site_id,
            'category': category,
            'timestamp': timestamp.isoformat(),
            'flow': float(reading_data['flow']),
            'temperature': float(reading_data['temperature']),
            'turbidity': float(reading_data['turbidity']),
            'tds': float(reading_data['tds']),
            'ph': float(reading_data['ph']),
            'is_safe': prediction_data['is_safe'],
            'confidence': float(prediction_data['confidence']),
            'shap_values': json.dumps(prediction_data['shap_values'])
        }
        
        # Upsert to Pinecone
        self.index.upsert(vectors=[(vector_id, vector, metadata)])
        return True
    
    def get_recent_readings(self, site_id, category, hours=24, limit=1000):
        """Get recent readings for a site and category"""
        if not self.index:
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            results = self.index.query(
                vector=[0.0] * 11,
                top_k=limit,
                include_metadata=True,
                filter={
                    'site_id': {'$eq': site_id},
                    'category': {'$e`q': category}
                }
            )
            
            data = []
            for match in results.matches:
                meta = match.metadata
                reading_time = datetime.fromisoformat(meta['timestamp'])
                if reading_time > cutoff_time:
                    data.append({
                        'timestamp': reading_time,
                        'flow': meta['flow'],
                        'temperature': meta['temperature'],
                        'turbidity': meta['turbidity'],
                        'tds': meta['tds'],
                        'ph': meta['ph'],
                        'is_safe': meta['is_safe'],
                        'confidence': meta['confidence']
                    })
            
            return pd.DataFrame(data).sort_values('timestamp')
            
        except Exception as e:
            st.error(f"Query error: {e}")
            return pd.DataFrame()
    
    def find_similar_readings(self, reading_data, shap_values, category, top_k=5):
        """Find similar historical readings using vector similarity"""
        if not self.index:
            return []
        
        vector = self._encode_reading(reading_data, shap_values, datetime.now())
        
        try:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter={'category': {'$eq': category}}
            )
            
            similar = []
            for match in results.matches:
                meta = match.metadata
                similar.append({
                    'similarity': float(match.score),
                    'timestamp': meta['timestamp'],
                    'flow': meta['flow'],
                    'temperature': meta['temperature'],
                    'turbidity': meta['turbidity'],
                    'tds': meta['tds'],
                    'ph': meta['ph'],
                    'is_safe': meta['is_safe'],
                    'confidence': meta['confidence']
                })
            
            return similar
            
        except Exception as e:
            st.error(f"Similarity search error: {e}")
            return []
    
    def get_statistics(self):
        """Get index statistics"""
        if not self.index:
            return None
        try:
            stats = self.index.describe_index_stats()
            return stats
        except:
            return None

# ============================================================================
# ALERT MANAGER
# ============================================================================

class AlertManager:
    """Manages alerts in memory"""
    
    def __init__(self):
        self.alerts = []
    
    def add_alert(self, site_id, category, alert_type, message, severity):
        """Add new alert"""
        self.alerts.append({
            'timestamp': datetime.now(),
            'site_id': site_id,
            'category': category,
            'alert_type': alert_type,
            'message': message,
            'severity': severity
        })
        
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
    
    def get_recent_alerts(self, site_id, hours=24):
        """Get recent alerts for a site"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if alert['site_id'] == site_id and alert['timestamp'] > cutoff_time
        ]

# ============================================================================
# REAL-TIME DATA COLLECTOR (with ThingSpeak)
# ============================================================================

class RealTimeCollector:
    """Handles real-time data collection from ThingSpeak"""
    
    def __init__(self, db_connector, models_dict, alert_manager, thingspeak_connector):
        self.db = db_connector
        self.models_dict = models_dict
        self.alert_manager = alert_manager
        self.thingspeak = thingspeak_connector
        self.data_queue = queue.Queue()
        self.is_running = False
        self.last_entry_id = None
        
    def start_collection(self, site_id, category, interval_seconds=15):
        """Start collecting data in background"""
        self.is_running = True
        self.collection_thread = Thread(
            target=self._collect_loop,
            args=(site_id, category, interval_seconds),
            daemon=True
        )
        self.collection_thread.start()
    
    def stop_collection(self):
        """Stop data collection"""
        self.is_running = False
    
    def _collect_loop(self, site_id, category, interval):
        """Background loop for data collection from ThingSpeak"""
        while self.is_running:
            try:
                # Get latest reading from ThingSpeak
                reading = self.thingspeak.get_latest_reading()
                
                if reading:
                    # Extract sensor data
                    sensor_data = {
                        'flow': reading['flow'],
                        'temperature': reading['temperature'],
                        'turbidity': reading['turbidity'],
                        'tds': reading['tds'],
                        'ph': reading['ph']
                    }
                    
                    # Make prediction
                    prediction = self._predict(category, sensor_data)
                    
                    # Store in Pinecone
                    if self.db and self.db.index:
                        self.db.insert_reading(site_id, category, sensor_data, prediction)
                        
                        # Generate alert if unsafe
                        if not prediction['is_safe']:
                            self.alert_manager.add_alert(
                                site_id, category, 'UNSAFE_WATER',
                                f"Water parameters unsafe for {category} use",
                                'HIGH'
                            )
                    
                    # Add to queue for UI update
                    self.data_queue.put({
                        'timestamp': datetime.now(),
                        'reading': sensor_data,
                        'prediction': prediction
                    })
                else:
                    st.warning("No data received from ThingSpeak")
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Collection error: {e}")
                time.sleep(interval)
    
    def _predict(self, category, reading):
        """Make safety prediction"""
        model = self.models_dict[category]["model"]
        scaler = self.models_dict[category]["scaler"]
        
        input_data = [
            reading['flow'],
            reading['temperature'],
            reading['turbidity'],
            reading['tds'],
            reading['ph']
        ]
        
        X = scaler.transform([input_data])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values_display = shap_values[1]
        else:
            shap_values_display = shap_values
        
        return {
            'is_safe': bool(prediction == 1),
            'confidence': float(max(probability)),
            'shap_values': shap_values_display[0].tolist()
        }
    
    def get_latest_data(self):
        """Get latest data from queue"""
        if not self.data_queue.empty():
            return self.data_queue.get()
        return None

# ============================================================================
# STREAMLIT APP
# ============================================================================

# Load ML models
@st.cache_resource
def load_models():
    model_file = "../model/water_safety_models.pkl"
    with open(model_file, "rb") as f:
        return pickle.load(f)

# Initialize Pinecone connection
@st.cache_resource
def init_database():
    db = PineconeConnector()  # Uses default API key from class
    if db.connect():
        return db
    return None

# Initialize alert manager
@st.cache_resource
def init_alert_manager():
    return AlertManager()

# Page config
st.set_page_config(page_title="ThingSpeak Water Monitoring", layout="wide", page_icon="💧")

# Load resources
models_dict = load_models()
db_connector = init_database()
alert_manager = init_alert_manager()

# Initialize session state
if 'collector' not in st.session_state:
    st.session_state.collector = None
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False
if 'thingspeak' not in st.session_state:
    st.session_state.thingspeak = None

# ============================================================================
# UI LAYOUT
# ============================================================================

st.title("💧 Real-Time Water Quality Monitoring System")
st.markdown("**ThingSpeak IoT Integration + Pinecone Vector Database + AI Analysis**")

# Sidebar controls
st.sidebar.header("🌐 ThingSpeak Configuration")

# ThingSpeak connection
channel_id = st.sidebar.text_input(
    "ThingSpeak Channel ID",
    help="Find this in your ThingSpeak channel settings"
)

read_api_key = st.sidebar.text_input(
    "Read API Key",
    type="password",
    help="Get this from API Keys tab in ThingSpeak"
)

if channel_id and read_api_key:
    if st.sidebar.button("🔌 Test ThingSpeak Connection"):
        thingspeak = ThingSpeakConnector(channel_id, read_api_key)
        success, message = thingspeak.test_connection()
        
        if success:
            st.sidebar.success(f"✅ {message}")
            st.session_state.thingspeak = thingspeak
            
            # Show channel info
            info = thingspeak.get_channel_info()
            if info:
                st.sidebar.info(f"""
**Channel:** {info['name']}
**Last Entry:** #{info['last_entry_id']}

**Field Mapping:**
- Field 1: {info['field1']}
- Field 2: {info['field2']}
- Field 3: {info['field3']}
- Field 4: {info['field4']}
- Field 5: {info['field5']}
                """)
        else:
            st.sidebar.error(f"❌ {message}")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Monitoring Controls")

# Pinecone API Key
if not db_connector or not db_connector.index:
    pinecone_key = st.sidebar.text_input(
        "Pinecone API Key",
        type="password",
        help="Get from https://app.pinecone.io/"
    )
    if pinecone_key and st.sidebar.button("Connect to Pinecone"):
        db_connector = PineconeConnector(api_key=pinecone_key)
        if db_connector.connect():
            st.sidebar.success("✅ Pinecone Connected!")
            st.rerun()

site_id = st.sidebar.text_input("Site ID", value="SITE_001")
category = st.sidebar.selectbox(
    "Water Category",
    options=list(models_dict.keys())
)
collection_interval = st.sidebar.slider(
    "Collection Interval (seconds)",
    min_value=15, max_value=300, value=30,
    help="ThingSpeak updates every 15 seconds minimum"
)

# Start/Stop monitoring
col1, col2 = st.sidebar.columns(2)

start_enabled = (
    st.session_state.thingspeak is not None and 
    db_connector and db_connector.index and
    not st.session_state.is_monitoring
)

if col1.button("▶️ Start", type="primary", disabled=not start_enabled):
    st.session_state.collector = RealTimeCollector(
        db_connector, models_dict, alert_manager, st.session_state.thingspeak
    )
    st.session_state.collector.start_collection(site_id, category, collection_interval)
    st.session_state.is_monitoring = True
    st.rerun()

if col2.button("⏹️ Stop", disabled=not st.session_state.is_monitoring):
    if st.session_state.collector:
        st.session_state.collector.stop_collection()
    st.session_state.is_monitoring = False
    st.rerun()

# Status indicators
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Connection Status")

if st.session_state.thingspeak:
    st.sidebar.success("✅ ThingSpeak Connected")
else:
    st.sidebar.warning("⚠️ ThingSpeak Not Connected")

if db_connector and db_connector.index:
    st.sidebar.success("✅ Pinecone Connected")
    stats = db_connector.get_statistics()
    if stats:
        st.sidebar.info(f"📊 Vectors: {stats.total_vector_count}")
else:
    st.sidebar.warning("⚠️ Pinecone Not Connected")

if st.session_state.is_monitoring:
    st.sidebar.success(f"🟢 **MONITORING ACTIVE**")
    st.sidebar.info(f"Site: {site_id}\nCategory: {category}\nInterval: {collection_interval}s")
else:
    st.sidebar.info("⚪ Monitoring Stopped")

# ============================================================================
# MAIN CONTENT
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Dashboard",
    "🔍 Similar Patterns",
    "📈 Historical Trends",
    "🚨 Alerts",
    "⚙️ Setup Guide"
])

# TAB 1: Live Dashboard
with tab1:
    if st.session_state.is_monitoring and st.session_state.collector:
        
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        refresh_count = 0
        while refresh_count < 100 and st.session_state.is_monitoring:
            latest = st.session_state.collector.get_latest_data()
            
            if latest:
                reading = latest['reading']
                prediction = latest['prediction']
                timestamp = latest['timestamp']
                
                with status_placeholder.container():
                    if prediction['is_safe']:
                        st.success(f"✅ **SAFE** - Water quality acceptable for {category} use | {timestamp.strftime('%H:%M:%S')}")
                    else:
                        st.error(f"❌ **UNSAFE** - Water quality NOT acceptable for {category} use | {timestamp.strftime('%H:%M:%S')}")
                
                with metrics_placeholder.container():
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("Flow", f"{reading['flow']:.1f}")
                    col2.metric("Temp", f"{reading['temperature']:.1f}°C")
                    col3.metric("Turbidity", f"{reading['turbidity']:.1f}")
                    col4.metric("TDS", f"{reading['tds']:.0f}")
                    col5.metric("pH", f"{reading['ph']:.2f}")
                    col6.metric("Confidence", f"{prediction['confidence']*100:.1f}%")
                
                if db_connector and db_connector.index:
                    with chart_placeholder.container():
                        hist_data = db_connector.get_recent_readings(site_id, category, hours=1)
                        
                        if not hist_data.empty and len(hist_data) > 1:
                            st.subheader("📉 Last Hour Trends")
                            
                            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                            fig.suptitle('Parameter Trends (Last Hour)', fontsize=16, fontweight='bold')
                            
                            params = ['flow', 'temperature', 'turbidity', 'tds', 'ph']
                            for idx, param in enumerate(params):
                                ax = axes[idx // 3, idx % 3]
                                ax.plot(hist_data['timestamp'], hist_data[param], marker='o', linewidth=2, markersize=4)
                                ax.set_title(param.capitalize(), fontweight='bold')
                                ax.set_xlabel('Time')
                                ax.set_ylabel('Value')
                                ax.grid(True, alpha=0.3)
                                ax.tick_params(axis='x', rotation=45)
                            
                            ax = axes[1, 2]
                            safety_colors = ['red' if not safe else 'green' for safe in hist_data['is_safe']]
                            ax.scatter(hist_data['timestamp'], hist_data['confidence'], c=safety_colors, s=100, alpha=0.6)
                            ax.set_title('Safety Status', fontweight='bold')
                            ax.set_xlabel('Time')
                            ax.set_ylabel('Confidence')
                            ax.grid(True, alpha=0.3)
                            ax.tick_params(axis='x', rotation=45)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                
                refresh_count += 1
            
            time.sleep(collection_interval)
    else:
        st.info("👆 Configure ThingSpeak and Pinecone in the sidebar, then click 'Start'")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 Quick Start Guide
            
            **Step 1: ThingSpeak Setup**
            1. Create account at [thingspeak.com](https://thingspeak.com)
            2. Create a new channel
            3. Configure 5 fields:
               - Field 1: Flow
               - Field 2: Temperature
               - Field 3: Turbidity
               - Field 4: TDS
               - Field 5: pH
            4. Get Channel ID and Read API Key
            
            **Step 2: Connect Your Sensors**
            - Send data to ThingSpeak every 15+ seconds
            - Use ThingSpeak Arduino/Python libraries
            
            **Step 3: Start Monitoring**
            - Enter credentials in sidebar
            - Test connections
            - Click Start!
            """)
        
        with col2:
            st.markdown("""
            ### 📡 ThingSpeak Field Mapping
            
            Your IoT sensors must send data to:
            
            | Field | Parameter | Unit |
            |-------|-----------|------|
            | field1 | Flow rate | - |
            | field2 | Temperature | °C |
            | field3 | Turbidity | NTU |
            | field4 | TDS | ppm |
            | field5 | pH | 0-14 |
            
            ### ✨ Features
            
            - **Live IoT Data**: Direct from ThingSpeak
            - **AI Safety Analysis**: ML predictions
            - **Vector Search**: Find similar patterns
            - **Historical Tracking**: Pinecone storage
            - **Smart Alerts**: Automatic notifications
            """)

# TAB 2: Similar Patterns
with tab2:
    st.subheader("🔍 Find Similar Historical Patterns")
    
    if db_connector and db_connector.index:
        st.write("### Enter Water Parameters to Search")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        search_flow = col1.number_input("Flow", value=100.0, step=1.0)
        search_temp = col2.number_input("Temp (°C)", value=25.0, step=0.1)
        search_turb = col3.number_input("Turbidity", value=50.0, step=1.0)
        search_tds = col4.number_input("TDS", value=500.0, step=10.0)
        search_ph = col5.number_input("pH", value=7.0, step=0.1)
        
        num_similar = st.slider("Number of similar patterns", 3, 20, 5)
        
        if st.button("🔍 Search Similar Readings", type="primary"):
            search_reading = {
                'flow': search_flow,
                'temperature': search_temp,
                'turbidity': search_turb,
                'tds': search_tds,
                'ph': search_ph
            }
            
            model = models_dict[category]["model"]
            scaler = models_dict[category]["scaler"]
            input_data = [search_flow, search_temp, search_turb, search_tds, search_ph]
            X = scaler.transform([input_data])
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            similar = db_connector.find_similar_readings(
                search_reading,
                shap_values[0].tolist(),
                category,
                top_k=num_similar
            )
            
            if similar:
                st.success(f"✅ Found {len(similar)} similar readings!")
                
                for idx, match in enumerate(similar, 1):
                    with st.expander(f"#{idx} - Similarity: {match['similarity']:.2%} | {match['timestamp'][:19]}"):
                        cols = st.columns([2, 3])
                        
                        with cols[0]:
                            st.write("**Parameters:**")
                            st.write(f"- Flow: {match['flow']:.1f}")
                            st.write(f"- Temperature: {match['temperature']:.1f}°C")
                            st.write(f"- Turbidity: {match['turbidity']:.1f}")
                            st.write(f"- TDS: {match['tds']:.0f}")
                            st.write(f"- pH: {match['ph']:.2f}")
                        
                        with cols[1]:
                            if match['is_safe']:
                                st.success(f"✅ SAFE (Confidence: {match['confidence']*100:.1f}%)")
                            else:
                                st.error(f"❌ UNSAFE (Confidence: {match['confidence']*100:.1f}%)")
                            
                            st.write(f"**Similarity Score:** {match['similarity']:.2%}")
                            st.write(f"**Timestamp:** {match['timestamp'][:19]}")
            else:
                st.info("No similar readings found. Start monitoring to build history.")
    else:
        st.warning("⚠️ Connect to Pinecone first")

# TAB 3: Historical Trends
with tab3:
    st.subheader("📈 Historical Analysis")
    
    if db_connector and db_connector.index:
        hours_back = st.slider("Time Range (hours)", 1, 168, 24)
        
        with st.spinner("Loading historical data..."):
            hist_data = db_connector.get_recent_readings(site_id, category, hours=hours_back)
        
        if not hist_data.empty:
            st.write(f"Showing {len(hist_data)} readings from the last {hours_back} hours")
            
            col1, col2, col3, col4 = st.columns(4)
            safe_count = hist_data['is_safe'].sum()
            total_count = len(hist_data)
            
            col1.metric("Total Readings", total_count)
            col2.metric("Safe", int(safe_count), f"{safe_count/total_count*100:.1f}%")
            col3.metric("Unsafe", int(total_count - safe_count), f"{(total_count-safe_count)/total_count*100:.1f}%")
            col4.metric("Avg Confidence", f"{hist_data['confidence'].mean()*100:.1f}%")
            
            st.markdown("### Parameter History")
            
            params = ['flow', 'temperature', 'turbidity', 'tds', 'ph']
            selected_params = st.multiselect("Select parameters to display", params, default=params[:3])
            
            if selected_params:
                fig, ax = plt.subplots(figsize=(14, 6))
                for param in selected_params:
                    ax.plot(hist_data['timestamp'], hist_data[param], marker='o', label=param.capitalize(), linewidth=2, markersize=4)
                
                ax.set_xlabel('Timestamp', fontweight='bold')
                ax.set_ylabel('Value', fontweight='bold')
                ax.set_title('Parameter Trends Over Time', fontweight='bold', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with st.expander("📋 View Raw Data"):
                st.dataframe(hist_data.sort_values('timestamp', ascending=False), use_container_width=True)
        else:
            st.info("No historical data available. Start monitoring to collect data!")
    else:
        st.warning("⚠️ Connect to Pinecone first")

# TAB 4: Alerts
with tab4:
    st.subheader("🚨 Alert History")
    
    alert_hours = st.slider("Alert Time Range (hours)", 1, 168, 24, key="alert_hours")
    
    alerts = alert_manager.get_recent_alerts(site_id, hours=alert_hours)
    
    if alerts:
        st.write(f"Showing {len(alerts)} alerts from the last {alert_hours} hours")
        
        for alert in sorted(alerts, key=lambda x: x['timestamp'], reverse=True):
            severity_colors = {
                'HIGH': '🔴',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }
            
            icon = severity_colors.get(alert['severity'], '⚪')
            
            with st.expander(f"{icon} {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {alert['alert_type']}"):
                st.write(f"**Category:** {alert['category']}")
                st.write(f"**Severity:** {alert['severity']}")
                st.write(f"**Message:** {alert['message']}")
    else:
        st.success("✅ No alerts in the selected time range")

# TAB 5: Setup Guide
with tab5:
    st.subheader("⚙️ Complete Setup Guide")
    
    st.markdown("""
    ## 🎯 ThingSpeak + Pinecone Integration Guide
    
    ### Part 1: ThingSpeak Setup (IoT Platform)
    
    #### Step 1: Create ThingSpeak Account
    1. Go to [https://thingspeak.com](https://thingspeak.com)
    2. Sign up for a free account
    3. Verify your email
    
    #### Step 2: Create a Channel
    1. Click "Channels" → "My Channels" → "New Channel"
    2. Fill in channel details:
       - **Name**: "Water Quality Monitor - Site 001"
       - **Description**: "Real-time water quality monitoring"
    3. Configure 5 fields:
       - **Field 1**: Flow
       - **Field 2**: Temperature
       - **Field 3**: Turbidity
       - **Field 4**: TDS
       - **Field 5**: pH
    4. Click "Save Channel"
    
    #### Step 3: Get API Keys
    1. Go to "API Keys" tab in your channel
    2. Copy **Channel ID** (numbers)
    3. Copy **Read API Key** (for reading data)
    4. Copy **Write API Key** (for your sensors)
    
    ---
    
    ### Part 2: Send Data to ThingSpeak
    
    #### Option A: Arduino/ESP8266/ESP32
    ```cpp
    #include <WiFi.h>
    #include "ThingSpeak.h"
    
    const char* ssid = "YOUR_WIFI";
    const char* password = "YOUR_PASSWORD";
    unsigned long channelID = YOUR_CHANNEL_ID;
    const char* writeAPIKey = "YOUR_WRITE_KEY";
    
    WiFiClient client;
    
    void setup() {
      WiFi.begin(ssid, password);
      ThingSpeak.begin(client);
    }
    
    void loop() {
      // Read sensor values
      float flow = readFlowSensor();
      float temp = readTempSensor();
      float turbidity = readTurbiditySensor();
      float tds = readTDSSensor();
      float ph = readPhSensor();
      
      // Send to ThingSpeak
      ThingSpeak.setField(1, flow);
      ThingSpeak.setField(2, temp);
      ThingSpeak.setField(3, turbidity);
      ThingSpeak.setField(4, tds);
      ThingSpeak.setField(5, ph);
      
      ThingSpeak.writeFields(channelID, writeAPIKey);
      
      delay(20000); // Wait 20 seconds (ThingSpeak free limit)
    }
    ```
    
    #### Option B: Python (Raspberry Pi / Computer)
    ```python
    import requests
    import time
    
    CHANNEL_ID = "YOUR_CHANNEL_ID"
    WRITE_API_KEY = "YOUR_WRITE_KEY"
    
    def send_to_thingspeak(flow, temp, turbidity, tds, ph):
        url = f"https://api.thingspeak.com/update"
        params = {
            'api_key': WRITE_API_KEY,
            'field1': flow,
            'field2': temp,
            'field3': turbidity,
            'field4': tds,
            'field5': ph
        }
        response = requests.get(url, params=params)
        return response.text
    
    # Example loop
    while True:
        # Read your sensors here
        flow = read_flow_sensor()
        temp = read_temp_sensor()
        turbidity = read_turbidity_sensor()
        tds = read_tds_sensor()
        ph = read_ph_sensor()
        
        # Send to ThingSpeak
        send_to_thingspeak(flow, temp, turbidity, tds, ph)
        
        time.sleep(20)  # Wait 20 seconds
    ```
    
    #### Option C: HTTP Request (Any Language)
    ```bash
    # Simple curl command
    curl "https://api.thingspeak.com/update?api_key=YOUR_WRITE_KEY&field1=100&field2=25&field3=50&field4=500&field5=7.0"
    ```
    
    ---
    
    ### Part 3: Pinecone Setup (Vector Database)
    
    #### Step 1: Create Pinecone Account
    1. Go to [https://app.pinecone.io](https://app.pinecone.io)
    2. Sign up for free (no credit card required)
    3. Verify email
    
    #### Step 2: Create Index
    1. Click "Create Index"
    2. Fill in:
       - **Name**: `water-monitoring`
       - **Dimensions**: `11`
       - **Metric**: `cosine`
       - **Cloud**: AWS
       - **Region**: Choose closest to you
    3. Click "Create Index"
    
    #### Step 3: Get API Key
    1. Click "API Keys" in left sidebar
    2. Copy your API key
    3. Paste into Streamlit sidebar
    
    ---
    
    ### Part 4: Run This App
    
    #### Install Dependencies
    ```bash
    pip install streamlit pandas numpy matplotlib seaborn shap
    pip install pinecone-client requests
    ```
    
    #### Run the App
    ```bash
    streamlit run water_monitoring_app.py
    ```
    
    #### Configure in Sidebar
    1. Enter ThingSpeak Channel ID
    2. Enter ThingSpeak Read API Key
    3. Test ThingSpeak connection
    4. Enter Pinecone API Key
    5. Connect to Pinecone
    6. Click "Start" to begin monitoring
    
    ---
    
    ### 🔧 Troubleshooting
    
    #### ThingSpeak Issues
    
    **"No data received"**
    - ✅ Check if sensors are sending data
    - ✅ Verify Write API Key is correct
    - ✅ Check ThingSpeak channel has recent data
    - ✅ Free accounts: max 1 update every 15 seconds
    
    **"Connection failed"**
    - ✅ Verify Channel ID is correct
    - ✅ Verify Read API Key is correct
    - ✅ Check internet connection
    - ✅ Make channel public or use correct API key
    
    #### Pinecone Issues
    
    **"Connection failed"**
    - ✅ Verify API key is correct
    - ✅ Check index name matches: `water-monitoring`
    - ✅ Verify dimension is 11
    - ✅ Check internet connection
    
    **"No vectors stored"**
    - ✅ Start monitoring first
    - ✅ Wait a few collection cycles
    - ✅ Check sidebar for vector count
    
    ---
    
    ### 📊 Understanding the Data Flow
    
    ```
    IoT Sensors → ThingSpeak → This App → ML Model → Pinecone
         ↓            ↓           ↓           ↓          ↓
    Real-time    Cloud API    Safety      Vector    Historical
     Readings     Storage     Prediction  Storage    Analysis
    ```
    
    ---
    
    ### 🎓 Sensor Recommendations
    
    | Parameter | Recommended Sensor | Price Range |
    |-----------|-------------------|-------------|
    | Flow | YF-S201 Flow Sensor | $5-10 |
    | Temperature | DS18B20 Waterproof | $2-5 |
    | Turbidity | SEN0189 Turbidity | $10-15 |
    | TDS | TDS Meter Sensor | $5-10 |
    | pH | PH-4502C pH Sensor | $15-25 |
    
    **Total Cost**: ~$40-70 for complete sensor setup
    
    ---
    
    ### 🚀 Scaling Considerations
    
    #### ThingSpeak Limits
    - **Free**: 3 million messages/year, 1 update/15 sec
    - **Student**: 33 million messages/year
    - **Standard**: 1 update/second, $10/month
    
    #### Pinecone Limits
    - **Free**: 100K vectors (≈69 days at 1 min intervals)
    - **Starter**: $70/month, 1M vectors
    - **Enterprise**: Unlimited, custom pricing
    
    #### Recommendations
    - Start with free tiers
    - Monitor usage in dashboards
    - Upgrade when needed
    - Implement data retention policies
    
    ---
    
    ### 🔒 Security Best Practices
    
    1. **Never commit API keys to GitHub**
       ```toml
       # .streamlit/secrets.toml
       THINGSPEAK_CHANNEL_ID = "your_channel_id"
       THINGSPEAK_READ_API_KEY = "your_read_key"
       PINECONE_API_KEY = "your_pinecone_key"
       ```
    
    2. **Use environment variables in production**
       ```python
       import os
       api_key = os.environ.get('PINECONE_API_KEY')
       ```
    
    3. **Set ThingSpeak channels to private**
    4. **Rotate API keys regularly**
    5. **Use separate keys for dev/production**
    
    ---
    
    ### 📚 Additional Resources
    
    - [ThingSpeak Documentation](https://www.mathworks.com/help/thingspeak/)
    - [Pinecone Documentation](https://docs.pinecone.io/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Arduino ThingSpeak Library](https://github.com/mathworks/thingspeak-arduino)
    
    ---
    
    ### 💡 Next Steps
    
    Once you have basic monitoring working:
    
    1. **Add Email Alerts**
       - Use SMTP to send alerts
       - Integrate with Twilio for SMS
    
    2. **Add More Sites**
       - Monitor multiple locations
       - Compare across sites
    
    3. **Advanced Analytics**
       - Anomaly detection
       - Predictive maintenance
       - Trend forecasting
    
    4. **Mobile App**
       - React Native integration
       - Push notifications
    
    5. **Dashboard Customization**
       - Custom visualizations
       - Export reports
       - Scheduled reports
    """)
    
    st.markdown("---")
    
    # Connection test buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔌 Test ThingSpeak", type="primary"):
            if st.session_state.thingspeak:
                success, message = st.session_state.thingspeak.test_connection()
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("⚠️ Configure ThingSpeak first")
    
    with col2:
        if st.button("🔌 Test Pinecone", type="primary"):
            if db_connector and db_connector.index:
                st.success("✅ Pinecone connected!")
                stats = db_connector.get_statistics()
                if stats:
                    st.json({
                        'total_vectors': stats.total_vector_count,
                        'dimension': stats.dimension,
                        'index_fullness': stats.index_fullness
                    })
            else:
                st.error("❌ Connect to Pinecone first")

st.markdown("---")
st.caption("💧 Real-Time Water Quality Monitoring | ThingSpeak IoT + Pinecone Vector DB + AI Analysis | Built with ❤️ using Streamlit")