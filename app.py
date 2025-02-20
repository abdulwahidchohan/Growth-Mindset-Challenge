import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import haversine_distances
from scipy.stats import pearsonr
import requests
from streamlit_lottie import st_lottie

# IMPORTANT: st.set_page_config must be the first Streamlit command.
st.set_page_config(page_title="Abdul Wahid's Earthquake Explorer", layout="wide")

# Custom CSS for a modern, friendly, and responsive look
st.markdown("""
    <style>
    /* Base styling */
    body { 
        background-color: #F0F2F6; 
        font-family: 'Segoe UI', sans-serif;
    }
    .big-font { font-size:20px; }
    .header { color: #4C4C6D; }

    /* Responsive header container */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    /* Make images responsive */
    img {
        max-width: 100%;
        height: auto;
    }
    /* Animate tab buttons on hover */
    button[role="tab"] {
        transition: transform 0.3s ease;
    }
    button[role="tab"]:hover {
        transform: scale(1.1);
    }
    /* Media queries for smaller screens */
    @media only screen and (max-width: 768px) {
        .header-container {
            flex-direction: column;
            text-align: center;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Function to load Lottie animations (cached for performance)
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
main_animation  = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_bhw1ul4g.json")
character_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_jcikwtux.json")

# --- Personalized User Setup ---
if "user" not in st.session_state:
    st.session_state["user"] = {"name": "user", "email": "user@example.com"}

st.sidebar.markdown("## Hello, User!")
st.sidebar.info("Welcome to your personalized Earthquake Explorer!")

# Responsive header with animated character
with st.container():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    col_header1, col_header2 = st.columns([1, 3])
    with col_header1:
        if character_animation:
            st_lottie(character_animation, height=150, key="character_anim")
    with col_header2:
        st.title("Abdul Wahid's Earthquake Explorer")
        st.markdown("Hi User, welcome to your personal journey into the world of earthquakes. Enjoy exploring the data with interactive charts, modern animations, and a host of analysis tools!")
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar: Data Upload and Filtering Options
st.sidebar.header("Data Input & Filters")
uploaded_file = st.sidebar.file_uploader("Upload your earthquake CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        return df
    except Exception as e:
        st.error("Oops! There was an error reading your file: " + str(e))
        return None

# Load data from file or use a sample dataset
if uploaded_file is not None:
    with st.spinner("Loading your data..."):
        data = load_data(uploaded_file)
    if data is None:
        st.stop()
else:
    st.info("No file uploaded. We're using a built-in sample dataset for you.")
    np.random.seed(42)
    sample_size = 300
    regions = ["North America", "South America", "Asia", "Europe", "Africa", "Oceania"]
    # Generate realistic earthquake locations along tectonic plates
    tectonic_plates = [
        (30, 120), (-20, -70), (40, -120), (-10, 150), (60, -150)
    ]
    plate_points = np.random.choice(len(tectonic_plates), sample_size)
    data = pd.DataFrame({
        'time': pd.date_range(start="2015-01-01", periods=sample_size, freq='W'),
        'latitude': np.random.normal([tectonic_plates[i][0] for i in plate_points], 5),
        'longitude': np.random.normal([tectonic_plates[i][1] for i in plate_points], 5),
        'depth': np.random.uniform(0, 700, sample_size),
        'magnitude': np.random.exponential(scale=2.5, size=sample_size) + 2.5,
        'region': np.random.choice(regions, sample_size)
    })

# Validate required columns
required_columns = ['time', 'latitude', 'longitude', 'depth', 'magnitude']
for col in required_columns:
    if col not in data.columns:
        st.error(f"Uh-oh, your data is missing the '{col}' column.")
        st.stop()

# Sidebar: Additional Filters
st.sidebar.header("Customize Your View")
min_mag, max_mag = float(data['magnitude'].min()), float(data['magnitude'].max())
mag_range = st.sidebar.slider("Select Magnitude Range", min_value=min_mag, max_value=max_mag,
                               value=(min_mag, max_mag), step=0.1)
min_depth, max_depth = float(data['depth'].min()), float(data['depth'].max())
depth_range = st.sidebar.slider("Select Depth Range (km)", min_value=min_depth, max_value=max_depth,
                                value=(min_depth, max_depth), step=1.0)
if "region" in data.columns:
    regions_available = data['region'].unique().tolist()
    selected_regions = st.sidebar.multiselect("Choose Regions", options=regions_available, default=regions_available)
else:
    selected_regions = None
if data['time'].notnull().any():
    min_date, max_date = data['time'].min(), data['time'].max()
    date_range = st.sidebar.date_input("Pick a Date Range", value=(min_date, max_date),
                                       min_value=min_date, max_value=max_date)
else:
    date_range = None

# Filter data based on selections
with st.spinner("Filtering data for you..."):
    filtered_data = data[
        (data['magnitude'] >= mag_range[0]) & (data['magnitude'] <= mag_range[1]) &
        (data['depth'] >= depth_range[0]) & (data['depth'] <= depth_range[1])
    ]
    if date_range:
        filtered_data = filtered_data[
            (filtered_data['time'] >= pd.to_datetime(date_range[0])) &
            (filtered_data['time'] <= pd.to_datetime(date_range[1]))
        ]
    if selected_regions is not None:
        filtered_data = filtered_data[filtered_data['region'].isin(selected_regions)]

st.markdown("### Here's Your Filtered Earthquake Data")
st.write(f"Displaying {len(filtered_data)} records. Enjoy exploring!")
st.dataframe(filtered_data)

# --- Multi-Tab Layout ---
tabs = st.tabs(["Map", "Charts", "Cluster Analysis", "Statistics", "Download Report"])

###########################
# Tab 1: Interactive Map
with tabs[0]:
    st.subheader("Explore Earthquakes on the Map")
    if not filtered_data.empty:
        map_layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_data,
            get_position=['longitude', 'latitude'],
            get_color='[255, 0, 0, 140]',
            get_radius='magnitude * 10000',
            pickable=True,
        )
        view_state = pdk.ViewState(
            latitude=filtered_data['latitude'].mean(),
            longitude=filtered_data['longitude'].mean(),
            zoom=1,
            pitch=0
        )
        deck = pdk.Deck(
            layers=[map_layer],
            initial_view_state=view_state,
            tooltip={"text": "Magnitude: {magnitude}\nDepth: {depth} km"}
        )
        st.pydeck_chart(deck)
    else:
        st.warning("Oh no! There's no data available for the map at the moment.")

###########################
# Tab 2: Interactive Charts
with tabs[1]:
    st.subheader("Dive into Interactive Charts")
    hist_fig = px.histogram(filtered_data, x='magnitude', nbins=20,
                            title="Distribution of Earthquake Magnitudes")
    st.plotly_chart(hist_fig, use_container_width=True)
    
    scatter_fig = px.scatter(filtered_data, x='depth', y='magnitude',
                             title="How Depth and Magnitude Relate",
                             labels={"depth": "Depth (km)", "magnitude": "Magnitude"})
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    if not filtered_data.empty:
        ts_data = filtered_data.set_index('time').resample('W').size().reset_index(name='count')
        line_fig = px.line(ts_data, x='time', y='count', title="Weekly Earthquake Frequency")
        st.plotly_chart(line_fig, use_container_width=True)

###########################
# Tab 3: Cluster Analysis
with tabs[2]:
    st.subheader("Clustering Analysis")
    if filtered_data.empty:
        st.warning("It looks like there's no data to cluster right now.")
    else:
        algorithm = st.radio("Select Clustering Algorithm", ("DBSCAN", "KMeans"))
        coords = filtered_data[['latitude', 'longitude']]
        
        if algorithm == "DBSCAN":
            eps_km = st.slider("Max distance (km)", 10, 1000, 100)
            min_samples = st.slider("DBSCAN min_samples", min_value=1, max_value=20, value=5, step=1)
            try:
                # Convert to radians and use Haversine distance
                coords_rad = np.radians(coords)
                earth_radius_km = 6371
                cluster_model = DBSCAN(
                    eps=eps_km/earth_radius_km, 
                    min_samples=min_samples, 
                    metric='haversine'
                ).fit(coords_rad)
                cluster_labels = cluster_model.labels_
            except Exception as e:
                st.error("Error with DBSCAN: " + str(e))
                cluster_labels = np.array([-1]*len(filtered_data))
        else:
            num_clusters = st.slider("KMeans - Number of Clusters", min_value=1, max_value=10, value=3, step=1)
            try:
                cluster_model = KMeans(n_clusters=num_clusters, random_state=42).fit(coords)
                cluster_labels = cluster_model.labels_
            except Exception as e:
                st.error("Error with KMeans: " + str(e))
                cluster_labels = np.array([0]*len(filtered_data))
        
        filtered_data['cluster'] = cluster_labels
        
        def compute_color(label):
            if label == -1:
                return [200, 200, 200, 140]  # For noise in DBSCAN
            else:
                return [int((label * 50) % 256), int((label * 80) % 256), 150, 140]
        filtered_data['color'] = filtered_data['cluster'].apply(compute_color)
        
        cluster_layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered_data,
            get_position=['longitude', 'latitude'],
            get_color='color',
            get_radius='magnitude * 10000',
            pickable=True,
        )
        view_state_cluster = pdk.ViewState(
            latitude=filtered_data['latitude'].mean(),
            longitude=filtered_data['longitude'].mean(),
            zoom=1,
            pitch=0
        )
        deck_cluster = pdk.Deck(
            layers=[cluster_layer],
            initial_view_state=view_state_cluster,
            tooltip={"text": "Cluster: {cluster}\nMagnitude: {magnitude}\nDepth: {depth} km"}
        )
        st.pydeck_chart(deck_cluster)
        st.write("**Cluster Summary:**")
        st.dataframe(filtered_data['cluster'].value_counts().rename_axis("Cluster").reset_index(name="Counts"))
        
        cluster_csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Clustered Data as CSV",
            data=cluster_csv,
            file_name="clustered_earthquake_data.csv",
            mime="text/csv"
        )

###########################
# Tab 4: Descriptive Statistics
with tabs[3]:
    st.subheader("Statistics & Insights")
    st.write("**Here's a quick summary of your data:**")
    st.write(filtered_data.describe())
    
    # Correlation analysis
    if not filtered_data.empty:
        corr, p_value = pearsonr(filtered_data['depth'], filtered_data['magnitude'])
        st.write(f"**Correlation between Depth and Magnitude:** {corr:.2f} (p-value: {p_value:.4f})")
    
    num_cols = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if num_cols:
        corr = filtered_data[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numerical columns available for correlation analysis.")

###########################
# Tab 5: Download Report
with tabs[4]:
    st.subheader("Save Your Report")
    if not filtered_data.empty:
        csv_report = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
             label="Download Filtered Data as CSV",
             data=csv_report,
             file_name="earthquake_report.csv",
             mime="text/csv"
        )
    else:
        st.info("There's no data available to download right now.")