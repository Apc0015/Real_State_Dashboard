import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Real Estate Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('Real_Estate.csv')
        return df
    except FileNotFoundError:
        st.error("Real_Estate.csv file not found. Please ensure the file is in the same directory.")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Convert Transaction date to datetime
    df['Transaction date'] = pd.to_datetime(df['Transaction date'])
    
    # Extract temporal features
    df['year'] = df['Transaction date'].dt.year
    df['month'] = df['Transaction date'].dt.month
    df['day'] = df['Transaction date'].dt.day
    
    # Feature Engineering
    df['distance_category'] = pd.cut(df['Distance to the nearest MRT station'], 
                                    bins=[0, 500, 1000, 2000, float('inf')], 
                                    labels=['Very Close', 'Close', 'Moderate', 'Far'])
    
    df['convenience_score'] = df['Number of convenience stores'] / df['Number of convenience stores'].max()
    
    # Location features
    df['lat_zone'] = pd.cut(df['Latitude'], bins=5, labels=['South', 'South-Mid', 'Central', 'North-Mid', 'North'])
    df['long_zone'] = pd.cut(df['Longitude'], bins=5, labels=['West', 'West-Mid', 'Central', 'East-Mid', 'East'])
    
    return df

@st.cache_resource
def train_model(df):
    """Train the machine learning model"""
    # Prepare features
    numerical_features = ['House age', 'Distance to the nearest MRT station', 
                         'Number of convenience stores', 'Latitude', 'Longitude', 
                         'year', 'month', 'day', 'convenience_score']
    
    categorical_features = pd.get_dummies(df[['distance_category', 'lat_zone', 'long_zone']])
    
    X = pd.concat([df[numerical_features], categorical_features], axis=1)
    y = df['House price of unit area']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, X.columns, {'RMSE': rmse, 'MAE': mae, 'R²': r2}, y_test, y_pred

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Real Estate Price Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "Data Exploration", "Price Analysis", "ML Model", "Price Prediction"]
    )
    
    if page == "Overview":
        show_overview(df)
    elif page == "Data Exploration":
        show_data_exploration(df)
    elif page == "Price Analysis":
        show_price_analysis(df)
    elif page == "ML Model":
        show_ml_model(df)
    elif page == "Price Prediction":
        show_prediction(df)

def show_overview(df):
    st.header("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(df))
    with col2:
        st.metric("Average Price", f"${df['House price of unit area'].mean():.2f}")
    with col3:
        st.metric("Price Range", f"${df['House price of unit area'].min():.2f} - ${df['House price of unit area'].max():.2f}")
    with col4:
        st.metric("Average House Age", f"{df['House age'].mean():.1f} years")
    
    st.subheader("Dataset Sample")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Missing Values:**")
        missing = df.isnull().sum()
        st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values")
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes)

def show_data_exploration(df):
    st.header("🔍 Data Exploration")
    
    # Distribution plots
    st.subheader("Price Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(df, x='House price of unit area', nbins=30, 
                           title="House Price Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.histogram(df, x=np.log1p(df['House price of unit area']), nbins=30,
                           title="Log-Transformed Price Distribution")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                     title="Correlation Matrix")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Scatter plots
    st.subheader("Feature Relationships")
    feature = st.selectbox("Select feature to plot against price:", 
                          ['House age', 'Distance to the nearest MRT station', 
                           'Number of convenience stores', 'Latitude', 'Longitude'])
    
    fig4 = px.scatter(df, x=feature, y='House price of unit area',
                      title=f"Price vs {feature}")
    st.plotly_chart(fig4, use_container_width=True)

def show_price_analysis(df):
    st.header("💰 Price Analysis")
    
    # Price by distance category
    st.subheader("Price by Distance to MRT")
    fig1 = px.box(df, x='distance_category', y='House price of unit area',
                  title="House Price by Distance Category")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Price by convenience stores
    st.subheader("Price by Number of Convenience Stores")
    fig2 = px.scatter(df, x='Number of convenience stores', y='House price of unit area',
                      size='convenience_score', color='distance_category',
                      title="Price vs Convenience Stores")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Geographic analysis
    st.subheader("Geographic Price Distribution")
    fig3 = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', 
                            color='House price of unit area',
                            size='House price of unit area',
                            hover_data=['House age', 'Number of convenience stores'],
                            mapbox_style="open-street-map",
                            title="Property Locations and Prices")
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Time series analysis
    st.subheader("Price Trends Over Time")
    monthly_prices = df.groupby(['year', 'month'])['House price of unit area'].mean().reset_index()
    monthly_prices['date'] = pd.to_datetime(monthly_prices[['year', 'month']].assign(day=1))
    
    fig4 = px.line(monthly_prices, x='date', y='House price of unit area',
                   title="Average Price Trends Over Time")
    st.plotly_chart(fig4, use_container_width=True)

def show_ml_model(df):
    st.header("🤖 Machine Learning Model")
    
    # Train model
    model, scaler, feature_names, metrics, y_test, y_pred = train_model(df)
    
    # Model metrics
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RMSE", f"{metrics['RMSE']:.3f}")
    with col2:
        st.metric("MAE", f"{metrics['MAE']:.3f}")
    with col3:
        st.metric("R² Score", f"{metrics['R²']:.3f}")
    
    # Actual vs Predicted
    st.subheader("Actual vs Predicted Prices")
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    fig1 = px.scatter(results_df, x='Actual', y='Predicted',
                      title="Actual vs Predicted Prices")
    fig1.add_shape(type="line", x0=results_df['Actual'].min(), 
                   y0=results_df['Actual'].min(),
                   x1=results_df['Actual'].max(), 
                   y1=results_df['Actual'].max(),
                   line=dict(dash="dash", color="red"))
    st.plotly_chart(fig1, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig2 = px.bar(importance_df, x='Importance', y='Feature', 
                  orientation='h', title="Top 10 Feature Importances")
    st.plotly_chart(fig2, use_container_width=True)

def show_prediction(df):
    st.header("🔮 Price Prediction")
    
    # Train model
    model, scaler, feature_names, metrics, _, _ = train_model(df)
    
    st.subheader("Enter Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        house_age = st.slider("House Age (years)", 0.0, 50.0, 10.0)
        mrt_distance = st.slider("Distance to MRT (m)", 0.0, 6000.0, 1000.0)
        convenience_stores = st.slider("Number of Convenience Stores", 0, 10, 5)
        latitude = st.slider("Latitude", 24.93, 25.02, 24.97)
    
    with col2:
        longitude = st.slider("Longitude", 121.47, 121.57, 121.52)
        year = st.selectbox("Transaction Year", [2012, 2013])
        month = st.slider("Transaction Month", 1, 12, 6)
        day = st.slider("Transaction Day", 1, 31, 15)
    
    if st.button("Predict Price", type="primary"):
        # Create input data
        input_data = pd.DataFrame({
            'House age': [house_age],
            'Distance to the nearest MRT station': [mrt_distance],
            'Number of convenience stores': [convenience_stores],
            'Latitude': [latitude],
            'Longitude': [longitude],
            'year': [year],
            'month': [month],
            'day': [day]
        })
        
        # Add engineered features
        input_data['convenience_score'] = convenience_stores / df['Number of convenience stores'].max()
        
        # Add categorical features (simplified)
        if mrt_distance <= 500:
            distance_cat = 'Very Close'
        elif mrt_distance <= 1000:
            distance_cat = 'Close'
        elif mrt_distance <= 2000:
            distance_cat = 'Moderate'
        else:
            distance_cat = 'Far'
        
        # Create dummy variables for categorical features
        for cat in ['Very Close', 'Close', 'Moderate', 'Far']:
            input_data[f'distance_category_{cat}'] = 1 if distance_cat == cat else 0
        
        # Add other categorical dummies (simplified)
        for zone in ['South', 'South-Mid', 'Central', 'North-Mid', 'North']:
            input_data[f'lat_zone_{zone}'] = 0
        input_data['lat_zone_Central'] = 1  # Default to central
        
        for zone in ['West', 'West-Mid', 'Central', 'East-Mid', 'East']:
            input_data[f'long_zone_{zone}'] = 0
        input_data['long_zone_Central'] = 1  # Default to central
        
        # Ensure all features are present
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data[feature_names]
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        st.success(f"🏠 Predicted House Price: **${prediction:.2f}** per unit area")
        
        # Show confidence interval (simplified)
        st.info(f"📊 Model R² Score: {metrics['R²']:.3f} (Higher is better)")

if __name__ == "__main__":
    main()