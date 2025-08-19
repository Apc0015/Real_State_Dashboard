# 🏠 Real Estate Dashboard

A comprehensive real estate price analysis and prediction dashboard built with Streamlit and machine learning.

## Features

- **Data Exploration**: Interactive visualizations of the real estate dataset
- **Price Analysis**: Geographic and temporal price analysis
- **Machine Learning**: Random Forest model for price prediction
- **Interactive Prediction**: Real-time price prediction based on property features
- **Dashboard Navigation**: Multi-page dashboard with different analysis views

## Dataset

The dashboard uses a real estate dataset with the following features:
- Transaction date
- House age
- Distance to the nearest MRT station
- Number of convenience stores
- Latitude and Longitude
- House price of unit area (target variable)

## Installation

1. Clone the repository or download the files
2. Install the required dependencies:
```bash
pip install -r requirement.txt
```

## Usage

### Local Development
Run the Streamlit dashboard locally:
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Docker Deployment
Build and run using Docker:
```bash
# Build the Docker image
docker build -t real-estate-dashboard .

# Run the container
docker run -p 8501:8501 real-estate-dashboard
```

## Dashboard Pages

1. **Overview**: Dataset summary and basic statistics
2. **Data Exploration**: Interactive data visualization and correlation analysis
3. **Price Analysis**: Geographic and temporal price analysis
4. **ML Model**: Machine learning model performance and feature importance
5. **Price Prediction**: Interactive price prediction tool

## Deployment Options

### Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy the app

### Heroku
1. Create a `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

### Local Network Access
To access the dashboard from other devices on your network:
```bash
streamlit run app.py --server.address=0.0.0.0
```

## Project Structure
```
Real_state_dashboard/
├── app.py                 # Main Streamlit application
├── Real_Estate.csv        # Dataset
├── requirement.txt        # Python dependencies
├── Dockerfile            # Docker configuration
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## Technologies Used

- **Frontend**: Streamlit
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn (Random Forest)
- **Deployment**: Docker, Streamlit Cloud

## Model Performance

The dashboard uses a Random Forest Regressor with the following performance metrics:
- Feature engineering for better predictions
- Cross-validation for model reliability
- Interactive prediction interface

## Contributing

Feel free to fork this project and submit pull requests for any improvements.

## License

This project is open source and available under the MIT License.