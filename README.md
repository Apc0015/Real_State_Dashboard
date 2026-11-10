# 🏠 Real Estate Dashboard & Valuation Engine

A full-featured Streamlit application and machine learning pipeline for exploratory data analysis, geospatial visualization, and property price prediction.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://realstatedashboard-sqqfjc8adbm7tps8pfujii.streamlit.app/)
[![CI](https://github.com/Apc0015/Real_State_Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/Apc0015/Real_State_Dashboard/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌐 Live Demo

You can interact with the live, deployed dashboard here:
👉 **[Launch Real Estate Dashboard on Streamlit Cloud](https://realstatedashboard-sqqfjc8adbm7tps8pfujii.streamlit.app/)**

---

## Overview

I built this project to analyze property valuation drivers and provide an end-to-end interactive valuation tool. Using historical transaction data from Sindian District (New Taipei City, Taiwan), I built an automated feature engineering pipeline, benchmarked multiple regression models, and packaged the entire workflow into a multi-page Streamlit application.

The application allows users to explore price distributions, inspect geospatial patterns with OpenStreetMap, study feature correlations, evaluate model metrics, and generate real-time price predictions with an interactive slider interface.

---

## The Dataset

The project uses historical real estate transaction records (`Real_Estate.csv`) containing 414 residential property sales:

| Column | Description | Type |
|---|---|---|
| `Transaction date` | Date of the transaction (2012–2013) | Datetime |
| `House age` | Age of the property in years | Numeric (0.0 to 43.8 years, mean ~18.4) |
| `Distance to the nearest MRT station` | Distance to mass transit in meters | Numeric (23.4m to 6,488.0m) |
| `Number of convenience stores` | Number of convenience stores in the local circle | Integer (0 to 10 stores) |
| `Latitude` | Geographic coordinate | Float (24.93° to 25.01° N) |
| `Longitude` | Geographic coordinate | Float (121.47° to 121.57° E) |
| `House price of unit area` | Target variable: property price per unit area | Numeric ($0.00 to $65.57, mean ~$29.10) |

---

## Feature Engineering Pipeline

Raw features alone don't capture non-linear market behaviors like transit access tiers or neighborhood clustering. In `app.py` (`preprocess_data` and `prepare_features`), I engineered the following feature sets:

1. **Temporal Features**:
   - Extracted `year`, `month`, and `day` from transaction timestamps to track macroeconomic trends and seasonal market movement.
2. **Transit Proximity Tiers (`distance_category`)**:
   - Binned transit distance into four operational tiers:
     - `Very Close`: $0 \le d \le 500\text{ m}$ (direct walking access)
     - `Close`: $500 < d \le 1000\text{ m}$
     - `Moderate`: $1000 < d \le 2000\text{ m}$
     - `Far`: $d > 2000\text{ m}$
3. **Commercial Amenity Density (`convenience_score`)**:
   - Normalized convenience store counts relative to the maximum observed count in the district:
     $$\text{convenience\_score} = \frac{\text{stores}}{\max(\text{stores})}$$
4. **Spatial Discretization (`lat_zone`, `long_zone`)**:
   - Discretized latitude and longitude into 5 equal-width zones each (`South` to `North`, `West` to `East`) to capture localized geographic micro-markets.
5. **Categorical Dummy Encoding**:
   - One-hot encoded all categorical bins (`distance_category`, `lat_zone`, `long_zone`), yielding a 23-column numeric feature matrix for scaling and regression modeling.

---

## Machine Learning Models & Results

During my initial research in `Real state.ipynb`, I benchmarked multiple regressors on an 80/20 train/test split:

| Model | Test RMSE | Test MAE | Test $R^2$ Score |
|---|---|---|---|
| Linear Regression (OLS) | $11.29 | ~$9.42 | 0.539 |
| Ridge Regression ($\alpha=1.0$) | $11.28 | ~$9.41 | 0.540 |
| Lasso Regression ($\alpha=1.0$) | $11.19 | ~$9.35 | 0.547 |
| Random Forest (100 Trees Baseline) | $11.50 | $9.67 | 0.522 |
| Gradient Boosting (100 Trees) | $11.85 | ~$9.88 | 0.493 |
| Tuned Random Forest (GridSearchCV, 300 Trees) | $11.44 | $9.73 | 0.527 |

### Why Random Forest in the Application?
I deployed a **Random Forest Regressor** (100 estimators, random state 42) coupled with `StandardScaler` inside the Streamlit app. Even though linear models achieved slightly lower test RMSE on this small sample, Random Forest naturally captures non-linear price cliffs (e.g., properties beyond 1,000m dropping off non-linearly) and feature interactions between building age and location without forcing artificial linearity.

Given 414 historical sales and real-world micro-market variance, an $R^2$ of ~0.52 is a realistic reflection of the signal present in transit, age, and store density. The feature importances show that **Distance to the nearest MRT station** and **House age** drive the overwhelming majority of valuation variance.

---

## Dashboard Pages

The Streamlit dashboard is organized into 5 dedicated views:

1. **Overview**: Key metrics (total properties, mean unit price, price range, average property age), preview table, and missing-value checks.
2. **Data Exploration**: Distribution histograms (standard and log-transformed price distributions), full correlation matrix heatmap, and customizable bivariate scatter plots.
3. **Price Analysis**: Price distributions by transit distance category (box plots), price vs. convenience store density, geospatial price map on OpenStreetMap (`plotly.express.scatter_mapbox`), and monthly price trend lines.
4. **ML Model Performance**: Live test-set metrics (RMSE, MAE, $R^2$), actual vs. predicted price scatter plot with a 45-degree parity line, and horizontal feature importance ranking.
5. **Price Prediction**: Interactive valuation calculator where users adjust house age, transit distance, convenience stores, latitude, longitude, and date sliders to receive an instant predicted unit price.

---

## Repository Structure

```
Real_State_Dashboard/
├── app.py                     # Streamlit web application & ML pipeline
├── Real_Estate.csv            # Property transaction dataset (414 rows)
├── Real state.ipynb           # Research notebook (EDA, model comparison, tuning)
├── requirements.txt           # Python application & test dependencies
├── requirement.txt           # Deployment compatibility requirements
├── pytest.ini                 # Pytest test runner configuration
├── Dockerfile                 # Container image specification (Python 3.11-slim)
├── .env.example               # Template environment configuration
├── .devcontainer/             # VS Code Dev Container configuration
│   └── devcontainer.json
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow (Python 3.10, 3.11, 3.12)
├── tests/
│   ├── __init__.py
│   └── test_dashboard.py      # Automated unit & integration tests
└── README.md                  # Project documentation
```

---

## Local Development Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/Apc0015/Real_State_Dashboard.git
cd Real_State_Dashboard
```

### 3. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Application Locally
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Running Automated Tests

I wrote an automated test suite with `pytest` in `tests/test_dashboard.py` covering data loading, feature preprocessing, model training, metric validity, and prediction edge cases.

To execute the test suite:
```bash
pytest tests/ -v
```

All 15 tests verify:
- Complete dataset loading and environment variable override support.
- Temporal extraction and distance binning logic.
- Convenience score normalization within $[0, 1]$.
- Spatial zone discretization into 5 categories.
- 23-column feature matrix assembly and lack of NaNs.
- Model artifact generation, RMSE/MAE/R² finite bounds, and feature importance distributions.
- Prediction input building across distance thresholds (`Very Close`, `Close`, `Moderate`, `Far`).
- Transit distance sensitivity (properties closer to transit predict higher valuations).
- Deterministic inference across repeated calls.

---

## Docker Deployment

You can containerize and run the application locally or on any cloud server:

```bash
# Build the Docker image
docker build -t real-estate-dashboard .

# Run the container on port 8501
docker run -d -p 8501:8501 --name real-estate-app real-estate-dashboard
```

The container includes a built-in healthcheck at `http://localhost:8501/_stcore/health`.

---

## Technologies Used

- **Application Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Random Forest Regressor, StandardScaler, Train-Test Split)
- **Interactive Visualizations**: [Plotly Express & Graph Objects](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Testing & Quality Assurance**: [Pytest](https://docs.pytest.org/), [GitHub Actions CI](https://github.com/features/actions)
- **Containerization**: [Docker](https://www.docker.com/)

---

## License

This project is open-source and available under the terms of the [MIT License](LICENSE).
