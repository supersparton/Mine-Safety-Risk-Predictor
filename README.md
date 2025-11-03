# ⛏️ MSHA Mine Safety Risk Predictor

An AI-powered machine learning system that predicts workplace accident severity and lost workdays using historical MSHA (Mine Safety and Health Administration) accident data.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)


## 🎯 Overview

This project uses advanced machine learning techniques to predict:
1. **Injury Severity Classification**: Categorizes accidents into severity levels (No Days Away, Days Away, Restricted Work, Disability, Fatality)
2. **Lost Workdays Regression**: Predicts the number of workdays lost due to accidents

The system helps mining companies:
- ✅ Assess risk before accidents occur
- ✅ Allocate safety resources effectively
- ✅ Estimate compensation and planning needs
- ✅ Identify high-risk scenarios for intervention

## ✨ Features

### Machine Learning Models
- **XGBoost Classifier** for injury severity prediction
- **XGBoost Regressor** for lost workdays estimation
- Advanced feature engineering (temporal features, experience levels, interaction terms)
- Comprehensive preprocessing pipeline (imputation, encoding, scaling)

### Interactive Web Application
- User-friendly Streamlit interface
- Real-time predictions with confidence scores
- Risk assessment with color-coded alerts
- Personalized safety recommendations
- Model performance metrics and documentation

### Data Analysis
- Exploratory Data Analysis (EDA) with 200,000+ records
- Feature importance analysis using SHAP values
- Model interpretability and explainability
- Comprehensive visualizations

## 📊 Dataset

**Source**: MSHA (Mine Safety and Health Administration)

**Size**: 200,000+ accident records

**Features**:
- **Numerical**: Total experience, mine experience, job experience, occupation code, number of injuries
- **Categorical**: Accident type, subunit, classification, coal/metal indicator
- **Engineered**: Experience level, cyclical time features, restriction days, schedule charge

**Target Variables**:
- **Classification**: DEGREE_INJURY (5 classes)
- **Regression**: DAYS_LOST (continuous)

## 🤖 Models

### Classification Model
- **Algorithm**: XGBoost Classifier
- **Purpose**: Predict injury severity
- **Classes**: 
  - No Days Away From Work
  - Days Away From Work Only
  - Days of Restricted Work Activity
  - Permanent Total/Partial Disability
  - Fatality
- **Performance**: High precision with balanced accuracy

### Regression Model
- **Algorithm**: XGBoost Regressor
- **Purpose**: Predict lost workdays
- **Transformation**: Log-transformed target for better accuracy
- **Metrics**: R² score, RMSE, MAE, RMSLE
- **Performance**: Optimized through hyperparameter tuning and feature engineering

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ml-project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
category-encoders>=2.6.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
shap>=0.44.0
joblib>=1.3.0
streamlit>=1.29.0
scipy>=1.11.0
```

## 💻 Usage

1. Enter accident details in the form:
   - Mine information (Mine ID, Operator ID, Type)
   - Worker experience (Total, Mine-specific, Job-specific)
   - Accident details (Type, Classification, Occupation Code)

2 Click **"🔮 Predict Risk & Lost Workdays"** to get:
   - Predicted injury severity with confidence
   - Estimated lost workdays
   - Risk assessment (Low/Moderate/High)
   - Personalized safety recommendations


## 🛠️ Technologies Used

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting models

### Visualization
- **matplotlib**: Static plots
- **seaborn**: Statistical visualizations
- **plotly**: Interactive charts
- **SHAP**: Model explainability

### Web Framework
- **Streamlit**: Interactive web application

### Feature Engineering
- **category-encoders**: Advanced categorical encoding (Target Encoding)
- **scipy**: Statistical tests and transformations

## 📈 Results

### Model Performance

#### Classification Model
- **Accuracy**: High precision across all severity classes
- **Multi-class Classification**: Balanced performance for imbalanced data
- **Feature Importance**: Experience metrics and accident type most predictive

#### Regression Model
- **R² Score**: Optimized through advanced techniques
- **RMSLE**: Low error for workdays prediction
- **Generalization**: Robust performance on test data
- **Improvements Applied**:
  - Optimal target transformation (Box-Cox/Yeo-Johnson/Log)
  - IQR-based outlier removal (~18% outliers removed)
  - 12 engineered features (experience ratios, severity scores, temporal features)
  - Mutual information feature selection
  - Weighted voting ensemble (XGBoost + RandomForest + HistGradientBoosting)

### Key Insights
1. **Experience Matters**: Workers with less than 2 years experience have significantly higher accident severity
2. **Accident Type Impact**: Machinery-related accidents result in longer recovery times
3. **Temporal Patterns**: Weekend and night shifts show elevated risk profiles
4. **Prevention Focus**: Early intervention for inexperienced workers reduces severe outcomes

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request




<div align="center">
  
**⛏️ Making mines safer through AI-powered predictions**

Made with ❤️ using Python, XGBoost, and Streamlit

</div>
