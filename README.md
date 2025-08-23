# OpenMind

A Machine Learning Approach to Big Five Personality Prediction

## Overview

OpenMind predicts Big Five personality traits (Extraversion, Neuroticism, Agreeableness, Conscientiousness, Openness) using responses to 20 short questions. It leverages modern machine learning techniques and provides an interactive Streamlit web app for predictions and visualization.

## Features

- Data cleaning, feature engineering, and model training in Jupyter notebooks
- Per-trait models using LightGBM and Random Forest
- Scalers for target normalization
- Interactive Streamlit app for personality prediction
- Radar chart and detailed results display
- Modular code and reproducible workflow

## Directory Structure

```
OpenMind/
├── app/
│   ├── streamlit_app.py         # Streamlit web app
│   └── questions.json           # Questionnaire
├── bigfive-ml/                  # Python virtual environment
├── data/
│   ├── raw/                     # Raw data
│   ├── interim/                 # Cleaned/preprocessed data
│   └── processed/               # Feature engineered data
├── LightGBModel/
│   └── lgbm_20q_scaled/
│       ├── lgbm_<TRAIT>.pkl     # LightGBM models per trait
│       ├── scaler_<TRAIT>.pkl   # Scalers per trait
│       └── training_results.csv # Training metrics
├── models/
│   ├── rf_regression_model.pkl  # Random Forest regression model
│   └── rf_classification_model.pkl # Random Forest classification model
├── notebooks/                   # Jupyter notebooks for all steps
│   ├── ITBIN-2211-0122-feature_engineering.ipynb
│   ├── ITBIN-2211-0278-training_save_model.ipynb
│   ├── ITBIN-2211-0312-data-exploration.ipynb
│   └── ITBIN-2211-0324-Preprocessing&Cleaning.ipynb
├── requirements.txt             # Python dependencies
└── README.md
```

## Quickstart

1. **Clone the repository:**
   ```powershell
   git clone <your-repo-url>
   cd OpenMind
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Activate the virtual environment:**
   ```powershell
   .\bigfive-ml\Scripts\Activate.ps1
   ```
4. **Run the Streamlit app:**
   ```powershell
   streamlit run app/streamlit_app.py
   ```
5. **Open the app in your browser:**
   - Local: http://localhost:8501

## How It Works

- User answers 20 questions in the web app
- Answers are mapped to model features
- Each trait is predicted using a trained LightGBM model and scaler
- Results are visualized and summarized with interactive charts

## Notebooks

- All steps from data cleaning to model training are documented in the `notebooks/` folder for full reproducibility.

## Requirements

- Python 3.12+
- See `requirements.txt` for all dependencies (streamlit, pandas, numpy, scikit-learn, lightgbm, plotly, etc.)

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features. Follow standard Python and ML best practices:

- Use clear, modular code and docstrings
- Add tests for new functionality
- Document changes in the README

## License

This project is licensed under the MIT License.

## Authors

- Project by OpenMind Team
- See commit history for contributors
