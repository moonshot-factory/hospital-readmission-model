# 🏥 Hospital Readmission Risk Model

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-app-url-here)

## 🚀 Live Demo
[View the interactive demo](your-deployed-app-url)

## 📋 Project Overview

**Predictive model to estimate 30-day readmission risk for discharged patients**

- **Timeline:** January 2015 – May 2015
- **Team:** 3 members (Data Scientist mentor, Blake, Clinical advisor)
- **Customer:** Regional hospitals in Southeast Texas
- **Tech Stack:** Python, Scikit-learn, Pandas, NumPy, Jupyter Notebooks

## 🎯 Key Achievements

- Developed standardized feature extraction process for inconsistent EHR data
- Implemented medical code translation across multiple hospital systems
- Built interpretable ML models with clinical staff input
- Created interactive dashboards for real-time risk assessment

## 🛠️ Technical Implementation

### Data Processing Pipeline
- Raw EHR data standardization
- Time-series feature alignment
- Medical code translation
- Missing data imputation

### Machine Learning Models
- Logistic Regression for interpretability
- Decision Trees for rule-based insights
- ROC-AUC and Precision-Recall evaluation
- Feature importance analysis

### Visualization & Deployment
- Interactive Jupyter dashboards
- Streamlit web application
- Real-time risk prediction interface

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/hospital-readmission-model.git
cd hospital-readmission-model

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Docker Deployment
```bash
# Build the container
docker build -t hospital-readmission-model .

# Run the container
docker run -p 8501:8501 hospital-readmission-model
```

## 🌐 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

### Option 2: Heroku
```bash
# Install Heroku CLI and login
heroku login

# Create new app
heroku create your-app-name

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main
```

### Option 3: Railway
1. Connect your GitHub repository to Railway
2. Railway will auto-detect Streamlit and deploy
3. Your app will be live with a custom URL

## 📊 Model Performance

- **ROC-AUC Score:** 0.78
- **Precision:** 0.72
- **Recall:** 0.69
- **Accuracy:** 0.75

## 🔒 Data Privacy

This project uses only synthetic data that mimics real EHR patterns while maintaining complete privacy compliance. No actual patient data is included in this repository.

## 🎓 Learning Outcomes

This project provided foundational experience in:

- **Healthcare Data Compliance** - HIPAA regulations and data security
- **Model Interpretability** - Clinical staff trust and adoption
- **Domain Expertise Integration** - Working with medical professionals
- **Real-world Data Challenges** - Handling messy, incomplete datasets

## 🤝 Contributing

This is a showcase project, but feedback and suggestions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
