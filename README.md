<div align="center">
  <img src="https://github.com/01Ruwantha/Heart-Disease-Risk-Assessment/blob/36a00fd2ef5cfed3f8f2102ec0d5f103227f3bc2/Pictures/Heart.jpg" alt="Heart pic" height="500px"  />
</div>

<h1>Heart Disease Risk Assessment System</h1> 

## Introduction
The Heart Disease Risk Assessment System is a web-based application that leverages machine learning to predict the risk of heart disease. This system aims to provide real-time predictions with a user-friendly interface, making heart disease risk assessment more accessible.

## Objective
The main objective of this project is to develop a predictive system using machine learning algorithms to assess heart disease risk efficiently.

## Key Highlights
- *Data-driven insights*
- *Real-time predictions*
- *User-friendly access*

## Problem Statement
### Importance
Heart disease is one of the leading causes of death worldwide. Early detection plays a crucial role in saving lives.

### Challenges
Developing accurate, scalable, and user-friendly systems is essential, as existing solutions lack accessibility.

## Dataset and Features
- *Dataset Source:* UCI Heart Disease Dataset
- *Key Features:*
  - Age
  - Sex
  - Chest pain type
  - Cholesterol
  - Maximum heart rate achieved (thalach)
  - And more

## Machine Learning Models
The following machine learning models were implemented to predict heart disease risk:
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)

### Evaluation Metrics
Model performance was evaluated using:
- Accuracy
- ROC-AUC score
- Cross-validation

## Hyperparameter Tuning
GridSearchCV was used to optimize model parameters such as:
- n_estimators
- max_depth
- Other relevant hyperparameters

## Feature Importance
Significant features that influence prediction include:
- Age
- Chest pain type
- Maximum heart rate achieved (thalach)

A horizontal bar chart is used to visualize feature importance.

## Flask Web Application
### Backend
- Developed using Flask, serving the machine learning model and providing an API endpoint for predictions.

### Frontend
- Built with HTML/CSS, offering a user-friendly interface with an input form for user data and result display.

## Challenges and Future Work
### Challenges
- Dataset limitations
- Balancing model performance with interpretability

### Future Enhancements
- Deploying the system on a cloud platform
- Exploring deep learning models
- Further optimization of the model

## How to Run the Project
1. Clone the repository.
2. Install dependencies using:
   bash
   pip install -r requirements.txt
   
3. Run the Flask application:
   bash
   python app.py
   
4. Access the web interface via http://localhost:5000.

## Team Contributions
Each team member contributed to different aspects of the project, including data preprocessing, model implementation, and web development.
