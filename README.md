# 🚦 Traffic Prediction

Welcome to the Traffic Prediction project! This project uses machine learning to predict traffic congestion. We look at factors like time of day, day of the week, weather, and the number of vehicles. Our goal is to help city residents and commuters by providing insights for better traffic management and planning.


## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Methodology](#methodology)
4. [Experimentation](#experimentation)
5. [Algorithms](#algorithms)
6. [Implementation](#implementation)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [How to Run](#HowtoRun)
10. [Conclusion](#conclusion)

## 📝 Introduction
The Traffic Prediction project leverages a machine learning approach to analyze traffic data and predict congestion levels. By understanding the patterns and factors contributing to traffic, we aim to provide actionable insights for efficient traffic management.

## 📊 Dataset Description
The dataset used for this project is sourced from Kaggle. It contains 5952 data points and includes the following features:
- **Time:** The hour of data collection (converted to datetime format).
- **Date:** The date of data collection.
- **Day of the Week:** The day on which data was collected.
- **CarCount:** Number of cars detected in a 15-minute interval.
- **BikeCount:** Number of bikes detected in a 15-minute interval.
- **BusCount:** Number of buses detected in a 15-minute interval.
- **TruckCount:** Number of trucks detected in a 15-minute interval.
- **Total:** Total number of vehicles detected in a 15-minute interval.
- **Traffic Situation:** Classified into four categories: Heavy, High, Normal, Low.
- **Weather:** Weather conditions categorized as "Sunny", "Neutral", and "Foggy".
- **Is Peak Hour:** A binary indicator for peak traffic hours (6-10 AM and 3-9 PM).

## ⚙️ Methodology
### Data Preprocessing
- **Encoding:** Categorical features were encoded using `LabelEncoder` from Scikit-learn.
- **Feature Selection:** Important features were selected using `mutual_info_classif`.
- **Cleaning:** The dataset was cleaned by removing irrelevant features like the `Date` column.

### Experimentation
- **Models:** Four models were trained and tested: Decision Tree, K-Nearest Neighbors (KNN), Multiple Linear Regression, and Random Forest.
- **Metrics:** The models were evaluated based on accuracy, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared score (R2 Score).

## 🧠 Algorithms
### Decision Trees
A supervised learning method suitable for both classification and regression tasks, which predicts the target variable using simple decision rules derived from data features.

### K-Nearest Neighbors (KNN)
A supervised learning algorithm used for classification and regression, which classifies data points based on the proximity to training examples.

### Multiple Linear Regression
A statistical technique modeling the relationship between a dependent variable and multiple independent variables, assuming a linear relationship.

### Random Forest
An ensemble method that constructs multiple decision trees during training and outputs the mean prediction of the individual trees for regression tasks.

## 💻 Implementation
The project was implemented using the following tools and technologies:
- **Operating System:** Windows
- **Development Environments:** Google Colab, VS Code
- **Libraries:** Scikit-learn, Numpy, Pandas, Matplotlib, Pickle, Seaborn
- **Framework:** Streamlit for the web interface

## 📏 Evaluation Metrics
- **Accuracy:** Percentage of correct predictions.
- **MSE:** Average squared difference between actual and predicted values.
- **MAE:** Average absolute difference between actual and predicted values.
- **RMSE:** Square root of the average squared differences.
- **R2 Score:** Proportion of the variance in the dependent variable predictable from the independent variables.

## 🏆 Results
The Decision Tree model achieved the highest accuracy of 0.9988, making it the best performer among the tested models. It also showed superior results in other metrics like R2 Score, MSE, RMSE, and MAE.

## ⚠️ Limitations
The dataset size (5952 rows) may lead to underfitting. Additionally, traffic prediction involves numerous factors beyond those considered, such as road maintenance, fuel prices, and real-time traffic updates, which were not included in this analysis.

## 🚀 How to Run

1. **Download** all the files in the repository.
2. **Save** them in a folder on your computer.
3. **Open** the folder in VS Code.
4. **Run** the following command in the terminal:  streamlit run Final_GUI.py
                                    
## 🔚 Conclusion
This project demonstrates the efficacy of machine learning in predicting traffic congestion. By focusing on critical features like time, weather, and vehicle counts, the Decision Tree model emerged as the top performer, providing a robust foundation for future enhancements and real-time traffic management solutions.

```bash


Happy Predicting! 🚗🚴‍♂️🚌🚛
