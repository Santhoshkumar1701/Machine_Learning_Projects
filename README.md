# Machine_Learning_Projects

[1. House Price Predcition](https://github.com/Santhoshkumar1701/Machine_Learning_Projects/blob/main/Machine_learning_solution_houseprice_dataset.ipynb)

[2. Feed Grains Analysis](https://github.com/Santhoshkumar1701/Machine_Learning_Projects/blob/main/MachineLearning_solution_FeedGrains_Dataset.ipynb)

[3. Fuel Economy Analysis](https://github.com/Santhoshkumar1701/Machine_Learning_Projects/blob/main/MachineLearning_solution_for_Fuel_Economy_DataSet.ipynb)

[4. Titanic Survival Prediction](https://github.com/Santhoshkumar1701/Machine_Learning_Projects/blob/main/MachineLearning_Solution_Titanic_Dataset.ipynb)

## 1. House Price Prediction ML
### Overview:
This project focuses on predicting house prices using machine learning techniques. It utilizes a dataset containing various features such as lot size, number of bedrooms, and location to train predictive models.

### Dataset:
The dataset used for this project is sourced from a CSV file (train.csv). It contains both numerical and categorical variables related to houses.

### Data Preprocessing:
Imputation: Missing values in numerical columns are filled using the median value of each respective column.
Feature Selection: Only columns with a correlation coefficient greater than 0.3 or less than -0.3 with the target variable ("SalePrice") are considered for modeling.
Outlier Removal: Outliers are identified and removed from the dataset based on the interquartile range (IQR) method.
Scaling: Numerical features are scaled using both StandardScaler and MinMaxScaler techniques.
### Model Training:
Three regression models are trained on the preprocessed dataset:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
### Evaluation:
The trained models are evaluated using the R-squared (R2) metric on both training and testing datasets to assess their performance.
R2 score measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
### Results: 
Linear Regression: Achieved a training score of [1.0] and a testing score of [0.68].
Decision Tree Regressor: Achieved a training score of [1.0] and a testing score of [-0.34].
Random Forest Regressor: Achieved a training score of [0.82] and a testing score of [0.34].
### Conclusion:
The Random Forest Regressor outperformed the other models with the highest testing score. Further optimization and feature engineering could potentially enhance the predictive performance of the models.



## 2. Feed Grains Analysis
### Overview:
This project focuses on analyzing feed grains data using machine learning techniques. The dataset contains various features related to feed grains, including commodity descriptions, geographic information, attributes, and quantities.

### Dataset:
The dataset (FeedGrains.csv) used in this project consists of both numerical and categorical variables related to feed grains.

### Data Preprocessing:
Imputation: Missing values in numerical columns are filled using the mode value of each respective column.
Outlier Removal: Outliers are identified and removed from the dataset based on the interquartile range (IQR) method.
Scaling: Numerical features are scaled using both StandardScaler and MinMaxScaler techniques.
### Model Training:
Three regression models are trained on the preprocessed dataset:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
### Evaluation:
The trained models are evaluated using the R-squared (R2) metric on both training and testing datasets to assess their performance.
R2 score measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
### Results:
Linear Regression: Achieved a training score of [0.10] and a testing score of [0.10].
Decision Tree Regressor: Achieved a training score of [1.0] and a testing score of [0.57].
Random Forest Regressor: Achieved a training score of [0.86] and a testing score of [0.72].
### Conclusion:
The Random Forest Regressor outperformed the other models with the highest testing score. Further analysis and feature engineering could potentially improve the predictive performance of the models.


## 3. Fuel Economy Analysis
Overview
This project focuses on analyzing fuel economy data using machine learning techniques. The dataset contains various features related to vehicle specifications, fuel types, and fuel costs.

### Dataset:
The dataset (vehicles.csv) used in this project consists of numerical and categorical fuel economy variables.

### Data Preprocessing:
Imputation: Missing values in numerical columns are filled using the median value of each respective column.
Outlier Removal: Outliers are identified and removed from the dataset based on the interquartile range (IQR) method.
Scaling: Numerical features are scaled using both StandardScaler and MinMaxScaler techniques.
### Model Training:
Three regression models are trained on the preprocessed dataset:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
### Evaluation:
The trained models are evaluated using the R-squared (R2) metric on both training and testing datasets to assess their performance.
R2 score measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
### Results:
Linear Regression: Achieved a training score of [0.99] and a testing score of [0.99].
Decision Tree Regressor: Achieved a training score of [1.0] and a testing score of [1.0].
Random Forest Regressor: Achieved a training score of [0.99] and a testing score of [0.99].
### Conclusion:
All Models Performed well outperformed the other models with the highest testing score.


## 4. Titanic Survival Prediction
### Overview:
This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The dataset contains various features such as age, gender, ticket class, and cabin, which are used to train the models.

### Dataset:
The dataset used in this project is provided as a CSV file named "Titanic.csv".
It includes information about passengers aboard the Titanic, including whether they survived or not.
The dataset contains both numerical and categorical features.
### Preprocessing:
Loading the Dataset: Read the dataset using pandas.
Handling Missing Values:
For numerical columns, fill missing values with the median.
For categorical columns, fill missing values with the mode.
Outlier Removal:
Identify outliers in numerical features.
Remove outliers using the interquartile range (IQR) method.
Feature Scaling:
Scale numerical features using StandardScaler and MinMaxScaler.
Feature Encoding:
Encode categorical features using LabelEncoder.
Model Training:
Splitting the Data: Split the preprocessed data into training and testing sets (e.g., 70% training, 30% testing).
### Model Selection:
Train three different classification models:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier.
### Results:
The trained models achieve certain accuracy scores on both training and testing datasets, indicating their predictive performance.
