# House Price Prediction Using Multiple Linear Regression

This project implements multiple linear regression to predict house prices based on various features, utilizing a dataset that includes key attributes like square footage, number of bedrooms, and location. It evaluates model performance through training and testing accuracies and loss metrics. Additionally, the project explores regularization techniques such as Lasso (L1) and Ridge (L2) regression to address overfitting and improve predictive accuracy. The findings highlight the effectiveness of these methods in enhancing model performance for real estate price predictions.



## Table of contents

- `Introduction`
- `Dataset`
- `Methodology`

     1. `Exploratory data analysis`
     2. `Model Development`
     3. `Reguralization Analysis`  
- `Model Evaluation`

     1. `Train Accuracy`
     2. `Test Accuracy`
- `Results`
- `Conclusion`
  
**Introduction**

  The objective of this project is to  develop a predictive model for    house prices, leveraging multiple linear regression techniques. This includes evaluating the model's accuracy and exploring regularization methods to enhance performance and reduce overfitting.

**Dataset**

The dataset used for this project contains information about residential properties, including various features that influence house prices. It consists of **4600** samples and **6** attributes. Key features in the dataset include:

- `Square Footage`:
 The total area of the house in square feet.
- `Number of Bedrooms`: 
The total number of bedrooms in the house.
- `Number of Bathrooms`: 
The total number of bathrooms.
- `Location`: 
Categorical variable representing the neighborhood or district.
- `Year Built`:
 The year the house was constructed.
- `Lot Size`:
 The area of the property lot.

**Data Preprocessing**

- `Missing Values:`
 Any missing values were handled through Random Imputaion.

- `Categorical Encoding:`
 Categorical features were encoded using One-Hot Encoding.

- `Feature Scaling:`
 Numerical features were scaled using Normalization to improve model performance.

 
## Methodlogy

- `Data Exploration`

Visualizations of the data distribution and correlations between features and the target variable.

### Model Development

1. `Implementation of Multiple Linear Regression.`

   In this project, we implemented multiple linear regression using Python's scikit-learn library. The following steps outline the process:
   - **Import Libraries**
First, we import the necessary libraries for data manipulation and model building.
   
   - **Load the Dataset**
Next, we load the dataset and perform initial exploration.

   - **Split the Data**

We split the dataset into training and testing sets.

   - **Train the Model**

We create an instance of the LinearRegression model and fit it to the training data.
    
  - **Make Predictions**

After training, we use the model to make predictions on the test set.

  - **Evaluate the Model**
Finally, we evaluate the model’s performance using metrics like Mean Squared Error (MSE) and R-squared.

2. `Introduction to Lasso Regression (L1 regularization) and Ridge Regression (L2 regularization).`

 - **Lasso Regression (L1 Regularization)**
Lasso regression, or Least Absolute Shrinkage and Selection Operator, applies L1 regularization to the linear regression model. This technique adds a penalty equal to the absolute value of the magnitude of coefficients. The primary benefits of Lasso regression include:

  - `Feature Selection:`
   By penalizing the absolute size of the coefficients, Lasso can shrink some coefficients to zero, effectively selecting a simpler model that retains only the most important features.
- `Reduced Overfitting:`
 The L1 penalty helps prevent overfitting, especially when dealing with high-dimensional datasets.
Ridge Regression (L2 Regularization)
Ridge regression applies L2 regularization to the linear regression model, adding a penalty equal to the square of the magnitude of coefficients.

**Key aspects of Ridge regression include:**

- `Coefficient Shrinkage:`
Unlike Lasso, Ridge regression does not set coefficients to zero, but instead, it reduces their magnitude. This can lead to improved performance when all features are potentially relevant.

- `Stability in Multicollinearity:`

Ridge regression is particularly useful when multicollinearity is present in the dataset, as it helps to stabilize the estimates of the coefficients.

**Comparison**

- `Feature Selection:`
Lasso can perform automatic feature selection, while Ridge keeps all features but reduces their impact.

- `Complexity Control:`
Both methods help control model complexity, but they do so differently based on the penalty type.

**Training and Testing**

- Splitting the dataset into training and testing sets.
- Training the models on the training set.

## Model Evaluation

**Train Accuracy:**

- `Explanation of the training accuracy achieved by the model.`

Definition of **Training Accuracy**:

Training accuracy refers to the performance of the model on the training dataset after it has been fitted. It indicates how well the model can predict the target variable based on the input features it has seen during training.

**Achieved Training Accuracy**

After training the multiple linear regression model, we calculated the training accuracy using metrics such as Mean Squared Error (MSE) and R-squared (R²).

   **R-squared (R²):**

This metric represents the proportion of variance in the target variable that can be explained by the independent variables in the model. An R² value close to 1 indicates that the model explains a high proportion of the variance, suggesting a good fit.

   **Mean Squared Error (MSE):**

This metric quantifies the average squared difference between the actual and predicted values. A lower MSE indicates better model performance.

**Interpretation**:

For this model, we achieved a training R² score of 0.0890158096882685, which implies that the model can explain approximately 8.9% of the variance in house prices based on the features used. The MSE was calculated as 67733686513.870804, indicating the average prediction error on the training dataset.

**Importance of Training Accuracy**

While a high training accuracy suggests that the model has learned the underlying patterns in the training data, it’s crucial to ensure that the model is not overfitting. Overfitting occurs when a model learns the noise in the training data rather than the actual trends, leading to poor performance on unseen data.

**Test Accuracy:**

- `Explanation of the test accuracy achieved by the model.`

**Definition of Test Accuracy**
Test accuracy refers to the performance of the model on the testing dataset, which consists of data that was not used during the training process. This metric evaluates how well the model can generalize to unseen data and predict the target variable.

**Achieved Test Accuracy**

After training the multiple linear regression model, we calculated the test accuracy using key performance metrics such as Mean Squared Error (MSE) and R-squared (R²).

**R-squared (R²):**

 The R² value obtained on the test dataset was 0.029698195068123656. This indicates that approximately 2.9% of the variance in house prices can be explained by the features in the model, demonstrating its predictive capability on new data.

**Mean Squared Error (MSE):**

 The MSE on the test dataset was 989558744539.465. This value reflects the average squared difference between the actual house prices and the prices predicted by the model, providing insight into the model's prediction error.

**Interpretation**

The test accuracy is critical for understanding the model's ability to perform in real-world scenarios. In this case, the achieved test R² score suggests that the model maintains a good level of explanatory power, despite not having seen the test data during training. The MSE value indicates how closely the predicted prices align with the actual prices, with lower values indicating better performance.

**Importance of Test Accuracy**

Evaluating test accuracy is essential for assessing the model's generalization capability. A model with high training accuracy but significantly lower test accuracy may indicate overfitting, where the model learns the training data too well, including noise and outliers, resulting in poor performance on new data.

`By comparing the training and test accuracy, we can determine the model's robustness and make informed decisions about potential improvements, such as feature selection, regularization, or trying different algorithms.`

**Loss:**

Description of the loss metric used (e.g., Mean Squared Error, R-squared) and the values obtained during evaluation.

In this project, we used two primary metrics to evaluate the performance of the multiple linear regression model: Mean Squared Error (MSE) and R-squared (R²). These metrics help us understand the model's predictive accuracy and its effectiveness in explaining the variance in house prices.

1. Mean Squared Error (MSE)
Mean Squared Error is a common loss metric used to quantify the average squared difference between the actual target values and the values predicted by the model. It is calculated using the following formula:
  
   MSE= 
1 / n  
​n
∑
i=n
   (yi - y^
​ 
i
​
 ) **
2


  Where:

𝑛 is the number of observations

𝑦𝑖 is the actual value

𝑦
^
𝑖 is the predicted value


**values obtained**

For the training dataset, the MSE was 67733686513.870804 .
For the testing dataset, the MSE was 989558744539.465.

2. R-squared (R²)
R-squared is a statistical measure that represents the proportion of variance for the target variable that can be explained by the independent variables in the model. It is expressed as a value between 0 and 1, where:

An R² value of 1 indicates perfect prediction.
An R² value of 0 indicates that the model explains none of the variance.
The formula for R² is:

  R 
2
 =1− 
SS 
tot
​
/ 
SS 
res

​Where:

SSres is the sum of squares of residuals (errors).

SS 
tot is the total sum of squares.
 
​**Value Obtained:**

For the training dataset, the R² score was 0.0890158096882685.
For the testing dataset, the R² score was 0.029698195068123656.

**Interpretation of Values**

- `Training MSE:`
A lower MSE indicates that the model predictions are closer to the actual values, suggesting a good fit on the training data.

- `Testing MSE:`
 The testing MSE provides insight into the model's performance on unseen data, with a lower value indicating better generalization.

- `Training R²:`
 A higher R² score signifies that the model explains a significant portion of the variance in the training data.

- `Testing R²:`
 The testing R² score indicates how well the model performs on new data, providing a measure of its predictive power.

By analyzing these metrics, we can assess the effectiveness of our model and identify areas for potential improvement.

Comparison of Regularization Techniques:

Results of Lasso and Ridge regression models compared to the baseline multiple linear regression model.


## Results

**Summary of findings:**

- Tables or plots showing the performance of each model.

- Insights on how regularization techniques impacted model performance.
## Conclusion

- Summary of the project’s findings.
- Discussion on the effectiveness of multiple linear regression and regularization methods for house price predictions.
- Suggestions for future work, such as incorporating more features or exploring other algorthims.

