# 🤖 Machine Learning Algorithms

## 📌 Overview
This repository contains implementations of Supervised and Unsupervised Machine Learning algorithms with preprocessing, feature engineering, model optimization, and pipelines using Python.

---

# 🧠 Machine Learning Types

## 1. Supervised Learning
Supervised Learning uses labelled data where input features and target outputs are provided.

Examples:
- Classification
- Regression


## 2. Unsupervised Learning
Unsupervised Learning finds hidden patterns from unlabelled data.

Examples:
- Clustering

---

# 📘 Supervised Learning Algorithms

## 📈 1. Linear Regression

Used for predicting continuous numerical values.

Applications:
- House Price Prediction
- Sales Forecasting

Evaluation Metrics:

- MAE
- MSE
- RMSE
- R² Score


---

# 📊 2. Polynomial Regression

Used when data contains non-linear relationships.

Concepts:
- Polynomial Features
- Degree
- Curve Fitting

Example:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2
)

X_poly = poly.fit_transform(X)
```

---

# 🎯 3. Logistic Regression

Used for classification problems.

Examples:
- Spam Detection
- Disease Prediction

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix


---

# 📍 4. K-Nearest Neighbors (KNN)

Distance based algorithm.

Concepts:

- Euclidean Distance
- Choosing K value
- Feature Scaling


---

# 🔥 5. Support Vector Machine (SVM)

Creates an optimal decision boundary called Hyperplane.

Types:

- Linear Kernel
- Polynomial Kernel
- RBF Kernel


---

# 🌳 6. Decision Tree

Tree based model using decision rules.

Concepts:

- Root Node
- Leaf Node
- Entropy
- Information Gain
- Gini Index


---

# 🌲 7. Random Forest

Ensemble learning technique using multiple decision trees.

Advantages:

- Reduces Overfitting
- Better Accuracy
- Handles large datasets


---

# ⚡ 8. XGBoost

Extreme Gradient Boosting algorithm.

Features:

- Boosting Technique
- Regularization
- High Performance


---

# 📨 9. Naive Bayes

Probability based classification algorithm using Bayes theorem.

Types:

- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

Applications:

- Text Classification
- Spam Detection
- Sentiment Analysis


Example:

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(
    X_train,
    y_train
)
```

---

# 🔤 Text Vectorization (NLP)

Converts text into numerical features.

## CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer()

X = cv.fit_transform(text)
```


## TF-IDF Vectorizer


```python
from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer()

X = tfidf.fit_transform(text)
```


Used For:

- NLP
- Sentiment Analysis
- Text Classification


---

# ⚙️ Feature Scaling


## StandardScaler

Mean = 0  
Standard Deviation = 1


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```


## MinMaxScaler

Converts values between 0 and 1.


---

# 🔢 Encoding Techniques


## Label Encoding

Converts categories into numbers.


## One Hot Encoding

Creates binary columns for categorical values.


Example:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

```


---

# 🎯 Regularization


Prevents model overfitting.


## Ridge Regression (L2)

- Shrinks coefficients
- Reduces complexity


## Lasso Regression (L1)

- Feature Selection
- Removes unnecessary features


## ElasticNet

Combination:

Ridge + Lasso


---

# 🔎 Hyperparameter Tuning


## GridSearchCV

Finds the best parameters automatically.


```python
from sklearn.model_selection import GridSearchCV


grid = GridSearchCV(
    model,
    params,
    cv=5
)


grid.fit(
    X_train,
    y_train
)


grid.best_params_
```


---

# 🔄 Pipeline


Combines preprocessing and model training steps.


```python
from sklearn.pipeline import Pipeline


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])


pipe.fit(
    X_train,
    y_train
)
```


Benefits:

- Clean Code
- Avoids Data Leakage
- Easy Deployment


---

# 🧩 ColumnTransformer


Applies different preprocessing to different columns.


Example:

```python
from sklearn.compose import ColumnTransformer


transformer = ColumnTransformer([
    
("encoder", OneHotEncoder(), categorical_columns),

("scaler", StandardScaler(), numerical_columns)

])

```

---

# 🔵 Unsupervised Learning Algorithms


# 📌 K-Means Clustering


Groups similar data points together without labels.


Concepts:

- Cluster
- Centroid
- Inertia
- Elbow Method


Steps:

1. Select number of clusters (K)

2. Assign points to nearest centroid

3. Update centroid

4. Repeat until convergence


Example:


```python
from sklearn.cluster import KMeans


model = KMeans(
    n_clusters=3
)


model.fit(X)

```


Evaluation:

- Inertia
- Silhouette Score


---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Jupyter Notebook


---

# 🔥 Complete ML Workflow

1. Import Dataset

2. Data Cleaning

3. Exploratory Data Analysis (EDA)

4. Feature Engineering

5. Encoding

6. Vectorization

7. Feature Scaling

8. Train Test Split

9. Model Training

10. Hyperparameter Tuning

11. Pipeline Creation

12. Model Evaluation

13. Prediction


---

# 👨‍💻 Author

**Mukeshkumar M**

Data Science | Machine Learning | Big Data

```
