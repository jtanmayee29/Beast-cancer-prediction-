# 🧠 Breast Cancer Prediction using Machine Learning

## 📌 Project Overview

This project focuses on building a Machine Learning model to predict whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** using medical diagnostic measurements.

The notebook demonstrates a complete beginner-friendly ML pipeline including:

* Data loading
* Data exploration
* Data cleaning
* Data visualization
* Data preprocessing
* Model training
* Model evaluation

The model is built using **Logistic Regression** and trained on the dataset **Cancer_Data.csv**.

---

## 🎯 Objective

The main goal of this project is to develop a classification model that can accurately predict the diagnosis of breast cancer based on input features.

The target variable:

* **M** → Malignant (1)
* **B** → Benign (0)

---

## 📂 Dataset

The dataset used: **Cancer_Data.csv**

It contains:

* Tumor measurement features
* Diagnosis label (M/B)
* ID column (not required for prediction)
* An extra unnamed column (to be removed)

---

## 🛠️ Technologies Used

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 📊 Machine Learning Pipeline

### 1️⃣ Load the Dataset

* Import dataset using Pandas
* Display basic structure

### 2️⃣ Explore the Dataset

* Check shape of data
* View column names
* Identify missing values
* Understand data distribution

### 3️⃣ Clean the Data

* Remove unnecessary columns:

  * `id`
  * `Unnamed: 32`
* Handle missing values if present

### 4️⃣ Visualize the Data

* Plot correlations
* Count plot for diagnosis
* Heatmap for feature relationships

### 5️⃣ Preprocess the Data

* Convert diagnosis:

  * Benign → 0
  * Malignant → 1
* Split into:

  * Features (X)
  * Target (y)
* Train/Test split
* Standardize features using `StandardScaler`

### 6️⃣ Train the Model

* Use **Logistic Regression**
* Fit model on training data

### 7️⃣ Evaluate the Model

* Accuracy score
* Confusion matrix
* Classification report:

  * Precision
  * Recall
  * F1-Score

---

## 📈 Model Performance

The Logistic Regression model is used to classify tumors and provides good accuracy for medical prediction tasks.

Evaluation metrics used:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

## 🚀 How to Run the Project

1. Install required libraries:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Place `Cancer_Data.csv` in the project folder.

3. Run the Jupyter Notebook step by step.

---

## 📁 Project Structure

```
Breast-Cancer-Prediction/
│
├── Cancer_Data.csv
├── Breast_Cancer_Prediction.ipynb
└── README.md
```

---

## 🔮 Future Improvements

* Try advanced models:

  * Random Forest
  * Support Vector Machine (SVM)
  * Decision Tree
* Perform feature selection
* Hyperparameter tuning
* Cross-validation
* Deploy as a web application

---

## 👩‍💻 Author

Developed as a beginner-friendly Machine Learning project to understand classification, preprocessing, and model evaluation.

---

## 📜 License

This project is open-source and free to use for learning purposes.
