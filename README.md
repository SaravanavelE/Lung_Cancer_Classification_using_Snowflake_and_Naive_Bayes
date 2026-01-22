# Lung Cancer Classification Using Snowflake and Naive Bayes

## Project Overview
This project implements a **Lung Cancer Classification system** by integrating the **Snowflake Cloud Data Warehouse** with **Machine Learning**. Patient data is fetched directly from Snowflake, preprocessed, and classified using the **Naive Bayes (GaussianNB)** algorithm.

The project demonstrates a real-world **cloud data engineering + machine learning** workflow using Google Colab.

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Snowflake Connector for Python
- Google Colab

---

## Workflow
- Connect to Snowflake database
- Fetch data using SQL query
- Inspect data and check for missing values
- Encode categorical variables
- Split data into training and testing sets (80:20)
- Train Naive Bayes model
- Evaluate model performance

---

## Model Training
```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)
Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, nb.predict(x_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb.predict(x_test)))
print("Classification Report:\n", classification_report(y_test, nb.predict(x_test)))
```
## Performance Metrics
Accuracy: 79.03%
Precision (Cancer): 0.83
Recall (Cancer): 0.92
F1-Score (Cancer): 0.87
Confusion Matrix
[[ 4  9]
 [ 4 45]]
 
## Project Structure
├── lung_cancer_classification_naive_bayes.ipynb
├── README.md

## Future Enhancements
Compare with Logistic Regression, KNN, and SVM
Hyperparameter tuning
Handle class imbalance
Feature importance analysis
Deploy model using Flask or FastAPI
Automate Snowflake data ingestion

## Author
Saravanavel E
AI & Data Science Student
GitHub: https://github.com/SaravanavelE

License
This project is intended for educational and academic purposes only.
