# Spam Email Detector

This project implements a **machine learning-based spam email detector** using the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).  
It classifies emails/messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and Logistic Regression.



##  Features
- Text preprocessing (cleaning, stopword removal, stemming).
- TF-IDF vectorization to convert text into numerical features.
- Logistic Regression for binary classification.
- Achieved **95% accuracy** on test data.
- Function to predict new email/text inputs.


##  Project Workflow
1. **Data Loading** – Load the dataset and select relevant columns.
2. **Preprocessing** – 
   - Remove special characters  
   - Lowercasing  
   - Stopword removal  
   - Stemming  
3. **Feature Engineering** – Convert text into TF-IDF vectors.
4. **Model Training** – Train Logistic Regression.
5. **Evaluation** – Accuracy + classification report.
6. **Prediction** – Detect if a new email is spam or not.


##  Model Performance
- **Accuracy:** 95.4%  
- **Precision (Spam):** 0.96  
- **Recall (Spam):** 0.69  

This means the model is very good at identifying spam, but still misses a few (lower recall).



