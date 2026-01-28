# Diabetes Analysis and Prediction using Machine Learning

## **Project Overview**

This project focuses on **predictive modeling of diabetes** using clinical data. While Phase 1 of the project focused on **data loading, cleaning, outlier removal, and exploratory analysis**, Phase 2 focuses on **training machine learning models** to classify diabetic status and evaluating their performance.  

**Objectives:**
- Encode categorical variables
- Split data into training and testing sets
- Train multiple classification models
- Evaluate model performance
- Compare models and identify the best one
- Analyze feature importance
- Perform subgroup analysis by age and gender

---

## **Data Preparation**

- **Dataset:** `cleaned_diabetes_dataset_iqr.csv` (678 rows × 14 columns)  
- **Libraries Used:**  
  - Data handling: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Machine learning: `sklearn.linear_model`, `sklearn.tree`, `sklearn.ensemble`, `sklearn.neighbors`  
  - Preprocessing & evaluation: `LabelEncoder`, `train_test_split`, `metrics`  

- **Categorical Encoding:**  
  - `Gender` and `CLASS` columns encoded using `LabelEncoder`  

- **Train-Test Split:**  
  - 80% training, 20% testing  

---

## **Machine Learning Models**

The following classification models were trained:

1. **Logistic Regression**  
   - Linear baseline model  
   - Parameters: `max_iter=5000`  

2. **Decision Tree Classifier**  
   - Captures nonlinear relationships  
   - Parameters: `random_state=42`  

3. **Random Forest Classifier**  
   - Ensemble method, strongest model  
   - Parameters: `n_estimators=200`, `random_state=42`  

4. **K-Nearest Neighbors (KNN)**  
   - Distance-based model  
   - Parameters: `n_neighbors=5`  

---

## **Model Evaluation**

Models were evaluated using:  
- Accuracy Score  
- Classification Report (precision, recall, F1-score)  
- Confusion Matrix  

**Results:**  
- **Random Forest:** Highest accuracy and best overall performance  
- **Decision Tree:** Strong performance  
- **Logistic Regression & KNN:** Baseline performance  

---

## **Feature Importance (Random Forest)**

The most predictive features were:  
1. BMI  
2. HbA1c  
3. Age  
4. Triglycerides (TG)  
5. Cholesterol  

---

## **Subgroup Analysis**

- **Younger Age Group (<40 years):**  
  - Lower prevalence, but higher risk if BMI is elevated  

- **Gender vs Diabetes:**  
  - Females show slightly higher diabetes prevalence (~1–2%)  
  - Patterns are similar across genders  

- **Average Age per Diabetes Class:**  
  - Non-Diabetic: lowest average age  
  - Prediabetic: slightly higher  
  - Diabetic: highest average age  

---

## **Conclusions**

- **Best Model:** Random Forest (highest accuracy & reliability)  
- **Key Predictors:** BMI, HbA1c, Age, Triglycerides, Cholesterol  
- **Insights:**  
  - Age strongly correlates with diabetes progression  
  - Subgroup analysis provides insights for targeted interventions  

---

## **Technologies Used**

- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  

---

## **Usage**

1. Load the dataset `cleaned_diabetes_dataset_iqr.csv`  
2. Run `Phase1.ipynb` for data cleaning & exploratory analysis  
3. Run `Phase2.ipynb` to train models, evaluate performance, and analyze results  

---

**Project Type:** Academic / Data Mining & Business Intelligence  

