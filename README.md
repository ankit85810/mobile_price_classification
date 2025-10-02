# 📱 Mobile Price Classification  

This project predicts the **price range of mobile phones** using different **machine learning classification models**.  
I worked with both **scikit-learn implementations** and my own **hard-coded models** to deepen my understanding of ML fundamentals.  

---

## 📂 Dataset
- **Source:** [Kaggle - Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data?select=train.csv)  
- **Description:** The dataset contains mobile specifications (battery power, RAM, camera, etc.) and the target variable is the **price range** (0 = low, 1 = medium, 2 = high, 3 = very high).  

---

## 🔑 Features of the Project
1. **Implemented Models (from scratch):**
   - Logistic Regression  
   - Decision Tree Classifier  
   - Random Forest Classifier  

2. **Library Models (scikit-learn):**
   - Logistic Regression  
   - Decision Tree Classifier  
   - Random Forest Classifier  
   - Support Vector Classifier (only sklearn, not self-implemented)  

3. **Evaluation Metrics:**
   - R² Score  
   - Classification Report (Precision, Recall, F1-score)  
   - Confusion Matrix (with Heatmap visualization)  

4. **Preprocessing:**
   - Feature Scaling (StandardScaler)  
   - Train-Test Split  

5. **Additional Testing:**
   - To ensure correctness of self-implemented models, I also tested them on the **Breast Cancer dataset** from `sklearn.datasets`.

---

## 📊 Results
- Comparison was made between **self-implemented models** and **scikit-learn models**.  
- The evaluation metrics confirmed that the manually implemented algorithms worked as expected.  

## 📊 Model Comparison (Breast Cancer Dataset)

| Model                          | Accuracy | Precision (Class 0 / 1) | Recall (Class 0 / 1) | F1-Score (Class 0 / 1) |
|--------------------------------|----------|--------------------------|-----------------------|-------------------------|
| **Sklearn Logistic Regression** | 98.25%   | 0.98 / 0.99             | 0.98 / 0.99          | 0.98 / 0.99            |
| **Custom Logistic Regression**  | 96.49%   | 0.93 / 0.99             | 0.98 / 0.96          | 0.95 / 0.97            |
| **Sklearn Decision Tree**       | 92.98%   | 0.89 / 0.96             | 0.93 / 0.93          | 0.91 / 0.94            |
| **Custom Decision Tree**        | 92.11%   | 0.87 / 0.96             | 0.93 / 0.92          | 0.90 / 0.94            |
| **Sklearn Random Forest**       | 95.61%   | 0.95 / 0.96             | 0.93 / 0.97          | 0.94 / 0.97            |
| **Custom Random Forest**        | 95.61%   | 0.91 / 0.99             | 0.98 / 0.94          | 0.94 / 0.96            |


---

## 🚀 Tech Stack
- Python  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- scikit-learn  

---

## 📌 Project Structure
```
├── data/                     # Dataset files
├── notebooks/                # Jupyter Notebooks
│   ├── mobile_price.ipynb     # Main project notebook
│   ├── breast_cancer_test.ipynb # Testing with sklearn dataset
├── models/                   # Hard-coded implementations
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
├── README.md                 # Project description
```

---

## ⚡ How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/ankit85810/mobile-price-classification.git
   cd mobile-price-classification
   ```
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook  
   ```bash
   jupyter notebook
   ```
4. Run `mobile_price.ipynb`  

---

## 🔮 Future Work
- Implement Support Vector Classifier from scratch.  
- Try advanced models (XGBoost, LightGBM).  
- Hyperparameter tuning for better accuracy.  

---

## 📝 Author
- **Ankit Vishwakarma**  
- Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/ankit-vishwakarma-500a7b324/) 🚀  
