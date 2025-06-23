
# 🧠 Disease Prediction from Symptoms

A machine learning-powered web application to predict potential diseases based on user-selected symptoms. The system leverages both **Random Forest** and **Deep Neural Network (DNN)** models for robust and intelligent health predictions.

This project is part of a hands-on learning initiative that explores both classical ML and deep learning methods.

---

## 📌 Objective

To develop a model that can predict diseases accurately using symptom-based binary input vectors, providing early guidance and awareness to users.

---

## 🧩 Problem Statement

> Given a list of symptoms, predict the most likely **disease** from a large set of medical conditions.

---

## 🗃️ Dataset

- 📂 **Source**: [Kaggle - Diseases and Symptoms Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)  
- 📄 **File Used**: `Final_Augmented_dataset_Diseases_and_Symptoms.csv`

This dataset maps over **650+ diseases** to their associated symptoms.

---

## ⚙️ Machine Learning Models

Two different models were trained and evaluated:

### 🔹 Random Forest Classifier
- ✅ Accuracy: **78.10%**
- Handles multi-class prediction well
- Quick inference for Streamlit deployment

### 🔹 Deep Neural Network (DNN)
- ✅ Accuracy: **81.89%**
- Learns deep patterns from the symptom combinations
- Structured with two dense hidden layers and a softmax output

> ⚠️ **Note**: Predictions may differ between the two models for the same inputs, due to differences in learning approach and internal representation.  
> **Ongoing improvements** are in progress to enhance the **efficiency and consistency** of both models.

---

## 🧠 Features Used

Symptom presence is encoded as binary features (1 = present, 0 = absent). The dataset includes **370+ unique symptoms**.

---

## 📊 App Demo

The Streamlit web app allows users to:
- Select symptoms from a multiselect dropdown
- Choose a model (Random Forest or DNN)
- Get instant disease predictions
- View precautions via Wikipedia link

### 🚀 To Run the App:

```bash
streamlit run app.py
```

---

## 📂 Folder Structure

```
disease_prediction_app/
├── app.py
├── random_forest_model.pkl
├── dnn_model.keras
├── label_encoder.pkl
├── Final_Augmented_dataset_Diseases_and_Symptoms.csv
├── README.md
```

---

## 🧪 Requirements

```bash
pip install pandas numpy scikit-learn tensorflow streamlit
```

---

## 📫 Author

**Vivekanand Ojha**  
📧 Email: vivekanandojha09@gmail.com  
🔗 GitHub: [@vivekOJ1129](https://github.com/vivekOJ1129)  
🔗 LinkedIn: [Vivekanand Ojha](https://www.linkedin.com/in/vivekanand-ojha-485462289/)

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
