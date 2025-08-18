# Mall Customer Segmentation (K-Means Clustering)

## 📌 Project Overview
This project applies **K-Means Clustering** to segment mall customers based on their **Annual Income** and **Spending Score**.  
The goal is to help businesses understand different customer groups and target them with personalized marketing strategies.

## 📊 Dataset
- Source: [Kaggle - Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Attributes: CustomerID, Gender, Age, Annual Income, Spending Score

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 🔑 Key Steps
1. Data loading and preprocessing
2. Feature scaling using StandardScaler
3. Optimal cluster selection (Elbow Method & Silhouette Score)
4. K-Means clustering
5. Visualization of customer segments
6. Cluster profiling (business interpretation)
7. Exporting final segmented dataset

## 📈 Results
- Identified **5 optimal customer clusters**
- Segments include:
  - High Income, High Spending (VIP customers)
  - Low Income, Low Spending (budget shoppers)
  - High Income, Low Spending (conservative spenders)
  - Low Income, High Spending (impulsive buyers)
  - Average Income, Average Spending (regular customers)

## 🚀 How to Run
```bash
git clone https://github.com/your-username/SCT_ML_2.git
cd SCT_ML_2
pip install -r requirements.txt
python customer_segmentation.py
