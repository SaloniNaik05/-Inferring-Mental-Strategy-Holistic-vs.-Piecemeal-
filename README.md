# Inferring Mental Strategy: Holistic vs. Piecemeal 🧠

This project investigates **how individuals process information**—whether they adopt a **holistic strategy** (seeing the big picture) or a **piecemeal strategy** (focusing on individual components). The analysis combines **data preprocessing, feature engineering, Hidden Markov Models (HMM), clustering, and visualization** to classify and understand mental strategies.

---

## 🔬 Objective
- Identify patterns in human decision-making strategies  
- Classify mental strategies as **Holistic** or **Piecemeal**  
- Provide visual insights for research and educational purposes  

---

## 📊 Methodology

### 1. Data Collection
- Collected raw datasets from eye-tracking, behavioral tests, and related experiments  
- Stored in `.csv` format for processing  

### 2. Data Preprocessing
- Handled missing and noisy values  
- Normalized data to improve model performance  
- Split dataset into **training** and **testing sets**  

### 3. Feature Engineering
- Extracted meaningful features such as:  
  - Fixation duration  
  - Saccade distance  
  - Attention shifts  
- Selected features that are most predictive of holistic or piecemeal strategies  

### 4. Model Training
- **Hidden Markov Model (HMM)** trained on processed features  
- Model saved as `.pkl` for reproducibility and deployment  

### 5. Clustering
- Applied clustering algorithms to identify groups of similar strategies  
- Compared clusters with ground truth to validate classification  

### 6. Visualization 📊
- Heatmaps for eye-tracking patterns  
- Scatter plots for feature clustering  
- Interactive plots to explore mental strategy behavior  

> Example Visualization:  


---

## 🛠 Tools & Libraries
- **Python**  
- **pandas, numpy** – Data handling  
- **scikit-learn** – Feature selection, clustering  
- **hmmlearn** – Hidden Markov Model implementation  
- **matplotlib, seaborn, plotly** – Visualization  

---

## 🗂 Project Structure
