# FairAffinity: Fairness-Aware Product Affinity Prediction & Recommendation

FairAffinity is a **machine learning project** designed to predict user affinity toward product categories and brands, and generate **personalized, fairness-aware recommendations**.  
It uses multiple datasets (clients, products, transactions, stores, stocks) to create a hybrid recommendation engine, monitor bias across demographic groups, and deploy results via APIs and dashboards.

---

## 🎯 Project Objectives
1. **Predict Affinity**  
   Estimate the probability that a client will interact with or purchase a given product or product category.

2. **Enhance Recommendations**  
   Combine collaborative filtering, content-based filtering, and affinity scores to serve personalized recommendations.

3. **Ensure Fairness**  
   Audit recommendations for demographic bias and apply mitigation strategies (e.g., re-ranking) to balance exposure across groups.

4. **Deploy in Production**  
   Package the model and re-ranking logic into an API, with dashboards for monitoring performance and fairness.

---

## 📊 Datasets Used
| Dataset        | Description |
|----------------|-------------|
| **Clients**    | Client demographics, segmentation, and opt-in info. |
| **Products**   | Product metadata including categories and family hierarchies. |
| **Transactions** | Historical purchases linking clients to products, with dates, quantities, and sales amounts. |
| **Stores**     | Store identifiers and associated countries. |
| **Stocks**     | Product stock levels by store and country. |

---

## 🛠 Tech Stack

**Languages & Libraries**
- Python 3.11+
- pandas, polars, numpy
- scikit-learn, LightGBM, XGBoost
- implicit (ALS collaborative filtering)
- SHAP (explainability)
- Aequitas / AIF360 (fairness auditing)

**APIs & Deployment**
- FastAPI, Uvicorn, Docker

**Experiment Tracking**
- MLflow

**Visualization**
- Plotly, Streamlit

---

## 📂 Project Structure
