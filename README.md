# CodeAlpha_Avertising
# 📊 Advertising Budget Optimizer (End-to-End ML App)

An interactive Decision Support System (DSS) built with **Python** and **Streamlit** that predicts sales and optimizes marketing budget allocation using Machine Learning.

## 🚀 Key Features
- **Predictive Modeling:** Compare between Linear Regression and Random Forest models to forecast sales.
- **AI Optimization:** Uses `scipy.optimize` (SLSQP) to redistribute budgets for maximum ROI.
- **Interactive Insights:** Visualizes market synergy (TV × Radio) and saturation points using Plotly.
- **Professional UI:** Dark-mode themed dashboard with real-time metric updates.

## 🛠️ Tech Stack
- **Languages:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, Plotly, Joblib, SciPy.
- **Deployment:** Streamlit.

## 📈 Optimization Logic
The app doesn't just predict; it prescribes. By defining a total budget constraint, the AI engine searches for the global optimum allocation across TV, Radio, and Newspaper channels to maximize units sold, often discovering 5-15% efficiency gains.

## 📂 Project Structure
- `ad.py`: The main Streamlit application.
- `*.pkl`: Pre-trained models and scalers.
- `AD.csv`: Historical advertising data.
- `Ad.ipynb` : Notebook with the full work
