
# 📊 Banking Analytics Dashboard

## Overview

An interactive, multi-tab dashboard built using **Streamlit** for end-to-end banking analytics.  
The system visualizes and analyzes customer data, accounts, cards, transactions, loans, fraud, churn, and support calls.

It supports dynamic integration with **Azure SQL Database** and falls back gracefully to local CSV files if needed.

---

## 🚀 Key Features

- **🧍 Customer Overview**  
  Track total/active/inactive customers, join trends, churn risk categories, state distributions.

- **🏦 Account Insights**  
  Monitor balances, account types, activity levels, and dormant capital concentration.

- **💸 Transactions Dashboard**  
  Analyze transaction types, seasonal/monthly/daily trends, and high-risk behaviors.

- **🚨 Fraud Detection**  
  Detect fraud using outlier detection and visualize trends by time, location, and transaction type.

- **📈 Loans Analysis**  
  View distribution by loan types, interest rate trends, maturities, and upcoming expirations.

- **💳 Card Analytics**  
  Evaluate card types, lifecycle status, issuance trends, active/expired distributions.

- **📞 Support Calls**  
  Assess resolution rate, call volume, peak periods, average duration, and top complaint types.

- **🧠 Advanced Insights**  
  Segment high-value customers, measure product adoption, and uncover growth opportunities.

- **🔮 Churn Prediction**  
  Uses a Random Forest model to classify customers at risk of churn.

- **☁️ Azure SQL Support**  
  Automatically connects to Azure SQL and handles fallback using local data when offline.

---

## 🗂️ Project Structure

```
banking-analytics-dashboard/
│
├── app.py                    # Streamlit application entry point
│
├── csv/                      # Fallback dataset folder
│   ├── Banking_Analytics_Dataset_Updated2.csv
│   ├── Banking_Analytics_Transactions_Updated.csv
│   └── ...                   # Other backup CSVs (Accounts, Loans, etc.)
│
├── best_rf_churn_model.pkl   # Pre-trained churn prediction model using Random Forest
│
├── requirements.txt          # Python dependencies for setting up the environment
│
├── README.md                 # Project documentation (this file)
│
└── .streamlit/
    └── secrets.toml          # Azure SQL credentials (DO NOT upload this file)
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone [https://github.com/username/banking-analytics-dashboard.git](https://github.com/AHMEDHANY146/Bank-project)
cd banking-analytics-dashboard
```

### 2. Create a virtual environment and install requirements
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Azure SQL credentials
Create a file: `.streamlit/secrets.toml` and paste your connection info:
```toml
[sql_credentials]
server = "your_server.database.windows.net"
database = "your_database"
username = "your_user"
password = "your_password"
```

### 4. Run the application
```bash
streamlit run app.py
```

---

## 🧠 Technologies Used

| Tool                  | Purpose                             |
|-----------------------|-------------------------------------|
| Streamlit             | Interactive UI framework            |
| Pandas, Plotly, Altair| Data manipulation & visualization   |
| PyDeck                | Geo maps & state-level insights     |
| Scikit-learn + Joblib | ML churn model                      |
| Azure SQL + pyodbc    | External database integration       |
| ydata-profiling       | Automated data profiling            |

---

## 🔐 Security Notice

Do **NOT** upload sensitive credentials or model files to GitHub.

Include the following in `.gitignore`:
```
.streamlit/secrets.toml
*.pkl
__pycache__/
```

---

## 💡 Future Enhancements

- Authentication & Role-based access
- Power BI report integration
- Executive dashboards
- Arabic language UI
- Automated Telegram bot reporting

---

## 👨‍💻 Author

**Developed by:** Ahmed Hany  
Pull requests and contributions are welcome!

---
