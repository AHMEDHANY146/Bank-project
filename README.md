
📊 Banking Analytics Dashboard

Overview
--------
An interactive dashboard built with Streamlit for comprehensive banking analytics. This app visualizes customer data, accounts, transactions, loans, cards, fraud detection, churn risk, and support call insights. It integrates with an Azure SQL database and falls back to CSV files if the connection fails.

Key Features
------------
- 🧍 Customer Overview: Activity analysis, churn detection, geographic distribution.
- 🏦 Account Insights: Account types, balances, dormant accounts.
- 💸 Transactions: Seasonal trends, daily heatmaps, fraud identification.
- 🚨 Fraud Detection: Visual and model-based detection.
- 📈 Loans: Types, interest rates, upcoming maturities.
- 💳 Cards: Types, issuance trends, lifecycle tracking.
- 📞 Support Calls: Resolution rates, issue types, customer service analysis.
- 🧠 Advanced Insights: High-value customers, product adoption.
- 🔮 Churn Prediction: Machine learning model integration.
- ☁️ Azure SQL integration with local fallback.

Project Structure
-----------------
project-root/
│
├── app.py                  # Main Streamlit application
├── csv/                    # Fallback datasets in CSV
├── best_rf_churn_model.pkl # Pre-trained churn prediction model
├── requirements.txt        # Dependencies
├── README.txt              # This documentation
└── .streamlit/
    └── secrets.toml        # Azure SQL credentials (DO NOT upload)

Setup Instructions
------------------
1. Clone the repository:
    git clone https://github.com/username/banking-analytics-dashboard.git
    cd banking-analytics-dashboard

2. Create a virtual environment and install dependencies:
    python -m venv venv
    source venv/bin/activate  # Or venv\Scripts\activate on Windows
    pip install -r requirements.txt

3. Configure Azure SQL connection:
    Create .streamlit/secrets.toml with the following:

    [sql_credentials]
    server = "your_server.database.windows.net"
    database = "your_database"
    username = "your_user"
    password = "your_password"

4. Run the app:
    streamlit run app.py

Technologies Used
-----------------
- Streamlit
- Pandas, Plotly, Altair
- PyDeck (for maps)
- Scikit-learn + Joblib (ML model)
- YData Profiling
- Azure SQL + pyodbc

Security Note
-------------
Be sure to exclude secrets.toml and .pkl model files from GitHub. Use a .gitignore like:

    .streamlit/secrets.toml
    *.pkl
    __pycache__/

Future Ideas
------------
- Authentication layer
- Executive summary tab
- Telegram Bot integration
- Power BI dashboard embedding
- Arabic language support

Author
------
Built by Ahmed Hany.
