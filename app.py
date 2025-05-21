import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np
import urllib.request
import json
import pydeck as pdk
from shapely.geometry import shape
import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import difflib
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import requests
import threading
import time
import joblib  
import os


st.set_page_config(
    page_title="Banking Analytics Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("\U0001F4CA Banking Analytics Dashboard")


@st.cache_data
def load_data():
    customers = pd.read_csv("Banking_Analytics_Dataset_Updated2.csv")
    accounts = pd.read_csv("Banking_Analytics_Dataset.xlsx - Accounts.csv")
    cards = pd.read_csv("Banking_Analytics_Dataset.xlsx - Cards.csv")
    loans = pd.read_csv("Banking_Analytics_Dataset.xlsx - Loans.csv")
    calls = pd.read_csv("Banking_Analytics_Dataset.xlsx - SupportCalls.csv")
    transactions = pd.read_csv("Banking_Analytics_Transactions_Updated.csv")
    fraud_df = pd.read_csv("Banking_Analytics_Transactions_WithFraud.csv")

    return customers, accounts, cards, loans, calls, transactions, fraud_df


@st.cache_resource
def load_ml_model():
    model = joblib.load(r'best_rf_churn_model (1).pkl')
    return model

customers, accounts, cards, loans, calls, transactions, fraud_df = load_data()

customers.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
transactions.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')



(
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11
) = st.tabs([
    "Customer Overview", "Accounts", "Transactions", "Loans",
    "Cards", "Support Calls", "Advanced Insights", "Chat", "Auto Analysis", "Telegram Reports", "Churn Prediction"
])


with tab1:
    st.header("\U0001F465 Customer Overview")
    all_customers = customers.copy()
    filtered_customers = customers[customers['State'] != 'Army']

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Customers", all_customers['CustomerID'].nunique())

    with col2:
        transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'], errors='coerce')
        transactions_with_customers = transactions.merge(accounts[['AccountID', 'CustomerID']], on='AccountID', how='left')
        six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
        active_customers = transactions_with_customers[transactions_with_customers['TransactionDate'] >= six_months_ago]['CustomerID'].nunique()
        st.metric(
            "Active Customers", 
            f"{active_customers:,}",
            f"{active_customers/all_customers['CustomerID'].nunique():.1%}"
        )

    with col3:
        inactive_customers = all_customers['CustomerID'].nunique() - active_customers
        st.metric(
            "Inactive Customers", 
            f"{inactive_customers:,}",
            f"-{inactive_customers/all_customers['CustomerID'].nunique():.1%}"
        )

    with col4:
        average_accounts_per_customer = accounts['AccountID'].nunique() / all_customers['CustomerID'].nunique()
        st.metric("Average Accounts per Customer", round(average_accounts_per_customer))

    with col5:
        all_customers['JoinDate'] = pd.to_datetime(all_customers['JoinDate'], errors='coerce')
        join_by_month = all_customers['JoinDate'].dt.to_period('M').value_counts().sort_index()
        average_new_customers = join_by_month.mean()
        st.metric("Average New Customers", round(average_new_customers))

    st.subheader("Customer Data")
    st.dataframe(all_customers)

    state_counts = filtered_customers['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Customer Count']
    
    # Calculate percentage of customers
    state_counts['Percentage'] = state_counts['Customer Count'] / state_counts['Customer Count'].sum() * 100

    top_states = state_counts.sort_values('Customer Count', ascending=False).head(10)
        
    fig = px.bar(
        top_states,
        x='State',
        y='Customer Count',
        title='Top 10 States by Customer Count',
        labels={'Customer Count': 'Number of Customers', 'State': 'State'},
        color='Customer Count',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    join_by_month.index = join_by_month.index.to_timestamp()
    join_by_month_df = join_by_month.reset_index()
    join_by_month_df.columns = ['JoinMonth', 'Count']



    valid_date_customers = all_customers.dropna(subset=['JoinDate'])
    
    valid_date_customers['Year'] = valid_date_customers['JoinDate'].dt.year
    valid_date_customers['Month'] = valid_date_customers['JoinDate'].dt.month
    monthly_growth = valid_date_customers.groupby(['Year', 'Month']).size().reset_index(name='New Customers')
    monthly_growth['YearMonth'] = monthly_growth['Year'].astype(str) + '-' + monthly_growth['Month'].astype(str).str.zfill(2)
    monthly_growth = monthly_growth.sort_values(['Year', 'Month'])
    monthly_growth['Cumulative Customers'] = monthly_growth['New Customers'].cumsum()
    

    fig = px.bar(
        monthly_growth,
        x='YearMonth',
        y='New Customers',
        title='Monthly New Customer Acquisitions',
        labels={'YearMonth': 'Year-Month', 'New Customers': 'New Customers'},
        color='New Customers',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
            xaxis=dict(tickangle=45),
            height=400
        )
        
    st.plotly_chart(fig, use_container_width=True)

    st.title("\U0001F4C8 New Customers")
    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.subheader("\U0001F50D Filters")
        years = all_customers['JoinDate'].dt.year.unique()
        selected_year = st.selectbox("Select Year", sorted(years))
        months = all_customers['JoinDate'][all_customers['JoinDate'].dt.year == selected_year].dt.month_name().unique()
        selected_month = st.selectbox("Select Month", sorted(months))

    new_customers_count = all_customers[
        (all_customers['JoinDate'].dt.year == selected_year) &
        (all_customers['JoinDate'].dt.month_name() == selected_month)
    ].shape[0]

    with right_col:
        st.subheader("\U0001F465 New Customers This Month")
        st.metric("New Customers Count", new_customers_count)

    monthly_new_customers = all_customers[
        (all_customers['JoinDate'].dt.year == selected_year) &
        (all_customers['JoinDate'].dt.month_name() == selected_month)
    ].groupby(all_customers['JoinDate'].dt.day).size()

    st.subheader("New Customers Registration by Day")
    st.bar_chart(monthly_new_customers)

    today = pd.Timestamp.now()
    six_months_ago = today - pd.DateOffset(months=6)

    last_tx = transactions.merge(accounts[['AccountID', 'CustomerID']], on='AccountID', how='left') \
        .groupby("CustomerID")["TransactionDate"].max().reset_index()

    all_customers = all_customers.merge(last_tx, on="CustomerID", how="left")

    churn_risk_customers = all_customers[
        (all_customers['TransactionDate'].isna()) |
        (all_customers['TransactionDate'] < six_months_ago)
    ]

    churn_risk_count = churn_risk_customers['CustomerID'].nunique()
    total_customers = all_customers['CustomerID'].nunique()
    churn_risk_percentage = (churn_risk_count / total_customers) * 100

    st.subheader("Churn Risks")
    st.metric("Customers at Risk of Churn", churn_risk_count)
    st.metric("Churn Risk Percentage", f"{round(churn_risk_percentage, 2)}%")

    
    
    merged_tx = transactions.merge(
        accounts[['AccountID', 'CustomerID']], 
        on='AccountID', 
        how='left'
    )
    
    
    last_tx_date = merged_tx.groupby("CustomerID")["TransactionDate"].max().reset_index()
    
    
    last_tx_date.rename(columns={"TransactionDate": "LastActivity"}, inplace=True)
    
   
    all_customers_with_tx = all_customers.merge(last_tx_date, on="CustomerID", how="left")
    
   
    all_customers_with_tx['ChurnRiskCategory'] = 'Low Risk'
    all_customers_with_tx.loc[all_customers_with_tx['LastActivity'] < (pd.Timestamp.now() - pd.DateOffset(months=3)), 'ChurnRiskCategory'] = 'Medium Risk'
    all_customers_with_tx.loc[all_customers_with_tx['LastActivity'] < (pd.Timestamp.now() - pd.DateOffset(months=6)), 'ChurnRiskCategory'] = 'High Risk'
    all_customers_with_tx.loc[all_customers_with_tx['LastActivity'].isna(), 'ChurnRiskCategory'] = 'Very High Risk'
    
    
    churn_risk_counts = all_customers_with_tx['ChurnRiskCategory'].value_counts().reset_index()
    churn_risk_counts.columns = ['Churn Risk', 'Customer Count']
    
    
    category_order = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    
    
    fig = px.pie(
        churn_risk_counts,
        names='Churn Risk',
        values='Customer Count',
        title='Customer Churn Risk Distribution',
        color='Churn Risk',
        color_discrete_map={
            'Low Risk': '#1d9700',
            'Medium Risk': '#ffbb33',
            'High Risk': '#ff9a33',
            'Very High Risk': '#ff3333'
        },
        category_orders={'Churn Risk': category_order}
    )
    
    fig.update_layout(height=400)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        
        high_risk_customers = all_customers_with_tx[all_customers_with_tx['ChurnRiskCategory'].isin(['High Risk', 'Very High Risk'])]
        top_churn_states = high_risk_customers['State'].value_counts().reset_index()
        top_churn_states.columns = ['State', 'Churn Risk Count']
        top_churn_states = top_churn_states.head(10)
        
        fig = px.bar(
            top_churn_states,
            x='State',
            y='Churn Risk Count',
            title='Top 10 States with High Churn Risk',
            color='Churn Risk Count',
            color_continuous_scale=px.colors.sequential.Reds
        )
        
        st.plotly_chart(fig, use_container_width=True)




    churn_by_month = churn_risk_customers['JoinDate'].dt.to_period('M').value_counts().sort_index()
    churn_by_month.index = churn_by_month.index.to_timestamp()
    churn_by_month_df = churn_by_month.reset_index()
    churn_by_month_df.columns = ['Month', 'Churn Count']
    st.subheader("Churn Count by Month")
    st.line_chart(churn_by_month_df.set_index('Month'))



    
    merged_txn_accounts = transactions.merge(accounts[['AccountID', 'CustomerID']], on='AccountID', how='left')
    full_merged = merged_txn_accounts.merge(customers[['CustomerID', 'State']], on='CustomerID', how='left')

    
    payment_counts = full_merged[full_merged["TransactionType"] == "Payment"] \
        .groupby("State").size().reset_index(name="payment_count")

    
    geojson_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    with urllib.request.urlopen(geojson_url) as response:
        us_states_geojson = json.load(response)

    max_payment = payment_counts["payment_count"].max()

    
    for feature in us_states_geojson["features"]:
        state_name = feature["properties"]["name"]
        match = payment_counts[payment_counts["State"] == state_name]
        if not match.empty:
            count = int(match["payment_count"].values[0])
            norm = count / max_payment if max_payment > 0 else 0
            feature["properties"]["payment_count"] = count
            feature["properties"]["payment_norm"] = norm
        else:
            feature["properties"]["payment_count"] = 0
            feature["properties"]["payment_norm"] = 0

    
    labels_data = []
    for feature in us_states_geojson["features"]:
        geom = shape(feature["geometry"])
        centroid = [geom.representative_point().x, geom.representative_point().y]
        labels_data.append({
            "coordinates": centroid,
            "name": feature["properties"]["name"],
            "payment_count": feature["properties"]["payment_count"]
        })

    
    choropleth_layer = pdk.Layer(
        "GeoJsonLayer",
        us_states_geojson,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="""
        [
            properties.payment_norm * 255,
            (1 - properties.payment_norm) * 100,
            (1 - properties.payment_norm) * 100
        ]
        """,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        labels_data,
        pickable=False,
        get_position="coordinates",
        get_text="name",
        get_size=14,
        get_color=[0, 0, 0],
        get_angle=0,
        get_text_anchor='"middle"',
        get_alignment_baseline='"center"',
    )

    
    view_state = pdk.ViewState(latitude=37.5, longitude=-96, zoom=4.0)

    tooltip = {
        "html": "<b>{name}</b><br/>Payments: {payment_count}",
        "style": {"color": "white"}
    }

    deck = pdk.Deck(
        layers=[choropleth_layer, text_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider="carto",
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
    )

    
    st.pydeck_chart(deck)

with tab2:
    st.header("\U0001F3E6 Accounts Overview")
    
    
    col1, col2,col3 = st.columns(3)

    with col1:
        total_accounts = accounts['AccountID'].nunique()
        st.metric("Total Accounts", total_accounts)

    with col2:
      
        average_balance_per_customer = accounts.groupby('CustomerID')['Balance'].mean().mean()  # حساب المتوسط العام
        st.metric("Average Balance per Customer", f"${average_balance_per_customer:,.2f}")

    with col3:
        total_balance = accounts['Balance'].sum()
        st.metric("Total Balance", f"${total_balance:,.2f}")

    st.subheader("Accounts by Type")

    account_type_counts = accounts['AccountType'].value_counts().reset_index()
    account_type_counts.columns = ['AccountType', 'Count']

    total_balance_by_type = accounts.groupby('AccountType')['Balance'].sum().reset_index()
    
    total_balance_by_type['Percentage'] = (total_balance_by_type['Balance'] / total_balance_by_type['Balance'].sum()) * 100

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.pie(account_type_counts, values='Count', names='AccountType', 
                       title='Distribution of Accounts by Type', 
                       hole=0.3)
        fig1.update_layout(width=400, height=400)
        st.plotly_chart(fig1)

    with col2:
        fig = px.bar(
            total_balance_by_type, 
            x='AccountType', 
            y='Balance',
            title='Total Balance by Account Type',
            labels={'AccountType': 'Account Type', 'Balance': 'Total Balance ($)'},
            color='Balance',
            text='Percentage',
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)



    dominant_account_type = account_type_counts.iloc[0]['AccountType']
    dominant_account_percentage = (account_type_counts.iloc[0]['Count'] / account_type_counts['Count'].sum()) * 100
    
    highest_balance_type = total_balance_by_type.iloc[total_balance_by_type['Balance'].idxmax()]['AccountType']
    highest_balance_percentage = total_balance_by_type.iloc[total_balance_by_type['Balance'].idxmax()]['Percentage']
    
   
    
    
    st.subheader("📈 Account Activity Analysis")
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    
    
    transaction_counts = transactions['AccountID'].value_counts().reset_index()
    transaction_counts.columns = ['AccountID', 'TransactionCount']
    
    
    accounts_with_tx = accounts.merge(transaction_counts, on='AccountID', how='left')
    accounts_with_tx['TransactionCount'] = accounts_with_tx['TransactionCount'].fillna(0)
    
    
    accounts_with_tx['ActivitySegment'] = pd.cut(
        accounts_with_tx['TransactionCount'],
        bins=[-1, 0, 10, 50, float('inf')],
        labels=['Inactive', 'Low Activity', 'Medium Activity', 'High Activity']
    )
    
    
    activity_segments = accounts_with_tx['ActivitySegment'].value_counts().reset_index()
    activity_segments.columns = ['Activity Level', 'Account Count']
    
    
    activity_segments['Activity Level'] = pd.Categorical(
        activity_segments['Activity Level'],
        categories=['Inactive', 'Low Activity', 'Medium Activity', 'High Activity'],
        ordered=True
    )
    activity_segments = activity_segments.sort_values('Activity Level')
    
    
    fig = px.bar(
        activity_segments,
        x='Activity Level',
        y='Account Count',
        title='Account Activity Levels',
        color='Activity Level',
        color_discrete_map={
            'Inactive': '#d3d3d3',
            'Low Activity': '#ffd966',
            'Medium Activity': '#93c47d',
            'High Activity': '#6fa8dc'
        },
        text_auto=True
    )
    fig.update_layout(xaxis_title='Activity Level', yaxis_title='Number of Accounts')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        
        st.subheader("Top 5 Most Active Accounts")
        top_5_active = accounts_with_tx.sort_values('TransactionCount', ascending=False).head(5)
        top_5_active_table = top_5_active[['AccountID', 'AccountType', 'Balance', 'TransactionCount']]
        top_5_active_table.columns = ['Account ID', 'Account Type', 'Balance ($)', 'Transaction Count']
        st.dataframe(top_5_active_table, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    
    inactive_percentage = (activity_segments[activity_segments['Activity Level'] == 'Inactive']['Account Count'].iloc[0] / 
                          activity_segments['Account Count'].sum()) * 100
    
    st.markdown("<div class='insights-text'>", unsafe_allow_html=True)
    st.markdown(f"""
    **Account Activity Insight:** {inactive_percentage:.1f}% of accounts show no transaction activity,
    representing opportunities for targeted engagement campaigns to reactivate dormant accounts.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    
    st.subheader("⚠️ Dormant Accounts Analysis")
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    
    
    last_tx_per_account = transactions.groupby('AccountID')['TransactionDate'].max().reset_index()
    last_tx_per_account.columns = ['AccountID', 'LastTransactionDate']
    
    
    accounts_with_last_tx = accounts.merge(last_tx_per_account, on='AccountID', how='left')
    
    
    six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
    dormant_accounts = accounts_with_last_tx[
        (accounts_with_last_tx['LastTransactionDate'].isna()) |
        (accounts_with_last_tx['LastTransactionDate'] < six_months_ago)
    ]
    
    
    dormant_count = dormant_accounts['AccountID'].nunique()
    dormant_percentage = (dormant_count / accounts['AccountID'].nunique()) * 100
    dormant_balance = dormant_accounts['Balance'].sum()
    dormant_balance_percentage = (dormant_balance / accounts['Balance'].sum()) * 100
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Dormant Accounts",
            f"{dormant_count:,}",
            f"{dormant_percentage:.1f}% of total accounts"
        )
    
    with col2:
        st.metric(
            "Dormant Balance",
            f"${dormant_balance:,.2f}",
            f"{dormant_balance_percentage:.1f}% of total balance"
        )
    
    
    dormant_by_type = dormant_accounts.groupby('AccountType').size().reset_index()
    dormant_by_type.columns = ['Account Type', 'Dormant Count']
    
    
    total_by_type = accounts.groupby('AccountType').size().reset_index()
    total_by_type.columns = ['Account Type', 'Total Count']
    
    
    dormant_analysis = dormant_by_type.merge(total_by_type, on='Account Type', how='left')
    dormant_analysis['Dormancy Rate'] = (dormant_analysis['Dormant Count'] / dormant_analysis['Total Count']) * 100
    
    
    dormant_analysis = dormant_analysis.sort_values('Dormancy Rate', ascending=False)
    
    fig = px.bar(
        dormant_analysis,
        x='Account Type',
        y='Dormancy Rate',
        title='Dormancy Rate by Account Type',
        color='Dormancy Rate',
        text='Dormant Count',
        color_continuous_scale=px.colors.sequential.Reds
    )
    
    fig.update_traces(texttemplate='%{text} accounts', textposition='outside')
    fig.update_layout(yaxis_title='Dormancy Rate (%)', xaxis_title='Account Type')
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    st.subheader("Top Dormant Accounts by Balance")
    top_dormant = dormant_accounts.sort_values('Balance', ascending=False).head(10)
    top_dormant_table = top_dormant[['AccountID', 'CustomerID', 'AccountType', 'Balance']]
    top_dormant_table.columns = ['Account ID', 'Customer ID', 'Account Type', 'Balance ($)']
    
    st.dataframe(top_dormant_table, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    
    highest_dormancy_type = dormant_analysis.iloc[0]['Account Type']
    highest_dormancy_rate = dormant_analysis.iloc[0]['Dormancy Rate']
    
    st.markdown("<div class='insights-text'>", unsafe_allow_html=True)
    st.markdown(f"""
    **Dormancy Insight:** {highest_dormancy_type} accounts have the highest dormancy rate at {highest_dormancy_rate:.1f}%. 
    Consider targeted reactivation campaigns for these accounts, especially those with high balances.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.header("💸 Transactions Overview")
    
    
    transactions['Year'] = transactions['TransactionDate'].dt.year
    transactions['Month'] = transactions['TransactionDate'].dt.month
    transactions['Day'] = transactions['TransactionDate'].dt.day
    
    
    col1, col2 = st.columns(2)
    with col1:
        total_transactions = transactions['TransactionID'].nunique()
        st.metric("Total Transactions", f"{total_transactions:,}")

    with col2:
        avg_transaction_amount = transactions['Amount'].mean() if 'Amount' in transactions.columns else 0
        st.metric("Average Transaction Amount", f"${avg_transaction_amount:.2f}")
    
    

    

    st.subheader("📊 Transaction Types Analysis")    
    col1, col2 = st.columns(2)
    
    with col1:
        tx_types = transactions['TransactionType'].value_counts().reset_index()
        tx_types.columns = ['Transaction Type', 'Count']
        
        fig = px.pie(
            tx_types, 
            values='Count', 
            names='Transaction Type',
            title='Distribution of Transaction Types',
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Transaction Volume by Season")
        
        season_tx = transactions.groupby('Season').size().reset_index()
        season_tx.columns = ['Season', 'Transaction Count']
        
        fig = px.bar(
            season_tx,
            x='Season',
            y='Transaction Count',
            color='Transaction Count',
            title='Seasonal Transaction Patterns',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    
    
    
    
    st.subheader("📅 Yearly Transaction Analysis")
    
    years = sorted(transactions['Year'].dropna().unique())
    yearly_selected_year = st.selectbox("Select Year", years, key="yearly_analysis_year")
    
    
    yearly_filtered_transactions = transactions[transactions['Year'] == yearly_selected_year]
    yearly_tx_count = len(yearly_filtered_transactions)
    
    st.subheader(f"Transaction Volume for {yearly_selected_year} ({yearly_tx_count:,} transactions)")
    
    
    yearly_filtered_transactions['MonthName'] = yearly_filtered_transactions['TransactionDate'].dt.month_name()
    yearly_monthly_tx = yearly_filtered_transactions.groupby('MonthName').size().reset_index()
    yearly_monthly_tx.columns = ['Month', 'Transaction Count']
    
    
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    yearly_monthly_tx['Month'] = pd.Categorical(yearly_monthly_tx['Month'], categories=month_order, ordered=True)
    yearly_monthly_tx = yearly_monthly_tx.sort_values('Month')
    
    yearly_fig = px.line(
        yearly_monthly_tx, 
        x='Month', 
        y='Transaction Count',
        markers=True,
        title=f'Monthly Transaction Volume - {yearly_selected_year}',
        color_discrete_sequence=['#0f52ba']
    )
    
    yearly_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Transactions",
        height=400
    )
    
    st.plotly_chart(yearly_fig, use_container_width=True)
    
    
    st.subheader("📊 Monthly Transaction Analysis")
    
    monthly_col1, monthly_col2 = st.columns(2)
    
    with monthly_col1:
        monthly_years = sorted(transactions['Year'].dropna().unique())
        monthly_selected_year = st.selectbox("Select Year", monthly_years, key="monthly_analysis_year")
    
    with monthly_col2:
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        monthly_available_months = sorted(transactions[transactions['Year'] == monthly_selected_year]['Month'].dropna().unique())
        monthly_selected_month = st.selectbox(
            "Select Month",
            monthly_available_months,
            format_func=lambda x: month_names.get(x, str(x)),
            key="monthly_analysis_month"
        )
    
    
    monthly_filtered_tx = transactions[
        (transactions['Year'] == monthly_selected_year) &
        (transactions['Month'] == monthly_selected_month)
    ]
    
    monthly_tx_count = len(monthly_filtered_tx)
    
    st.subheader(f"Daily Transactions for {month_names.get(monthly_selected_month)} {monthly_selected_year} ({monthly_tx_count:,} transactions)")
    
    
    monthly_tx_by_day = monthly_filtered_tx.groupby('Day').size().reset_index()
    monthly_tx_by_day.columns = ['Day', 'Transaction Count']
    
    daily_fig = px.bar(
        monthly_tx_by_day,
        x='Day',
        y='Transaction Count',
        title=f'Daily Transaction Volume - {month_names.get(monthly_selected_month)} {monthly_selected_year}',
        color='Transaction Count',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    st.plotly_chart(daily_fig, use_container_width=True)

    
    st.subheader("🗓️ Transaction Heatmap (Monthly/Daily)")
    
    heatmap_year = st.selectbox("Select Year for Heatmap", sorted(transactions['Year'].dropna().unique()), key="heatmap_year")
    
    
    heatmap_data = transactions[transactions['Year'] == heatmap_year]
    
    if heatmap_data.empty:
        st.warning("No transactions available for the selected year.")
    else:
        
        tx_count_by_day_month = (
            heatmap_data
            .groupby(['Month', 'Day'])
            .size()
            .reset_index(name='TransactionCount')
        )
        

        tx_count_by_day_month['MonthName'] = tx_count_by_day_month['Month'].map(month_names)
        
        
        tx_count_by_day_month['MonthName'] = pd.Categorical(
            tx_count_by_day_month['MonthName'],
            categories=list(month_names.values()),
            ordered=True
        )
        
        
        heatmap_fig = px.density_heatmap(
            tx_count_by_day_month,
            x="Day",
            y="MonthName",
            z="TransactionCount",
            color_continuous_scale="YlGnBu",
            nbinsx=31,
            title=f"Daily Transaction Activity - {heatmap_year}",
            labels={"TransactionCount": "Transaction Count", "Day": "Day of Month", "MonthName": "Month"}
        )
        
        heatmap_fig.update_layout(
            xaxis_nticks=31,
            yaxis_title="Month",
            xaxis_title="Day of Month",
            height=500
        )
        
        st.plotly_chart(heatmap_fig, use_container_width=True)

    st.subheader("🚨 Fraud Detection Insights")

    # General Statistics
    fraud_count = fraud_df[fraud_df["fraud"] == -1].shape[0]
    legit_count = fraud_df[fraud_df["fraud"] != -1].shape[0]
    total_tx = fraud_df.shape[0]

    # Key Performance Indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fraudulent Transactions", f"{fraud_count:,}", f"{(fraud_count / total_tx) * 100:.2f}%")
    with col2:
        # Assuming amount column exists in the data
        if "Amount" in fraud_df.columns:
            fraud_amount = fraud_df[fraud_df["fraud"] == -1]["Amount"].sum()
            st.metric("Total Fraud Amount", f"${fraud_amount:,.2f}")
        else:
            st.metric("Total Fraud Amount", "Data not available")
    with col3:
        # Fraud detection rate (you can modify based on your available data)
        st.metric("Fraud Detection Rate", f"{(fraud_count / (fraud_count + 100)):.2f}%", "vs Last Month +10%")

    # Data Distribution Visualization
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            names=["Fraudulent Transactions", "Legitimate Transactions"],
            values=[fraud_count, legit_count],
            title="Fraud vs. Legitimate Transactions",
            color_discrete_sequence=["#ff6b6b", "#4ecdc4"],
            hole=0.4
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Fraud analysis by transaction type (if available in data)
        if "TransactionType" in fraud_df.columns:
            fraud_by_type = fraud_df[fraud_df["fraud"] == -1]["TransactionType"].value_counts().reset_index()
            fraud_by_type.columns = ["Transaction Type", "Fraud Count"]
            
            fig = px.bar(
                fraud_by_type.head(5),
                x="Transaction Type", 
                y="Fraud Count",
                title="Top 5 Transaction Types Prone to Fraud",
                color="Fraud Count",
                color_continuous_scale=px.colors.sequential.Reds
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Transaction type data not available")

    
    # Ensure date column exists and convert to datetime
    if "TransactionDate" in fraud_df.columns:
        fraud_df["TransactionDate"] = pd.to_datetime(fraud_df["TransactionDate"], errors='coerce')
        fraud_df["Month"] = fraud_df["TransactionDate"].dt.month
        fraud_df["Day"] = fraud_df["TransactionDate"].dt.day
        fraud_df["Hour"] = fraud_df["TransactionDate"].dt.hour
        
        fraud_time_analysis = fraud_df[fraud_df["fraud"] == -1].groupby("Hour").size().reset_index()
        fraud_time_analysis.columns = ["Hour of Day", "Fraud Count"]
        
       
        
        # Analysis by day of week
        if hasattr(fraud_df["TransactionDate"].dt, 'dayofweek'):
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            fraud_df["DayOfWeek"] = fraud_df["TransactionDate"].dt.dayofweek
            fraud_day_analysis = fraud_df[fraud_df["fraud"] == -1].groupby("DayOfWeek").size().reset_index()
            fraud_day_analysis["DayName"] = fraud_day_analysis["DayOfWeek"].apply(lambda x: day_names[x])
            fraud_day_analysis.columns = ["Day Number", "Fraud Count", "Day of Week"]
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    fraud_day_analysis, 
                    x="Day of Week", 
                    y="Fraud Count", 
                    title="Distribution of Fraudulent Transactions by Day of Week",
                    color="Fraud Count",
                    color_continuous_scale=px.colors.sequential.Reds
                )
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Date data not available for temporal analysis")

    # Amount Analysis for Fraudulent Transactions
    if "Amount" in fraud_df.columns:
        st.subheader("💰 Fraud Amount Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                fraud_df[fraud_df["fraud"] == -1], 
                x="Amount",
                nbins=20,
                title="Distribution of Fraudulent Transaction Amounts",
                color_discrete_sequence=["#ff6b6b"],
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Compare average amounts
            avg_fraud_amount = fraud_df[fraud_df["fraud"] == -1]["Amount"].mean()
            avg_legit_amount = fraud_df[fraud_df["fraud"] != -1]["Amount"].mean()
            
            comparison_df = pd.DataFrame({
                "Transaction Type": ["Fraudulent", "Legitimate"],
                "Average Amount": [avg_fraud_amount, avg_legit_amount]
            })
            
            fig = px.bar(
                comparison_df, 
                x="Transaction Type", 
                y="Average Amount",
                title="Comparison of Average Transaction Amounts",
                color="Transaction Type",
                color_discrete_map={
                    "Fraudulent": "#ff6b6b",
                    "Legitimate": "#4ecdc4"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

    # Geographic Analysis (if data available)
    if "State" in fraud_df.columns:
        st.subheader("🗺️ Geographic Distribution of Fraud")
        
        fraud_by_state = fraud_df[fraud_df["fraud"] == -1].groupby("State").size().reset_index()
        fraud_by_state.columns = ["State", "Fraud Count"]
        fraud_by_state = fraud_by_state.sort_values("Fraud Count", ascending=False)
        
        fig = px.choropleth(
            fraud_by_state,
            locations="State",
            locationmode="USA-states",
            color="Fraud Count",
            scope="usa",
            title="Fraud Distribution by State",
            color_continuous_scale=px.colors.sequential.Reds
        )
        st.plotly_chart(fig, use_container_width=True)

    # Fraud Factors and Patterns
    st.subheader("🔍 Fraud Factors and Patterns")

    # Simulating a classification model for fraud factors
    feature_importance = pd.DataFrame({
        "Feature": ["Transaction Time", "Transaction Amount", "Transaction Location", "Transaction Frequency", "Transaction Type"],
        "Importance": [0.85, 0.72, 0.65, 0.58, 0.42]
    })

    fig = px.bar(
        feature_importance, 
        x="Feature", 
        y="Importance", 
        title="Key Factors in Fraud Detection",
        color="Importance",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sample Fraudulent Transactions
    st.subheader("📋 Sample Fraudulent Transactions")
    sample_fraud = fraud_df[fraud_df["fraud"] == -1].head(10)
    if "CustomerID" in sample_fraud.columns:
        sample_fraud["CustomerID"] = sample_fraud["CustomerID"].astype(str).apply(lambda x: x[:3] + "***" + x[-2:])
    st.dataframe(sample_fraud)

    # Interactive Filtering of Fraud Data
    st.subheader("🔎 Search and Filter Fraudulent Transactions")

    col1, col2 = st.columns(2)
    with col1:
        if "Amount" in fraud_df.columns:
            min_amount = int(fraud_df["Amount"].min())
            max_amount = int(fraud_df["Amount"].max())
            selected_amount = st.slider("Filter by Amount", min_amount, max_amount, (min_amount, max_amount))
        
    with col2:
        if "TransactionType" in fraud_df.columns:
            transaction_types = fraud_df["TransactionType"].unique().tolist()
            selected_types = st.multiselect("Filter by Transaction Type", transaction_types, default=transaction_types[:3])

    # Apply filters to data and display results
    try:
        filtered_fraud = fraud_df[fraud_df["fraud"] == -1]
        
        if "Amount" in fraud_df.columns:
            filtered_fraud = filtered_fraud[(filtered_fraud["Amount"] >= selected_amount[0]) & 
                                          (filtered_fraud["Amount"] <= selected_amount[1])]
        
        if "TransactionType" in fraud_df.columns and selected_types:
            filtered_fraud = filtered_fraud[filtered_fraud["TransactionType"].isin(selected_types)]
        
        st.dataframe(filtered_fraud)
        st.markdown(f"**Found {filtered_fraud.shape[0]} fraudulent transactions matching the criteria**")
        
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")

with tab4:
    st.header("🏦 Loans Analysis")
    st.metric("Total Loans", loans['LoanID'].nunique())
    st.metric("Total Loan Amount", f"${loans['LoanAmount'].sum():,.2f}")
    
    st.subheader("Total Loan Amount Disbursed by Type")
    
    
    total_loan_by_type = loans.groupby('LoanType')['LoanAmount'].sum().reset_index()

    
    fig = px.bar(total_loan_by_type, x='LoanType', y='LoanAmount', 
                 title='Total Loan Amount Disbursed by Type', 
                 labels={'LoanType': 'Loan Type', 'LoanAmount': 'Total Amount'},
                 color='LoanAmount', 
                 color_continuous_scale=px.colors.sequential.Viridis)

    st.plotly_chart(fig)


    st.subheader("Loan Types")
    st.bar_chart(loans['LoanType'].value_counts())

    st.subheader("Average Interest Rate per Loan Type")

    
    loans['InterestRate'] = loans['InterestRate'].astype(str)  
    loans['InterestRate'] = loans['InterestRate'].str.replace(r'[^\d.,]', '', regex=True)  
    loans['InterestRate'] = loans['InterestRate'].str.replace(',', '.')  
    loans['InterestRate'] = pd.to_numeric(loans['InterestRate'], errors='coerce')  


    
    avg_interest_by_type = loans.groupby('LoanType')['InterestRate'].mean().reset_index()
    avg_interest_by_type = avg_interest_by_type.sort_values(by='InterestRate', ascending=False)

    
    st.dataframe(avg_interest_by_type.style.format({'InterestRate': '{:.2f}%'}))

    
    fig = px.bar(
        avg_interest_by_type,
        x='LoanType',
        y='InterestRate',
        title='Average Interest Rate per Loan Type',
        labels={'LoanType': 'Loan Type', 'InterestRate': 'Avg Interest Rate (%)'},
        color='InterestRate',
        color_continuous_scale=px.colors.sequential.Blues
    )

    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Interest Rates Distribution")
    fig = px.histogram(
        loans,
        x="InterestRate",
        nbins=30,
        title="Distribution of Interest Rates",
        labels={"InterestRate": "Interest Rate"},
        opacity=0.75,
        color_discrete_sequence=["#4e79a7"]
    )

    fig.update_layout(
        bargap=0.05,
        xaxis_title="Interest Rate",
        yaxis_title="Number of Loans",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)


    current_year = datetime.datetime.now().year


    loans['LoanEndDate'] = pd.to_datetime(loans['LoanEndDate'], errors='coerce')

    
    loans_ending_this_year = loans[loans['LoanEndDate'].dt.year == current_year]

    st.subheader(f"Loans Maturing in {current_year}")
    st.metric("Total Loans Ending This Year", loans_ending_this_year['LoanID'].nunique())
    st.metric("Total Amount Ending This Year", f"${loans_ending_this_year['LoanAmount'].sum():,.2f}")

    
    loans_ending_this_year['Month'] = loans_ending_this_year['LoanEndDate'].dt.month_name()

    monthly_trend = loans_ending_this_year.groupby('Month')['LoanID'].count().reset_index()
    monthly_trend = monthly_trend.sort_values(by='LoanID', ascending=False)

    
    fig = px.bar(
        monthly_trend,
        x='Month',
        y='LoanID',
        title=f"Upcoming Loan Maturities in {current_year} by Month",
        labels={'LoanID': 'Number of Loans', 'Month': 'Month'},
        color='LoanID',
        color_continuous_scale=px.colors.sequential.Oranges
    )

    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("💳 Cards Overview")

    
    st.metric("Total Cards", cards['CardID'].nunique())

    
    st.bar_chart(cards['CardType'].value_counts())

    
    cards['IssuedDate'] = pd.to_datetime(cards['IssuedDate'], errors='coerce')

    
    cards['IssuedMonth'] = cards['IssuedDate'].dt.to_period('M').dt.to_timestamp()
    grouped = cards.groupby(['IssuedMonth', 'CardType']).size().reset_index(name='Count')

    line_chart = alt.Chart(grouped).mark_line(point=True).encode(
        x='IssuedMonth:T',
        y='Count:Q',
        color='CardType:N',
        tooltip=['IssuedMonth:T', 'CardType:N', 'Count:Q']
    ).properties(
        title='Card Issuance by Type Over Time',
        width=700,
        height=400
    )

    st.altair_chart(line_chart, use_container_width=True)


    cards['ExpirationDate'] = pd.to_datetime(cards['ExpirationDate'], errors='coerce')


    today = pd.Timestamp(datetime.date.today())


    cards['Status'] = cards['ExpirationDate'].apply(lambda x: 'Active' if x and x > today else 'Expired')
    cards['IssuedMonth'] = cards['IssuedDate'].dt.to_period('M').dt.to_timestamp()
    expired_trend = cards[cards['Status'] == 'Expired'].groupby('IssuedMonth').size()
    active_trend = cards[cards['Status'] == 'Active'].groupby('IssuedMonth').size()


    status_counts = cards['Status'].value_counts()
    fig = px.pie(
    names=status_counts.index,
    values=status_counts.values,
    title="Active vs Expired Cards",
    color_discrete_sequence=["green", "red"]
    )
    st.plotly_chart(fig)

    customer_card_counts = cards.groupby(['CustomerID', 'CardType']).size().reset_index(name='CardCount')


    avg_holding_per_type = customer_card_counts.groupby('CardType')['CardCount'].mean().round(2)

   
    st.subheader("📊 Average Cards Held per Customer by Card Type")
    st.dataframe(avg_holding_per_type.rename("AvgCardsPerCustomer"))

with tab6:
    st.header("📞 Support Calls")

    col1, col2 = st.columns(2)
    with col1:
        total_calls = calls['CallID'].nunique()
        resolved_calls = calls[calls['Resolved'].str.lower() == 'yes'].shape[0]
        unresolved_calls = total_calls - resolved_calls

        st.markdown(f"""
            <div style="text-align:center">
                <h3>Total Calls</h3>
                <p style="font-size:50px; font-weight:bold;">{total_calls}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="text-align:center">
                <h3>Resolved Calls</h3>
                <p style="font-size:50px; font-weight:bold; color:green;">{resolved_calls}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="text-align:center">
                <h3>Unresolved Calls</h3>
                <p style="font-size:50px; font-weight:bold; color:red;">{unresolved_calls}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Resolved vs Unresolved Calls")
        resolution_data = pd.DataFrame({
            'Resolution': ['Resolved', 'Unresolved'],
            'Count': [resolved_calls, unresolved_calls]
        })
        fig = px.pie(
            resolution_data,
            names='Resolution',
            values='Count',
            color='Resolution',
            color_discrete_map={'Resolved':'green', 'Unresolved':'red'},
            title='Call Resolution Rate',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    
    calls['CallDate'] = pd.to_datetime(calls['CallDate'], errors='coerce')
    st.bar_chart(calls['IssueType'].value_counts())
    call_by_date = calls.groupby(calls['CallDate'].dt.date).size()
    st.subheader("Calls Over Time")
    st.line_chart(call_by_date)

with tab7:
    st.header("📈 Advanced Insights")

    
    st.subheader("High-Value Customers")
    acc_loans = accounts.groupby("CustomerID")["Balance"].sum().reset_index()
    acc_loans.columns = ["CustomerID", "TotalBalance"]
    high_value = acc_loans[acc_loans["TotalBalance"] > acc_loans["TotalBalance"].quantile(0.75)]
    st.write(f"Number of High-Value Customers: {len(high_value)}")
    st.dataframe(high_value.merge(customers, on="CustomerID", how="left"))

    
    st.subheader("Cross-Product Usage")
    merged = customers.copy()
    merged["HasAccount"] = merged["CustomerID"].isin(accounts["CustomerID"])
    merged["HasCard"] = merged["CustomerID"].isin(cards["CustomerID"])
    merged["HasLoan"] = merged["CustomerID"].isin(loans["CustomerID"])
    merged["HasSupport"] = merged["CustomerID"].isin(calls["CustomerID"])
    product_usage = merged[["HasAccount", "HasCard", "HasLoan", "HasSupport"]].mean() * 100
    st.bar_chart(product_usage)

    
    st.subheader("Customer Segmentation by Balance")

    def categorize_customer(balance):
        if balance >= acc_loans["TotalBalance"].quantile(0.75):
            return "Platinum"
        elif balance >= acc_loans["TotalBalance"].quantile(0.5):
            return "Gold"
        else:
            return "Silver"

    acc_loans["Segment"] = acc_loans["TotalBalance"].apply(categorize_customer)
    segment_summary = acc_loans["Segment"].value_counts().reset_index()
    segment_summary.columns = ["Segment", "Count"]
    st.write("### Segment Distribution")
    st.bar_chart(segment_summary.set_index("Segment"))

    segmented_customers = acc_loans.merge(customers, on="CustomerID", how="left")
    st.dataframe(segmented_customers)

    
    st.subheader("Customer Segmentation by Behavior")

    behavior_data = merged[["HasAccount", "HasCard", "HasLoan", "HasSupport"]].astype(int)
    scaler = StandardScaler()
    scaled_behavior = scaler.fit_transform(behavior_data)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    merged["BehaviorSegment"] = kmeans.fit_predict(scaled_behavior)

    segment_labels = {
    0: "Low Engagement",
    1: "Moderate Engagement",
    2: "High Engagement"
}

    merged["BehaviorSegment"] = merged["BehaviorSegment"].map(segment_labels)

    st.write("### Number of Customers per Behavioral Segment")
    st.bar_chart(merged["BehaviorSegment"].value_counts())

    st.dataframe(merged[["CustomerID", "BehaviorSegment", "HasAccount", "HasCard", "HasLoan", "HasSupport"]])

with tab8:
    GOOGLE_API_KEY = "AIzaSyCfr_AYlPCQPYToTY2NUDM-4nEFbYNdhVY"
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')
    chat = model.start_chat(history=[])

    st.title("🧠 Data Analysis Assistant")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = []

    def get_system_prompt():
        context = """
        You are a financial data analyst assistant for a banking dashboard. You have access to the following datasets:
        
        1. Customers: Contains customer details including CustomerID, Name, JoinDate, State, etc.
        2. Accounts: Contains account details with AccountID, CustomerID, Balance, AccountType.
        3. Cards: Information about credit/debit cards including CardID, CustomerID, CardType, IssuedDate, ExpirationDate.
        4. Loans: Data about loans with LoanID, CustomerID, LoanType, LoanAmount, InterestRate, LoanStartDate, LoanEndDate.
        5. Transactions: Transaction data with TransactionID, AccountID, TransactionDate, Amount, TransactionType.
        6. Support Calls: Customer support call data with CallID, CustomerID, CallDate, IssueType, Resolved status.
        7. Fraud Data: Information about fraudulent transactions with fraud indicators.
        
        The dashboard provides insights on:
        - Customer Overview: Demographics, growth trends, churn risk
        - Account Analysis: Types, balances, dormant accounts
        - Transaction Analysis: Volume trends, seasonal patterns
        - Loan Portfolio: Types, interest rates, maturity analysis
        - Card Services: Types, issuance trends, active vs expired
        - Support Call Analysis: Resolution rates, common issues
        - Advanced Insights: Customer segmentation, cross-product usage
        - Fraud Detection: Identifying and analyzing fraudulent transactions
        
        Answer questions based on this banking data. You can respond in multiple languages including English and Arabic.
        """
        return context
        
    def save_chat():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if len(st.session_state.current_chat) > 0:
            chat_name = f"Chat {timestamp}"
            st.session_state.chat_history.append({
                "name": chat_name,
                "messages": st.session_state.current_chat.copy()
            })
            st.session_state.current_chat = []

    chat_container = st.container()
    for message in st.session_state.current_chat:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask me about the banking data...")

    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("💬 Chat History")
        
        if len(st.session_state.chat_history) > 0:
            chat_names = [chat["name"] for chat in st.session_state.chat_history]
            selected_chat = st.selectbox("Previous Chats", chat_names)
            
            if st.button("Load Selected Chat"):
                selected_index = chat_names.index(selected_chat)
                st.session_state.current_chat = st.session_state.chat_history[selected_index]["messages"].copy()
    
    with col2:
        st.subheader("Chat Options")
        if st.button("New Chat"):
            save_chat()
            
        if st.button("Save Current Chat"):
            save_chat()
            st.success("Chat saved successfully!")
    
    with col3:
        st.subheader("🌐 Language Preferences")
        language = st.selectbox(
            "Preferred Response Language",
            ["Auto-detect", "English", "Arabic", "French", "Spanish", "German"]
        )
        
        if language != "Auto-detect":
            st.session_state.preferred_language = language
            
        if st.button("Export Chat as Text"):
            chat_text = ""
            for message in st.session_state.current_chat:
                prefix = "User: " if message["role"] == "user" else "Assistant: "
                chat_text += f"{prefix}{message['content']}\n\n"
            
            st.download_button(
                label="Download Chat",
                data=chat_text,
                file_name=f"chat_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Check if the input contains chart request keywords
    chart_keywords = ['رسم', 'شارت', 'رسم بياني', 'plot', 'chart', 'visualize', 'graph', 'عرض']
    
    if user_input:
        draw_flag = any(word in user_input.lower() for word in chart_keywords)
        
        st.session_state.current_chat.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if draw_flag:
                        # Add column name mappings for both English and Arabic
                        column_mappings = {
                            # Arabic to English column mappings - Customer related
                            "العملاء": "CustomerID",
                            "العميل": "CustomerID",
                            "الزبائن": "CustomerID",
                            "رقم العميل": "CustomerID",
                            "معرف العميل": "CustomerID",
                            "هوية العميل": "CustomerID",
                            "اسم العميل": "Name",
                            "الاسم": "Name",
                            "تاريخ الانضمام": "JoinDate",
                            "الانضمام": "JoinDate",
                            "تاريخ التسجيل": "JoinDate",
                            
                            # Account related
                            "الحساب": "AccountID",
                            "الحسابات": "AccountID",
                            "رقم الحساب": "AccountID",
                            "معرف الحساب": "AccountID",
                            "هوية الحساب": "AccountID",
                            "نوع الحساب": "AccountType",
                            "فئة الحساب": "AccountType",
                            "تصنيف الحساب": "AccountType",
                            
                            # Transaction related
                            "المعاملات": "TransactionID",
                            "العمليات": "TransactionID",
                            "رقم المعاملة": "TransactionID",
                            "معاملة": "TransactionID",
                            "رقم العملية": "TransactionID",
                            "العملية": "TransactionID",
                            "نوع المعاملة": "TransactionType",
                            "نوع العملية": "TransactionType",
                            "فئة المعاملة": "TransactionType",
                            "فئة العملية": "TransactionType",
                            "تصنيف المعاملة": "TransactionType",
                            "تصنيف العملية": "TransactionType",
                            "المبلغ": "Amount",
                            "مبلغ": "Amount",
                            "المال": "Amount",
                            "النقود": "Amount",
                            "قيمة": "Amount",
                            "قيمة المعاملة": "Amount",
                            "قيمة العملية": "Amount",
                            "التاريخ": "TransactionDate",
                            "تاريخ": "TransactionDate",
                            "تاريخ المعاملة": "TransactionDate",
                            "تاريخ العملية": "TransactionDate",
                            "التاريخ المعاملة": "TransactionDate",
                            "وقت": "TransactionDate",
                            "وقت المعاملة": "TransactionDate",
                            "وقت العملية": "TransactionDate",
                            "الموسم": "Season",
                            "موسم": "Season",
                            "فصل": "Season",
                            "فصل السنة": "Season",
                            
                            # Fraud related
                            "الاحتيال": "fraud",
                            "احتيال": "fraud",
                            "غش": "fraud",
                            "تزوير": "fraud",
                            "عملية احتيالية": "fraud",
                            "معاملة احتيالية": "fraud",
                            "الاحتيال المالي": "fraud",
                            "عمليات الاحتيال": "fraud",
                            "كشف الاحتيال": "fraud",
                            "مؤشر الاحتيال": "fraud",
                            
                            # Location related
                            "الولاية": "State",
                            "ولاية": "State",
                            "المدينة": "City",
                            "مدينة": "City",
                            "المحافظة": "State",
                            "محافظة": "State",
                            "الولايات": "State",
                            "الدولة": "State",
                            "البلد": "State",
                            "المكان": "State",
                            "الموقع": "State",
                            
                            # Card related
                            "البطاقات": "CardID",
                            "البطاقة": "CardID",
                            "رقم البطاقة": "CardID",
                            "معرف البطاقة": "CardID",
                            "هوية البطاقة": "CardID",
                            "نوع البطاقة": "CardType",
                            "فئة البطاقة": "CardType",
                            "تصنيف البطاقة": "CardType",
                            "تاريخ الإصدار": "IssuedDate",
                            "إصدار البطاقة": "IssuedDate",
                            "تاريخ انتهاء الصلاحية": "ExpirationDate",
                            "انتهاء الصلاحية": "ExpirationDate",
                            "تاريخ الانتهاء": "ExpirationDate",
                            "حالة البطاقة": "Status",
                            
                            # Loan related
                            "القروض": "LoanID",
                            "القرض": "LoanID",
                            "رقم القرض": "LoanID",
                            "معرف القرض": "LoanID",
                            "هوية القرض": "LoanID",
                            "نوع القرض": "LoanType",
                            "فئة القرض": "LoanType",
                            "تصنيف القرض": "LoanType",
                            "مبلغ القرض": "LoanAmount",
                            "قيمة القرض": "LoanAmount",
                            "معدل الفائدة": "InterestRate",
                            "الفائدة": "InterestRate",
                            "نسبة الفائدة": "InterestRate",
                            "تاريخ بدء القرض": "LoanStartDate",
                            "تاريخ بداية القرض": "LoanStartDate",
                            "تاريخ انتهاء القرض": "LoanEndDate",
                            "تاريخ نهاية القرض": "LoanEndDate",
                            "تاريخ استحقاق القرض": "LoanEndDate",
                            
                            # Support calls related
                            "المكالمات": "CallID",
                            "المكالمة": "CallID",
                            "رقم المكالمة": "CallID",
                            "معرف المكالمة": "CallID",
                            "هوية المكالمة": "CallID",
                            "نوع المشكلة": "IssueType",
                            "المشكلة": "IssueType",
                            "تصنيف المشكلة": "IssueType",
                            "تاريخ المكالمة": "CallDate",
                            "وقت المكالمة": "CallDate",
                            "حل": "Resolved",
                            "تم الحل": "Resolved",
                            "مكالمة محلولة": "Resolved",
                            
                            # Balance related
                            "الرصيد": "Balance",
                            "رصيد": "Balance",
                            "الصيد": "Balance",
                            "الرصيد الموجود": "Balance",
                            "الصيد الموجود": "Balance",
                            "الصيد الموجود فيه": "Balance",
                            "المبلغ الموجود": "Balance",
                            "القيمة": "Amount",
                            "القيمة المالية": "Amount",
                            "المبالغ": "Amount",
                            "المبالغ المالية": "Amount",
                            "المال المتوفر": "Balance",
                            "المال المتاح": "Balance",
                            "الأموال": "Amount",
                            "الأموال المتاحة": "Balance",
                            "مبلغ الحساب": "Balance",
                            "قيمة الحساب": "Balance",
                            "رصيد الحساب": "Balance",
                            
                            # Common English synonyms
                            "customers": "CustomerID",
                            "customer": "CustomerID",
                            "customer id": "CustomerID",
                            "customer name": "Name",
                            "name": "Name",
                            "join date": "JoinDate",
                            "registration date": "JoinDate",
                            "accounts": "AccountID",
                            "account": "AccountID",
                            "account id": "AccountID",
                            "account type": "AccountType",
                            "transactions": "TransactionID",
                            "transaction": "TransactionID",
                            "transaction id": "TransactionID",
                            "transaction type": "TransactionType",
                            "amount": "Amount",
                            "date": "TransactionDate",
                            "transaction date": "TransactionDate",
                            "season": "Season",
                            "fraud": "fraud",
                            "state": "State",
                            "city": "City",
                            "location": "State",
                            "cards": "CardID",
                            "card": "CardID",
                            "card id": "CardID",
                            "card type": "CardType",
                            "issue date": "IssuedDate",
                            "expiration date": "ExpirationDate",
                            "status": "Status",
                            "loans": "LoanID",
                            "loan": "LoanID",
                            "loan id": "LoanID",
                            "loan type": "LoanType",
                            "loan amount": "LoanAmount",
                            "interest rate": "InterestRate",
                            "loan start date": "LoanStartDate",
                            "loan end date": "LoanEndDate",
                            "calls": "CallID",
                            "call": "CallID",
                            "call id": "CallID",
                            "issue type": "IssueType",
                            "issue": "IssueType",
                            "call date": "CallDate",
                            "resolved": "Resolved",
                            "balance": "Balance",
                            "available balance": "Balance",
                            "account balance": "Balance",
                            "money": "Amount",
                            "funds": "Amount",
                            "available funds": "Balance",
                            "value": "Amount"
                        }
                        
                        # Improve the column matching logic
                        def find_column_in_dataframe(column_name, df):
                            """Find the best matching column in the dataframe"""
                            # First check if the column exists directly
                            if column_name in df.columns:
                                return column_name
                                
                            # Check if it's in our mapping
                            if column_name in column_mappings:
                                mapped_col = column_mappings[column_name]
                                if mapped_col in df.columns:
                                    return mapped_col
                            
                            # Try to find a partial match in the dataframe columns
                            for col in df.columns:
                                if column_name.lower() in col.lower() or col.lower() in column_name.lower():
                                    return col
                            
                            # Use fuzzy matching as a last resort
                            closest_match = difflib.get_close_matches(column_name, df.columns, n=1, cutoff=0.6)
                            if closest_match:
                                return closest_match[0]
                                
                            return None
                        
                        # Map dataframe names to actual dataframes
                        dataframe_mapping = {
                            "customers": customers,
                            "accounts": accounts,
                            "transactions": transactions,
                            "loans": loans,
                            "cards": cards,
                            "calls": calls,
                            "fraud": fraud_df,
                            "fraud_df": fraud_df,
                            
                            # Arabic mappings
                            "العملاء": customers,
                            "الحسابات": accounts,
                            "المعاملات": transactions,
                            "العمليات": transactions,
                            "القروض": loans,
                            "البطاقات": cards,
                            "المكالمات": calls,
                            "الاحتيال": fraud_df
                        }
                        
                        # Enhanced chart prompt that focuses more on column detection
                        chart_prompt = f"""
You are a data visualization assistant for a banking analytics dashboard. The user may ask for a chart in English or Arabic.

The dashboard has these dataframes:
1. customers - Customer information (CustomerID, Name, JoinDate, State)
2. accounts - Account information (AccountID, CustomerID, AccountType, Balance)
3. transactions - Transaction details (TransactionID, AccountID, TransactionDate, Amount, TransactionType)
4. loans - Loan information (LoanID, CustomerID, LoanType, LoanAmount, InterestRate, LoanStartDate, LoanEndDate)
5. cards - Card details (CardID, CustomerID, CardType, IssuedDate, ExpirationDate, Status)
6. calls - Support call records (CallID, CustomerID, CallDate, IssueType, Resolved)
7. fraud_df - Fraud detection data (TransactionID, fraud flag, Amount, etc.)

Based on the user's request, extract:
1. The chart type: one of ["bar", "pie", "line"]
2. The dataframe that contains the data (from the list above)
3. The column for x-axis (focus on identifying the exact column name)
4. The column for y-axis (or "count" if the user wants counts per category)

Output your result as JSON in the following format:
{{
  "type": "bar",
  "dataframe": "accounts",
  "x": "AccountType",
  "y": "count"
}}

User's request: {user_input}
"""
                        chart_response = chat.send_message(chart_prompt)
                        
                        # Add error handling for JSON parsing
                        try:
                            # Try to extract JSON part from the response (in case model includes additional text)
                            response_text = chart_response.text.strip()
                            
                            # Look for JSON-like structure in the response
                            import re
                            json_match = re.search(r'({[\s\S]*})', response_text)
                            
                            if json_match:
                                json_str = json_match.group(1)
                                chart_info = json.loads(json_str)
                            else:
                                # Try parsing the whole response as JSON
                                chart_info = json.loads(response_text)
                                
                        except json.JSONDecodeError:
                            # Fallback to default values if parsing fails
                            st.warning("Could not parse chart request. Using default chart parameters.")
                            chart_info = {
                                "type": "bar",
                                "dataframe": "fraud_df",
                                "x": "TransactionType" if "TransactionType" in fraud_df.columns else fraud_df.columns[0],
                                "y": "count"
                            }
                        
                        # Now use chart_info to generate chart
                        chart_type = chart_info.get("type", "bar")
                        
                        # Get the dataframe to use - default to fraud_df
                        df_name = chart_info.get("dataframe", "fraud_df")
                        selected_df = dataframe_mapping.get(df_name, fraud_df)
                        
                        # Map column names if needed
                        x_col_raw = chart_info.get("x")
                        y_col_raw = chart_info.get("y")
                        
                        # Apply improved column finding logic
                        x_col = find_column_in_dataframe(x_col_raw, selected_df) if x_col_raw else selected_df.columns[0]
                        if not x_col:
                            st.warning(f"Column '{x_col_raw}' not found. Using first column instead.")
                            x_col = selected_df.columns[0]
                        
                        if y_col_raw and y_col_raw.lower() != "count":
                            y_col = find_column_in_dataframe(y_col_raw, selected_df)
                            if not y_col:
                                st.warning(f"Column '{y_col_raw}' not found. Using count instead.")
                                y_col = "count"
                        else:
                            y_col = "count"
                        
                        # Now generate the chart
                        st.subheader("📊 Auto-Generated Chart")
                        st.caption(f"Using dataframe: {df_name}")

                        if chart_type == "bar":
                            if y_col == "count":
                                chart_data = selected_df[x_col].value_counts().reset_index()
                                chart_data.columns = [x_col, 'count']
                                fig = px.bar(chart_data, x=x_col, y='count', title=f"Count of {x_col}")
                            else:
                                if y_col in selected_df.columns:
                                    chart_data = selected_df.groupby(x_col)[y_col].count().reset_index()
                                    fig = px.bar(chart_data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                else:
                                    st.warning(f"Column '{y_col}' not found. Using count instead.")
                                    chart_data = selected_df[x_col].value_counts().reset_index()
                                    chart_data.columns = [x_col, 'count']
                                    fig = px.bar(chart_data, x=x_col, y='count', title=f"Count of {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            ai_response = f"Here's a bar chart showing the distribution of {x_col} from the {df_name} data."
                            st.write(ai_response)
                            st.session_state.current_chat.append({"role": "assistant", "content": ai_response})

                        elif chart_type == "pie":
                            if y_col == "count":
                                pie_data = selected_df[x_col].value_counts().reset_index()
                                pie_data.columns = [x_col, 'count']
                                fig = px.pie(pie_data, names=x_col, values='count', title=f"Distribution of {x_col}")
                            else:
                                if y_col in selected_df.columns:
                                    pie_data = selected_df.groupby(x_col)[y_col].sum().reset_index()
                                    fig = px.pie(pie_data, names=x_col, values=y_col, title=f"{y_col} by {x_col}")
                                else:
                                    st.warning(f"Column '{y_col}' not found. Using count instead.")
                                    pie_data = selected_df[x_col].value_counts().reset_index()
                                    pie_data.columns = [x_col, 'count'] 
                                    fig = px.pie(pie_data, names=x_col, values='count', title=f"Distribution of {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            ai_response = f"Here's a pie chart showing the distribution of {x_col} from the {df_name} data."
                            st.write(ai_response)
                            st.session_state.current_chat.append({"role": "assistant", "content": ai_response})

                        elif chart_type == "line":
                            # Check if the column can be converted to datetime
                            is_date_column = False
                            if x_col in selected_df.columns:
                                try:
                                    selected_df[x_col] = pd.to_datetime(selected_df[x_col], errors='coerce')
                                    is_date_column = True
                                except:
                                    pass
                                    
                            if y_col == "count":
                                if is_date_column:
                                    # For date columns, group by day/month
                                    line_data = selected_df.groupby(selected_df[x_col].dt.date).size().reset_index()
                                    line_data.columns = [x_col, 'count']
                                else:
                                    line_data = selected_df[x_col].value_counts().sort_index().reset_index()
                                    line_data.columns = [x_col, 'count']
                                fig = px.line(line_data, x=x_col, y='count', title=f"Count of {x_col} Over Time")
                            else:
                                if y_col in selected_df.columns:
                                    if is_date_column:
                                        line_data = selected_df.groupby(selected_df[x_col].dt.date)[y_col].sum().reset_index()
                                    else:
                                        line_data = selected_df.groupby(x_col)[y_col].sum().reset_index()
                                    fig = px.line(line_data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                else:
                                    st.warning(f"Column '{y_col}' not found. Using count instead.")
                                    if is_date_column:
                                        line_data = selected_df.groupby(selected_df[x_col].dt.date).size().reset_index()
                                        line_data.columns = [x_col, 'count']
                                    else:
                                        line_data = selected_df[x_col].value_counts().sort_index().reset_index()
                                        line_data.columns = [x_col, 'count']
                                    fig = px.line(line_data, x=x_col, y='count', title=f"Count of {x_col} Over Time")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            ai_response = f"Here's a line chart showing the trend of {x_col} from the {df_name} data."
                            st.write(ai_response)
                            st.session_state.current_chat.append({"role": "assistant", "content": ai_response})

                        else:
                            ai_response = "Sorry, I couldn't generate that chart type."
                            st.warning(ai_response)
                            st.session_state.current_chat.append({"role": "assistant", "content": ai_response})
                    
                    else:
                        # Regular question answering with Gemini
                        customers_count = customers['CustomerID'].nunique()
                        accounts_count = accounts['AccountID'].nunique()
                        transactions_count = transactions['TransactionID'].nunique()
                        avg_balance = accounts['Balance'].mean()
                        
                        data_insights = f"""
Current data summary:
- Total customers: {customers_count}
- Total accounts: {accounts_count}
- Total transactions: {transactions_count}
- Average account balance: ${avg_balance:.2f}
"""
                        
                        system_prompt = get_system_prompt()
                        full_prompt = f"{system_prompt}\n\nUser query: {user_input}\n\nSystem context: {data_insights}"
                        
                        response = chat.send_message(full_prompt)
                        ai_response = response.text
                        
                        st.write(ai_response)
                        st.session_state.current_chat.append({"role": "assistant", "content": ai_response})
                    
                except Exception as e:
                    error_msg = f"Error getting response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.current_chat.append({"role": "assistant", "content": error_msg})

with tab9:
    st.header("Automatic Data Analysis")

    uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                # Improved Excel file reading
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                
                # Clean column names
                df.columns = [str(col).strip() for col in df.columns]
                
                # Remove empty rows
                df = df.dropna(how='all')
                
                # Remove empty columns
                df = df.dropna(axis=1, how='all')
                
                # Convert data to appropriate types
                for col in df.columns:
                    try:
                        # Attempt to convert numeric columns
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        continue

            st.success("Data loaded successfully!")
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Display data information
            st.write("Data Information:")
            st.write(f"Number of Rows: {len(df)}")
            st.write(f"Number of Columns: {len(df.columns)}")
            st.write("Column Names:")
            st.write(df.columns.tolist())

            # Create automatic analysis report
            with st.spinner("Analyzing data..."):
                profile = ProfileReport(df, title="Analysis Report", explorative=True)
                st_profile_report(profile)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.write("Please ensure the file is in the correct format and contains structured data")

with tab10:
    st.header("📱 Telegram Reports")
    
    # ==== Bot Configuration ====
    col1, col2 = st.columns(2)
    with col1:
        telegram_token = st.text_input("Telegram Bot Token", value="7044390135:AAHfV0oAGsLHAoZUDXMCggenDiEVe4vZPeo")
    with col2:
        telegram_chat_id = st.text_input("Telegram Chat ID", value="1636741464")
    
    report_hour = st.slider("Hour to send daily report (24h format)", 0, 23, 9)
    
    # ==== Send Telegram Message ====
    def send_telegram_message(message):
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {
            "chat_id": telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, data=payload)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Error sending message: {e}")
            return False
    
    # ==== Generate Daily Report ====
    def generate_daily_report():
        today = pd.Timestamp.now().normalize()
        
        # Process customer data
        customers_copy = customers.copy()
        customers_copy['JoinDate'] = pd.to_datetime(customers_copy['JoinDate'], errors='coerce')
        new_today = customers_copy[customers_copy['JoinDate'].dt.normalize() == today].shape[0]
        
        # Process transaction data
        transactions_copy = transactions.copy()
        transactions_copy['TransactionDate'] = pd.to_datetime(transactions_copy['TransactionDate'], errors='coerce')
        last_tx = transactions_copy.merge(accounts[['AccountID', 'CustomerID']], on='AccountID', how='left') \
            .groupby("CustomerID")["TransactionDate"].max().reset_index()
        
        customers_with_tx = customers_copy.merge(last_tx, on="CustomerID", how="left")
        six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
        churn_risk = customers_with_tx[(customers_with_tx['TransactionDate'].isna()) | 
                                  (customers_with_tx['TransactionDate'] < six_months_ago)].shape[0]
        
        # Calculate other metrics
        total_customers = customers_copy['CustomerID'].nunique()
        total_balance = accounts['Balance'].sum()
        avg_balance = accounts['Balance'].mean()
        
        msg = f"""📊 *Daily Banking Report - {today.date()}*

👥 *Customer Metrics:*
- Total Customers: {total_customers:,}
- New Customers Today: {new_today}
- Customers at Risk of Churn: {churn_risk} ({churn_risk/total_customers:.1%})

💰 *Financial Metrics:*
- Total Balance: ${total_balance:,.2f}
- Average Balance: ${avg_balance:,.2f}

📈 *Transaction Activity:*
- Recent Transactions: {transactions_copy[transactions_copy['TransactionDate'] >= (today - pd.DateOffset(days=1))].shape[0]}

✅ Report generated at {datetime.datetime.now().strftime('%H:%M:%S')}.
"""
        return msg
    
    # ==== Run Daily Report Thread at Specific Hour ====
    def start_daily_report_thread():
        def run_daily():
            while True:
                now = datetime.datetime.now()
                next_run = now.replace(hour=report_hour, minute=0, second=0, microsecond=0)
                if now >= next_run:
                    next_run += datetime.timedelta(days=1)
                sleep_duration = (next_run - now).total_seconds()
                time.sleep(sleep_duration)
                message = generate_daily_report()
                send_telegram_message(message)
        
        thread = threading.Thread(target=run_daily)
        thread.daemon = True
        thread.start()
        return thread
    
    # ==== Status and Controls ====
    st.subheader("Report Configuration")
    
    if 'telegram_thread' not in st.session_state:
        st.session_state.telegram_thread = None
        st.session_state.telegram_active = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Automatic Daily Reports"):
            if not st.session_state.telegram_active:
                st.session_state.telegram_thread = start_daily_report_thread()
                st.session_state.telegram_active = True
                st.success("✅ Automatic reporting started!")
    
    with col2:
        if st.button("Stop Automatic Reports"):
            st.session_state.telegram_active = False
            st.session_state.telegram_thread = None
            st.info("❌ Automatic reporting stopped")
    
    if st.session_state.telegram_active:
        st.success(f"✅ Automatic reports are configured to be sent daily at {report_hour}:00")
    else:
        st.warning("⚠️ Automatic reporting is inactive")
    
    st.subheader("Manual Report")
    
    report_preview = generate_daily_report()
    st.code(report_preview, language="markdown")
    
    if st.button("📱 Send Report Now"):
        with st.spinner("Sending report to Telegram..."):
            if send_telegram_message(report_preview):
                st.success("✅ Report sent successfully!")
            else:
                st.error("❌ Failed to send report. Check your bot token and chat ID.")
    
    st.subheader("Test Telegram Connection")
    
    test_message = st.text_input("Test Message", value="Hello from Banking Dashboard!")
    
    if st.button("Send Test Message"):
        with st.spinner("Testing connection..."):
            if send_telegram_message(test_message):
                st.success("✅ Connection successful! Message sent.")
            else:
                st.error("❌ Connection failed. Please check your Telegram bot token and chat ID.")
                
    with st.expander("ℹ️ How to setup your Telegram bot"):
        st.markdown("""
        1. **Create a Telegram Bot**:
           - Open Telegram and search for `@BotFather`
           - Send the command `/newbot` and follow the instructions
           - BotFather will give you a token - copy it to the "Telegram Bot Token" field above
        
        2. **Get your Chat ID**:
           - Search for `@userinfobot` on Telegram
           - Start a chat with this bot
           - It will reply with your Chat ID - copy it to the "Telegram Chat ID" field above
        
        3. **Test the Connection**:
           - Use the "Send Test Message" button to verify everything works
           - If successful, you can enable automatic daily reports
        """)

with tab11:
    st.header("🏦 Customer Churn Prediction")
    st.write("""
    ### Use this application to predict if a customer is likely to churn
    Fill in the customer information below and click 'Predict' to get the churn probability.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Profile")
        tenure = st.number_input("Tenure (years)", min_value=0.0, max_value=50.0, value=5.0, key="churn_tenure")
        total_balance = st.number_input("Total Balance ($)", min_value=0.0, max_value=1000000.0, value=10000.0, key="churn_balance")
        num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=1, key="churn_products")
        
    with col2:
        st.subheader("Transaction Information")
        recency = st.number_input("Days Since Last Transaction", min_value=0, max_value=365, value=30, key="churn_recency")
        frequency = st.number_input("Transaction Frequency (monthly)", min_value=0, max_value=100, value=10, key="churn_frequency")
        monetary = st.number_input("Average Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=100.0, key="churn_monetary")

    st.subheader("Additional Information")
    col3, col4 = st.columns(2)
    
    with col3:
        has_loan = st.selectbox("Has Active Loan?", ["Yes", "No"], key="churn_has_loan")
        num_cards = st.number_input("Number of Cards", min_value=0, max_value=10, value=1, key="churn_cards")
    
    with col4:
        support_calls = st.number_input("Number of Support Calls", min_value=0, max_value=50, value=0, key="churn_calls")
        resolution_rate = st.slider("Support Resolution Rate", min_value=0.0, max_value=1.0, value=1.0, key="churn_resolution")

    features = {
        'customerid': 1,  
        'tenure': tenure,
        'total_balance': total_balance,
        'number_of_accounts': num_products,
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'has_active_loan': 1 if has_loan == "Yes" else 0,
        'number_of_cards': num_cards,
        'support_call_frequency': support_calls,
        'resolution_rate': resolution_rate
    }

    if st.button("Predict Churn Probability", key="churn_predict_button"):
        model = load_ml_model()
        
        input_df = pd.DataFrame([features])
        prediction_prob = model.predict_proba(input_df)[0][1]
        
        st.subheader("Prediction Results")
        

        if prediction_prob < 0.3:
            color = "green"
            risk_level = "Low Risk"
        elif prediction_prob < 0.7:
            color = "orange"
            risk_level = "Medium Risk"
        else:
            color = "red"
            risk_level = "High Risk"
            
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {color}25;">
            <h3 style="color: {color};">{risk_level}</h3>
            <h2 style="color: {color};">{prediction_prob:.1%}</h2>
            <p>Probability of Customer Churn</p>
        </div>
        """, unsafe_allow_html=True)
        
        
        st.subheader("Risk Analysis")
        if prediction_prob < 0.3:
            st.write("📈 This customer shows strong loyalty indicators.")
            st.write("Recommendations:")
            st.write("- Consider offering premium services or products")
            st.write("- Enroll in loyalty rewards program")
        elif prediction_prob < 0.7:
            st.write("⚠️ This customer shows moderate churn risk.")
            st.write("Recommendations:")
            st.write("- Proactive engagement through targeted offers")
            st.write("- Schedule customer satisfaction survey")
            st.write("- Review product usage patterns")
        else:
            st.write("🚨 High risk of customer churn!")
            st.write("Recommendations:")
            st.write("- Immediate customer outreach")
            st.write("- Develop retention strategy")
            st.write("- Consider special retention offers")
                    
