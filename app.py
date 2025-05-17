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



st.set_page_config(layout="wide")
st.title("\U0001F4CA Banking Analytics Dashboard")


@st.cache_data
def load_data():
    customers = pd.read_csv("Banking_Analytics_Dataset_Updated2.csv")
    accounts = pd.read_csv("Banking_Analytics_Dataset.xlsx - Accounts.csv")
    cards = pd.read_csv("Banking_Analytics_Dataset.xlsx - Cards.csv")
    loans = pd.read_csv("Banking_Analytics_Dataset.xlsx - Loans.csv")
    calls = pd.read_csv("Banking_Analytics_Dataset.xlsx - SupportCalls.csv")
    transactions = pd.read_csv("Banking_Analytics_Transactions_Updated.csv")
    return customers, accounts, cards, loans, calls, transactions

customers, accounts, cards, loans, calls, transactions = load_data()

customers.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
transactions.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')



(
    tab1, tab2, tab3, tab4, tab5, tab6, tab7,tab8
) = st.tabs([
    "Customer Overview", "Accounts", "Transactions", "Loans",
    "Cards", "Support Calls", "Advanced Insights","Chat"
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

    st.title("🤖 Data Analysis Assistant")
    
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = []
    
    
    def save_chat():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if len(st.session_state.current_chat) > 0:
            chat_name = f"Chat {timestamp}"
            st.session_state.chat_history.append({
                "name": chat_name,
                "messages": st.session_state.current_chat.copy()
            })
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
        
        The dashboard provides insights on:
        - Customer Overview: Demographics, growth trends, churn risk
        - Account Analysis: Types, balances, dormant accounts
        - Transaction Analysis: Volume trends, seasonal patterns
        - Loan Portfolio: Types, interest rates, maturity analysis
        - Card Services: Types, issuance trends, active vs expired
        - Support Call Analysis: Resolution rates, common issues
        - Advanced Insights: Customer segmentation, cross-product usage
        
        Answer questions based on this banking data. You can respond in multiple languages including English and Arabic.
        """
        return context

    
    chat_container = st.container()
    with chat_container:
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
    
    
    if user_input:
        
        st.session_state.current_chat.append({"role": "user", "content": user_input})
        
        
        with st.chat_message("user"):
            st.write(user_input)
        
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:

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







