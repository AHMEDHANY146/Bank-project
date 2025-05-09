from pydoc import describe
from turtle import width
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import google.generativeai as genai
# To run Program the path streamlit run C:\Users\Ahmed\source\repos\DashBoared\DashBoared\DashBoared.py
# Import Data from CSV or Excel File

try:
    DF = pd.read_csv(r"D:\datacamp\heart_attack_dataset.csv")
   # DF = pd.read_csv(r"F:\DataSetsSamples\HeartAttackData\heart_attack_dataset.csv")
except Exception as ex: 
    print(ex)
# First Data Understanding and Exploration
# Index(['Gender', 'Age', 'Blood Pressure (mmHg)', 'Cholesterol (mg/dL)','Has Diabetes', 'Smoking Status', 'Chest Pain Type', 'Treatment'], dtype='object')
print(DF.columns)
print("***********************************************************************************************************")
print(DF.info())
print("***********************************************************************************************************")
print(DF.describe())
print("***********************************************************************************************************")
print(DF['Gender'].describe()) 
print("***********************************************************************************************************")
print(DF['Has Diabetes'].describe())
print("***********************************************************************************************************")
print(DF['Smoking Status'].describe())
print("***********************************************************************************************************")
print(DF['Chest Pain Type'].describe())
print("***********************************************************************************************************")
print(DF['Treatment'].describe())
print("***********************************************************************************************************")
print(DF.head(5))
print("***********************************************************************************************************")
print(DF.tail(5))
# ChatGpt help :
DF.columns = [col.replace(' ', '_') for col in DF.columns]
# streamlit run C:\Users\Ahmed\source\repos\DashBoared\DashBoared\DashBoared.py
# Set page for DashBoard
st.set_page_config(
    page_title="HeartAttack_Analysis_Dashboard",
    page_icon="🌡",
    layout="wide"
)
st.title("Heart Attack Dashboard Analysis")
st.caption("Directed By Ahmed Ehab Mostafa , Std_ID = 230923")
#st.dataframe(DF)
# ---SideBar---
st.sidebar.header("Filter")
# Gender
gender = st.sidebar.multiselect(
    "Select Gender:",
    options=DF['Gender'].unique(),
    default=DF['Gender'].unique()
    )
# Smoking Status
smoking_status = st.sidebar.multiselect(
    "Select Smoking Status:",
    options=DF['Smoking_Status'].unique(),
    default=DF['Smoking_Status'].unique()
    )
# Chest Pain Type
chest_pain_type = st.sidebar.multiselect(
    "Select Chest Pain Type:",
    options=DF['Chest_Pain_Type'].unique(),
    default=DF['Chest_Pain_Type'].unique()
    )
# Treatment
treatment_ = st.sidebar.multiselect(
    "Select Treatment:",
    options=DF['Treatment'].unique(),
    default=DF['Treatment'].unique()
    )
# Has Diabetes
has_diabetes = st.sidebar.multiselect(
    "Select Has Diabetes:",
    options=DF['Has_Diabetes'].unique(),
    default=DF['Has_Diabetes'].unique()
    ) 
# Promamming the Sidebar (Back-End) 
# Error Happened Remember Query method in pandas can't treat with spaces you should replace it with replace func
DF_selection = DF.query(
    "Gender == @gender & Smoking_Status == @smoking_status & Chest_Pain_Type == @chest_pain_type & Treatment == @treatment_ & Has_Diabetes == @has_diabetes"
    ) 
with st.container():
    st.dataframe(DF_selection, use_container_width=True)
# Index(['Gender', 'Age', 'Blood Pressure (mmHg)', 'Cholesterol_(mg/dL)',
#       'Has Diabetes', 'Smoking Status', 'Chest Pain Type', 'Treatment'],
col1, col2 ,col3 = st.columns(3)
# Plot for Blood Pressure (mmHg)
with col1:
    try:
        fig, ax = plt.subplots()

        DF_selection.boxplot(column='Blood_Pressure_(mmHg)', ax=ax)
        plt.title('Boxplot of Blood Pressure (mmHg)')
        plt.suptitle('')  # Suppress the default Matplotlib title
        plt.xlabel('Blood Pressure (mmHg)')
        plt.ylabel('Values')

        # Display the plot in Streamlit
        st.pyplot(fig)
    except Exception as ex:
        print(ex)

# Plot for Cholesterol (mg/dL)
with col2:
    try:
        fig2, ax2 = plt.subplots()
        DF_selection.boxplot(column='Cholesterol_(mg/dL)', ax=ax2)
        plt.title('Boxplot of Cholesterol (mg/dL)')
        plt.suptitle('')  # Suppress the default Matplotlib title
        plt.xlabel('Cholesterol (mg/dL)')
        plt.ylabel('Values')

        # Display the plot in Streamlit
        st.pyplot(fig2)
    except Exception as ex:
        print(ex)
with col3:
    try:
        fig3, ax3 = plt.subplots()
        DF_selection.boxplot(column='Age', ax=ax3)
        plt.title('Boxplot of Age')
        plt.suptitle('')  # Suppress the default Matplotlib title
        plt.xlabel('Age')
        plt.ylabel('Values')

        # Display the plot in Streamlit
        st.pyplot(fig3)
    except Exception as ex:
        print(ex)
st.title("Univariate Analysis: ")
col1, col2 = st.columns(2)
with col1:
    freq = DF_selection['Has_Diabetes'].value_counts().reset_index()
    freq.columns = ['Has_Diabetes', 'Count']

    fig4 = px.bar(
        freq,
        x='Has_Diabetes',
        y='Count',
        title='Has_Diabetes',
        labels={'Has_Diabetes': 'Has_Diabetes', 'Count': 'Has_Diabetes'},
        color='Has_Diabetes', 
        color_discrete_sequence=['#8A2BE2','#DA70D6', '#D8BFD8','#4B0082']  
    )

    # Display the Plotly bar chart in Streamlit
    st.plotly_chart(fig4)
with col2:
    # Create a pie chart using Plotly
    fig5 = px.pie(
        freq,
        names='Has_Diabetes',
        values='Count',
        title='Has_Diabetes',
        color='Has_Diabetes',
        color_discrete_sequence=['#8A2BE2','#DA70D6'],  # Customize colors
        hole=0.3  # Optional: makes it a donut chart if you like
    )
    # Display the Plotly pie chart in Streamlit
    st.plotly_chart(fig5)
col1, col2 = st.columns(2)
with col1:
    freq = DF_selection['Smoking_Status'].value_counts().reset_index()
    freq.columns = ['Smoking_Status', 'Count']

    fig4 = px.bar(
        freq,
        x='Smoking_Status',
        y='Count',
        title='Smoking_Status',
        labels={'Smoking_Status': 'Smoking_Status', 'Count': 'Smoking_Status'},
        color='Smoking_Status', 
        color_discrete_sequence=['#8A2BE2','#DA70D6', '#D8BFD8','#4B0082']  
    )

    # Display the Plotly bar chart in Streamlit
    st.plotly_chart(fig4)
with col2:
    # Create a pie chart using Plotly
    fig5 = px.pie(
        freq,
        names='Smoking_Status',
        values='Count',
        title='Smoking_Status',
        color='Smoking_Status',
        color_discrete_sequence=['#8A2BE2','#DA70D6', '#D8BFD8','#4B0082'],  # Customize colors
        hole=0.3  # Optional: makes it a donut chart if you like
    )
    # Display the Plotly pie chart in Streamlit
    st.plotly_chart(fig5)
col1, col2 = st.columns(2)
with col1:
    freq = DF_selection['Chest_Pain_Type'].value_counts().reset_index()
    freq.columns = ['Chest_Pain_Type', 'Count']

    fig4 = px.bar(
        freq,
        x='Chest_Pain_Type',
        y='Count',
        title='Chest_Pain_Type',
        labels={'Chest_Pain_Type': 'Chest_Pain_Type', 'Count': 'Chest_Pain_Type'},
        color='Chest_Pain_Type', 
        color_discrete_sequence=['#8A2BE2','#DA70D6', '#D8BFD8','#4B0082']  
    )

    # Display the Plotly bar chart in Streamlit
    st.plotly_chart(fig4)
with col2:
    # Create a pie chart using Plotly
    fig5 = px.pie(
        freq,
        names='Chest_Pain_Type',
        values='Count',
        title='Chest_Pain_Type',
        color='Chest_Pain_Type',
        color_discrete_sequence=['#8A2BE2','#DA70D6', '#D8BFD8','#4B0082'],  # Customize colors
        hole=0.3  # Optional: makes it a donut chart if you like
    )
    # Display the Plotly pie chart in Streamlit
    st.plotly_chart(fig5)
col1, col2 = st.columns(2)
#Bar
with col1:
    freq = DF_selection['Treatment'].value_counts().reset_index()
    freq.columns = ['Treatment', 'Count']

    fig6 = px.bar(
        freq,
        x='Treatment',
        y='Count',
        title='Treatment',
        labels={'Treatment': 'Treatment', 'Count': 'Frequency'},
        color='Treatment', 
        color_discrete_sequence=['#8A2BE2','#DA70D6', '#D8BFD8','#4B0082']  
    )
    # Display the Plotly bar chart in Streamlit
    st.plotly_chart(fig6)
# Create a pie chart using Plotly
with col2:
    fig7 = px.pie(
        freq,
        names='Treatment',
        values='Count',
        title='Treatments',
        color='Treatment',
        color_discrete_sequence=['#8A2BE2','#DA70D6', '#D8BFD8','#4B0082'],  # Customize colors
        hole=0.3  # Optional: makes it a donut chart if you like
    )
    # Display the Plotly pie chart in Streamlit
    st.plotly_chart(fig7)
col1, col2, col3 = st.columns(3)

for i, column in enumerate(['Age', 'Blood_Pressure_(mmHg)', 'Cholesterol_(mg/dL)']):
    if column == 'Age':
        with col1:
            fig = px.histogram(
                DF,
                x=column,
                title=f'{column}',
                labels={column: column},
                color_discrete_sequence=[['#8A2BE2', '#DA70D6', '#D8BFD8'][i]]  # Different color for each histogram
            )
            #fig.update_layout(width=600, height=500)
            st.plotly_chart(fig)
    elif column == 'Blood_Pressure_(mmHg)':
        with col2:
            fig = px.histogram(
                DF,
                x=column,
                title=f'{column}',
                labels={column: column},
                color_discrete_sequence=[['#8A2BE2', '#DA70D6', '#D8BFD8'][i]]  # Different color for each histogram
            )
            #fig.update_layout(width=600, height=500)
            st.plotly_chart(fig)
    else :
        with col3:
            fig = px.histogram(
                DF,
                x=column,
                title=f'{column}',
                labels={column: column},
                color_discrete_sequence=[['#8A2BE2', '#DA70D6', '#D8BFD8'][i]]  # Different color for each histogram
            )
            #fig.update_layout(width=600, height=500)
            st.plotly_chart(fig)
st.title("Correlation Analysis:")

# Select numerical data
numerical_data = DF_selection.select_dtypes(include=['number'])
# Calculate the correlation matrix
corr_matrix = numerical_data.corr()
# Create a heatmap
fig = plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.5f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
# Display the plot in Streamlit
st.pyplot(fig)  # Corrected method name

st.title("Bivariate Analysis:")
col1, col2, col3 = st.columns(3)
# Gender aganist Diabetes
with col1:
    freq_gender_diabetes = DF.groupby(['Gender', 'Has_Diabetes']).size().reset_index(name='Count')
    fig = px.bar(
        freq_gender_diabetes,
        x='Gender',
        y='Count',
        color='Has_Diabetes',
        title='Gender vs Diabetes Status',
        labels={'Gender': 'Gender', 'Count': 'Frequency', 'Has_Diabetes': 'Diabetes Status'},
        color_discrete_sequence=['#8A2BE2', '#DA70D6'] 
    )
    st.plotly_chart(fig)
# Gender aganist Smkoing Status
with col2:
    freq_gender_diabetes = DF.groupby(['Gender', 'Smoking_Status']).size().reset_index(name='Count')
    fig = px.bar(
        freq_gender_diabetes,
        x='Gender',
        y='Count',
        color='Smoking_Status',
        title='Gender vs Smoking_Status',
        labels={'Gender': 'Gender', 'Count': 'Frequency', 'Smoking_Status': 'Smoking_Status'},
        color_discrete_sequence=['#8A2BE2', '#DA70D6','#D8BFD8'] 
    )
    st.plotly_chart(fig)
with col3:
    freq_gender_chest_pain = DF.groupby(['Gender', 'Chest_Pain_Type']).size().reset_index(name='Count')
    fig = px.bar(
        freq_gender_chest_pain,
        x='Chest_Pain_Type',
        y='Count',
        color='Gender',
        title='Gender vs Chest Pain Type',
        labels={'Chest_Pain_Type': 'Chest_Pain_Type', 'Count': 'Frequency', 'Gender': 'Gender'},
        color_discrete_sequence=['#8A2BE2', '#D8BFD8']  
    )
    st.plotly_chart(fig)
freq_chest_pain_treatment = DF.groupby(['Gender', 'Treatment']).size().reset_index(name='Count')

fig = px.bar(
    freq_chest_pain_treatment,
    x='Gender',
    y='Count',
    color='Treatment',
    title='Gender vs Treatment',
    labels={'Gender': 'Gender', 'Count': 'Frequency', 'Treatment': 'Treatment'},
    color_discrete_sequence=['#8A2BE2', '#6A0D91', '#DA70D6', '#D8BFD8']  
)
st.plotly_chart(fig)

freq_chest_pain_treatment = DF.groupby(['Chest_Pain_Type', 'Treatment']).size().reset_index(name='Count')
fig = px.bar(
    freq_chest_pain_treatment,
    x='Chest_Pain_Type',
    y='Count',
    color='Treatment',
    title='Chest_Pain_Type vs Treatment',
    labels={'Chest_Pain_Type': 'Chest_Pain_Type', 'Count': 'Frequency', 'Treatment': 'Treatment'},
    color_discrete_sequence=['#8A2BE2', '#6A0D91', '#DA70D6', '#D8BFD8']  # Shades of violet
)
st.plotly_chart(fig)
# Relation between Blood Preasure and Cholesterol
fig = px.scatter(
    DF,
    x='Blood_Pressure_(mmHg)',
    y='Cholesterol_(mg/dL)',
    color='Blood_Pressure_(mmHg)',  
    color_continuous_scale=['#8A2BE2', '#DA70D6'],  
    title='Blood Pressure vs Cholesterol',
    labels={'Blood_Pressure_(mmHg)': 'Blood_Pressure_(mmHg)', 'Cholesterol_(mg/dL)': 'Cholesterol_(mg/dL)', 'Blood_Pressure_(mmHg)': 'Blood_Pressure_(mmHg)'}
)
st.plotly_chart(fig)

fig = px.scatter(
    DF,
    x='Age',
    y='Cholesterol_(mg/dL)',
    color='Age',  
    color_continuous_scale='Viridis',  
    title='Age vs Cholesterol',
    labels={'Age': 'Age', 'Cholesterol_(mg/dL)': 'Cholesterol_(mg/dL)', 'Age': 'Age'}
    )
st.plotly_chart(fig)

fig = px.scatter(
    DF,
    x='Blood_Pressure_(mmHg)',
    y='Cholesterol_(mg/dL)',
    color='Blood_Pressure_(mmHg)',  
    color_continuous_scale=['#FFD700', '#90EE90'],  
    title='Blood Pressure vs Cholesterol',
    labels={'Blood_Pressure_(mmHg)': 'Blood_Pressure_(mmHg)', 'Cholesterol_(mg/dL)': 'Cholesterol_(mg/dL)', 'Blood_Pressure_(mmHg)': 'Blood_Pressure_(mmHg)'}
    )
st.plotly_chart(fig)

st.title("Multivariate Analysis:")
fig = px.scatter_3d(
    DF,
    x='Age',
    y='Blood_Pressure_(mmHg)',
    z='Cholesterol_(mg/dL)',
    color='Age',  
    color_continuous_scale='Viridis',  
    title='Scatter Plot of Age, Blood Pressure, and Cholesterol',
    labels={'Age': 'Age', 'Blood_Pressure_(mmHg)': 'Blood_Pressure_(mmHg)', 'Cholesterol_(mg/dL)': 'Cholesterol_(mg/dL)'}
    )
fig.update_layout(width=1000, height=700)
st.plotly_chart(fig)

fig = px.scatter_3d(
    DF,
    x='Has_Diabetes',
    y='Blood_Pressure_(mmHg)',
    z='Cholesterol_(mg/dL)',
    color='Age',  
    color_continuous_scale='Viridis',  
    title='Scatter Plot of Has_Diabetes, Blood Pressure, and Cholesterol',
    labels={'Has_Diabetes': 'Has_Diabetes', 'Blood_Pressure_(mmHg)': 'Blood_Pressure_(mmHg)', 'Cholesterol_(mg/dL)': 'Cholesterol_(mg/dL)'}
    )
fig.update_layout(width=1000, height=700)
st.plotly_chart(fig)

freq = DF.groupby(['Has_Diabetes', 'Smoking_Status', 'Chest_Pain_Type']).size().reset_index(name='Count')
fig = px.bar(
    freq,
    x='Has_Diabetes',
    y='Count',
    color='Chest_Pain_Type',
    facet_col='Smoking_Status',
    title='Comparison of Has Diabetes, Smoking_Status, and Chest_Pain_Type',
    labels={'Has_Diabetes': 'Has_Diabetes', 'Count': 'Frequency', 'Chest_Pain_Type': 'Chest_Pain_Type'},
    color_discrete_sequence=['#8A2BE2', '#6A0D91', '#DA70D6', '#D8BFD8']  # Custom color sequence
)
fig.update_layout(width=1000, height=600)
st.plotly_chart(fig)
freq = DF.groupby(['Gender', 'Chest_Pain_Type', 'Treatment']).size().reset_index(name='Count')

fig = px.bar(
    freq,
    x='Chest_Pain_Type',
    y='Count',
    color='Treatment',
    facet_col='Gender',
    title='Comparison of Gender, Chest Pain Type, and Treatment',
    labels={'Chest_Pain_Type': 'Chest_Pain_Type', 'Count': 'Frequency', 'Treatment': 'Treatment'},
    width = 1200,
    color_discrete_sequence=['#8A2BE2', '#6A0D91', '#DA70D6', '#D8BFD8']  # Custom color sequence
)
fig.update_layout(width=1000, height=600)
# **************************************************************************************************************
# Open the image file
image = Image.open("D:\datacamp\ha.jpg")  # Or use a path like 'assets/image.jpg'
# Display the image
st.image(image, use_column_width=True)
st.title("Conclusion")
st.write("Age Distribution -> The minimum age is 30 years, while the maximum is 89 years. The average age of individuals in the")
st.write("dataset is approximately 60.34 years. This indicates a wide range of ages with a slightly higher concentration in older adults.")
st.write("Blood Pressure -> The minimum recorded blood pressure is 90 mmHg, and the maximum is 199 mmHg. The average blood")
st.write("pressure is 145.44 mmHg, suggesting a generally high average blood pressure level among the individuals in the dataset.")
st.write("Cholesterol Levels -> Cholesterol levels range from a minimum of 150 mg/dL to a maximum of 299 mg/dL. The average")
st.write("cholesterol level is 223.79 mg/dL, indicating that most individuals have elevated cholesterol levels.")
st.subheader("Correlation:")
st.write("Age and Blood Pressure: Almost no relationship, indicating age has a minimal impact on blood pressure.")
st.write("Blood Pressure and Cholesterol: Very weak positive correlation, suggesting a minimal connection between these two variables.")
st.write("Age and Cholesterol: Negligible correlation, indicating that age does not significantly influence cholesterol levels.")
#******************************************************88
# تكوين API
GOOGLE_API_KEY = "AIzaSyCfr_AYlPCQPYToTY2NUDM-4nEFbYNdhVY"
genai.configure(api_key=GOOGLE_API_KEY)

# إنشاء نموذج المحادثة
model = genai.GenerativeModel('gemini-2.0-flash')
chat = model.start_chat(history=[])

# إضافة قسم المحادثة في لوحة التحكم
st.title("🤖 Data Analysis Assistant")

# إنشاء مربع نص للمدخلات
user_question = st.text_input("Ask your question about the data:", key="user_input")

# تهيئة سجل المحادثات في session state إذا لم يكن موجوداً
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# دالة للكشف عن لغة النص
def detect_language(text):
    # تحقق بسيط للغة العربية
    arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
    text_chars = set(text)
    return 'ar' if any(char in arabic_chars for char in text_chars) else 'en'

# دالة لتحضير سياق البيانات
def prepare_data_context():
    # تحليل إحصائي أساسي
    numerical_stats = DF.describe()
    categorical_stats = {col: DF[col].value_counts().to_dict() 
                        for col in DF.select_dtypes(include=['object']).columns}
    
    # تحليل التوزيع
    distributions = {
        'Age': {
            'skew': DF['Age'].skew(),
            'range': f"{DF['Age'].min()} - {DF['Age'].max()}",
            'mean': DF['Age'].mean(),
            'median': DF['Age'].median(),
            'age_groups': {
                'young': len(DF[DF['Age'] < 40]),
                'middle': len(DF[(DF['Age'] >= 40) & (DF['Age'] < 60)]),
                'elderly': len(DF[DF['Age'] >= 60])
            }
        },
        'Blood_Pressure': {
            'skew': DF['Blood_Pressure_(mmHg)'].skew(),
            'range': f"{DF['Blood_Pressure_(mmHg)'].min()} - {DF['Blood_Pressure_(mmHg)'].max()}",
            'mean': DF['Blood_Pressure_(mmHg)'].mean(),
            'median': DF['Blood_Pressure_(mmHg)'].median(),
            'categories': {
                'normal': len(DF[DF['Blood_Pressure_(mmHg)'] < 120]),
                'elevated': len(DF[(DF['Blood_Pressure_(mmHg)'] >= 120) & (DF['Blood_Pressure_(mmHg)'] < 130)]),
                'high': len(DF[(DF['Blood_Pressure_(mmHg)'] >= 130) & (DF['Blood_Pressure_(mmHg)'] < 140)]),
                'very_high': len(DF[DF['Blood_Pressure_(mmHg)'] >= 140])
            }
        },
        'Cholesterol': {
            'skew': DF['Cholesterol_(mg/dL)'].skew(),
            'range': f"{DF['Cholesterol_(mg/dL)'].min()} - {DF['Cholesterol_(mg/dL)'].max()}",
            'mean': DF['Cholesterol_(mg/dL)'].mean(),
            'median': DF['Cholesterol_(mg/dL)'].median(),
            'categories': {
                'desirable': len(DF[DF['Cholesterol_(mg/dL)'] < 200]),
                'borderline': len(DF[(DF['Cholesterol_(mg/dL)'] >= 200) & (DF['Cholesterol_(mg/dL)'] < 240)]),
                'high': len(DF[DF['Cholesterol_(mg/dL)'] >= 240])
            }
        }
    }
    
    # تحليل الارتباطات
    correlations = DF.select_dtypes(include=['number']).corr()
    significant_correlations = []
    for i in correlations.columns:
        for j in correlations.columns:
            if i < j and abs(correlations.loc[i,j]) > 0.3:
                significant_correlations.append({
                    'variables': f"{i} - {j}",
                    'correlation': correlations.loc[i,j],
                    'strength': 'strong' if abs(correlations.loc[i,j]) > 0.7 else 'moderate' if abs(correlations.loc[i,j]) > 0.5 else 'weak'
                })
    
    # تحليل العلاقات بين المتغيرات الفئوية
    categorical_relationships = {}
    for col1 in DF.select_dtypes(include=['object']).columns:
        for col2 in DF.select_dtypes(include=['object']).columns:
            if col1 < col2:
                cross_tab = pd.crosstab(DF[col1], DF[col2])
                categorical_relationships[f"{col1}_vs_{col2}"] = {
                    'counts': cross_tab.to_dict(),
                    'percentages': (cross_tab / cross_tab.sum() * 100).round(2).to_dict()
                }
    
    # تحليل المخاطر
    risk_analysis = {
        'high_risk_patients': {
            'high_bp_high_chol': len(DF[(DF['Blood_Pressure_(mmHg)'] >= 140) & (DF['Cholesterol_(mg/dL)'] >= 240)]),
            'diabetic_high_bp': len(DF[(DF['Has_Diabetes'] == 'Yes') & (DF['Blood_Pressure_(mmHg)'] >= 140)]),
            'smoker_high_bp': len(DF[(DF['Smoking_Status'] == 'Yes') & (DF['Blood_Pressure_(mmHg)'] >= 140)])
        },
        'treatment_effectiveness': {
            'by_age_group': DF.groupby(pd.cut(DF['Age'], bins=[0, 40, 60, 100]))['Treatment'].value_counts().to_dict(),
            'by_risk_level': DF.groupby([
                pd.cut(DF['Blood_Pressure_(mmHg)'], bins=[0, 120, 140, 200]),
                pd.cut(DF['Cholesterol_(mg/dL)'], bins=[0, 200, 240, 300])
            ])['Treatment'].value_counts().to_dict()
        }
    }
    
    return {
        "basic_info": {
            "rows": len(DF),
            "columns": list(DF.columns),
            "missing_values": DF.isnull().sum().to_dict(),
            "data_types": DF.dtypes.to_dict()
        },
        "statistical_summary": {
            "numerical": numerical_stats.to_dict(),
            "categorical": categorical_stats
        },
        "distributions": distributions,
        "correlations": {
            "matrix": correlations.to_dict(),
            "significant_relationships": significant_correlations
        },
        "categorical_relationships": categorical_relationships,
        "risk_analysis": risk_analysis,
        "sample_data": DF.head(5).to_dict()
    }

if user_question:
    # الكشف عن لغة السؤال
    lang = detect_language(user_question)
    
    # تحضير سياق البيانات
    data_context = prepare_data_context()
    
    # تحضير السؤال مع السياق
    prompt = f"""
    Data Context:
    {data_context}
    
    Question ({lang}): {user_question}
    Respond based on the question; if they ask you a brief question, reply briefly, and if the question requires detail, respond in detail.
    You are a specialized medical data analyst assistant. Please provide a comprehensive analysis following these guidelines:
    
    1. Medical Context and Clinical Relevance:
    - Interpret values in clinical context
    - Reference medical guidelines and thresholds
    - Explain clinical significance of findings
    - Relate findings to heart disease risk factors
    
    2. Numerical Precision and Statistical Analysis:
    - Present exact numerical values with precise decimal places
    - Include all relevant statistical measures
    - Maintain mathematical precision in all calculations
    - Show confidence intervals where applicable
    
    3. Risk Assessment and Patient Stratification:
    - Identify high-risk patient groups
    - Analyze treatment effectiveness by risk level
    - Evaluate impact of multiple risk factors
    - Provide risk stratification insights
    
    4. Comparative Analysis:
    - Compare findings with medical standards
    - Analyze treatment patterns
    - Evaluate demographic variations
    - Assess risk factor interactions
    
    5. Actionable Insights and Recommendations:
    - Suggest specific clinical interventions
    - Recommend monitoring strategies
    - Propose preventive measures
    - Outline follow-up protocols
    
    Please provide your answer in the same language as the question ({lang}).
    Structure your response as follows:
    1. Clinical Summary
    2. Statistical Findings
    3. Risk Assessment
    4. Comparative Analysis
    5. Recommendations
    
    Make your response both clinically relevant and statistically precise, combining medical expertise with data analysis.
    """
    
    # إرسال السؤال والحصول على الإجابة
    response = chat.send_message(prompt)
    
    # إضافة المحادثة إلى السجل
    st.session_state.chat_history.append({
        "question": user_question,
        "answer": response.text,
        "language": lang
    })

# عرض سجل المحادثات
st.subheader("Chat History")
for chat in st.session_state.chat_history:
    with st.expander(f"Question ({chat['language']}): {chat['question']}"):
        st.write(chat['answer'])

def get_data_summary():
    return {
        "age_stats": DF['Age'].describe().to_dict(),
        "bp_stats": DF['Blood_Pressure_(mmHg)'].describe().to_dict(),
        "cholesterol_stats": DF['Cholesterol_(mg/dL)'].describe().to_dict(),
        "gender_dist": DF['Gender'].value_counts().to_dict(),
        "diabetes_dist": DF['Has_Diabetes'].value_counts().to_dict()
    }

def get_correlation_analysis():
    return DF.select_dtypes(include=['number']).corr().to_dict()

