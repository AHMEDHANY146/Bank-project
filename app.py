import streamlit as st
import google.generativeai as genai

# إعداد API من Gemini
GOOGLE_API_KEY = "AIzaSyCfr_AYlPCQPYToTY2NUDM-4nEFbYNdhVY"
genai.configure(api_key=GOOGLE_API_KEY)

# إعداد النموذج
model = genai.GenerativeModel('gemini-2.0-flash')
chat = model.start_chat(history=[])

# واجهة المستخدم
st.set_page_config(page_title="Chat with AI", page_icon="🤖")
st.title("🤖 مساعد الذكاء الاصطناعي")
st.write("اسأل أي سؤال عن البيانات الصحية أو النوبات القلبية!")

# مكان إدخال السؤال
question = st.text_input("اكتب سؤالك هنا:")

if question:
    response = chat.send_message(question)
    st.markdown("### 🤖 الرد:")
    st.write(response.text)
