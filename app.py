from flask import Flask, request, jsonify
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyCfr_AYlPCQPYToTY2NUDM-4nEFbYNdhVY"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "")
        chat = model.start_chat(history=[])
        response = chat.send_message(question)
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
