from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load tokenizer
with open('tokenizer_nbf.pkl', 'rb') as file:  # Change to 'rb' for read-binary mode
    tokenizer = pickle.load(file)

# Load the model from .pkl file
with open('nb_tfidf_model.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_text(text, tokenizer):
    # Use TfidfVectorizer to transform the text
    transformed_text = tokenizer.transform([text])
    return transformed_text

def detect_hoax(news_title, model, tokenizer):
    transformed_text = preprocess_text(news_title, tokenizer)
    prediction = (model.predict(transformed_text) > 0.5).astype("int32")
    if prediction[0] == 1:
        return "Hoax"
    else:
        return "Not Hoax"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_title = data.get('news_title')
    result = detect_hoax(news_title, model, tokenizer)
    return jsonify({"news_title": news_title, "result": result})

if __name__ == "__main__":
    app.run(debug=True)
