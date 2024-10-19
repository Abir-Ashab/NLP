import pickle
from flask import Flask, request, jsonify, render_template
import google.generativeai as gemini
import markdown
from googletrans import Translator

gemini.configure(api_key='AIzaSyCnJXSTdS4w5Il9r7URapwWBAmo2UkruV4')

model = pickle.load(open('./Models/logistic_regression_model.pkl', 'rb'))
vectorizer = pickle.load(open('./Models/vectorizer.pkl', 'rb'))

translator = Translator()
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    languages = request.form.getlist('languages')

    comment_vectorized = vectorizer.transform([comment])

    prediction = model.predict(comment_vectorized)[0]
    
    if prediction == 1:
        sentiment = "Positive"
    elif prediction == 0:
        sentiment = "Neutral"
    else:
        sentiment = "Negative"
    
    try:
        generative_model = gemini.GenerativeModel('gemini-1.5-flash')
        response = generative_model.generate_content(f"Explain why or why not {comment} is a {sentiment} ")
        explanation = markdown.markdown(response.text) 
    except Exception as e:
        explanation = f"Error generating explanation: {str(e)}"
    return jsonify({'sentiment': sentiment, 'explanation': explanation})

@app.route('/translate', methods=['POST'])
def translate():
    comment = request.form['comment']
    target_language = request.form['language']  

    translation = translator.translate(comment, dest=target_language)
    
    return jsonify({'translation': translation.text})
    
if __name__ == "__main__":
    app.run(debug=True)
