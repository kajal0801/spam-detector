from flask import Flask, render_template, request
import pickle
import os

# Load trained model
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Load trained model
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


# Load saved vectorizer
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get message from form
    message = request.form['message']
    
    # Transform message using the vectorizer
    data = vectorizer.transform([message])
    
    # Predict using the model
    prediction = model.predict(data)[0]   # 0 or 1 usually
    
    # Convert to human-readable result
    result = "Spam" if prediction == 1 else "Not Spam"
    
    return render_template('result.html', prediction=result, message=message)
    
if __name__ == '__main__':
    app.run(debug=True)
