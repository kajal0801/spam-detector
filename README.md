📧 Email Spam Detection Web App

A machine learning–based web application to detect whether an email message is Spam or Not Spam.
This project uses Naive Bayes Classifier, trained on a dataset of email texts, and provides a simple Flask web interface for users to input email text and get predictions.

🚀 Features

🔍 Classifies emails as Spam or Not Spam

🌐 Flask web app with a clean UI

🎨 Custom background images (different for input and result pages)

📦 Model & vectorizer stored as .pkl files for deployment

🛠️ Easy to deploy on platforms like Heroku, Render, or PythonAnywhere

🗂️ Project Structure
spam-detector/
│
├── spam_deployment/
│   ├── app.py                  # Flask web app
│   ├── spam_model.pkl          # Trained Naive Bayes model
│   ├── vectorizer.pkl          # CountVectorizer/TfidfVectorizer
│   ├── templates/
│   │   ├── index.html          # Input page
│   │   ├── result.html         # Result page
│   └── static/
│       ├── email.jpg           # Background image for input page
│       ├── result_bg.jpg       # Background image for result page
│
├── training_scripts/
│   ├── train_naive_bayes.py    # Script to train model
│   ├── dataset.csv             # Dataset used for training
│
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
