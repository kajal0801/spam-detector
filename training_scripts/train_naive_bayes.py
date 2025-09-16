import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load the dataset
df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'message'])

# See the first 5 rows
print(df.head())

# Check dataset shape
print("Total messages:", df.shape[0])

# Check the distribution of labels
print(df['label'].value_counts())

print(df.isnull().sum())

df = df.dropna()



# Create the vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the message column
X = vectorizer.fit_transform(df['message'])

# Labels
y = df['label'].map({'ham': 0, 'spam': 1})  # convert 'ham' to 0, 'spam' to 1

# Check the shape of X , x is input features
print(X.shape) # it will show number of messages and number of unique words found in all messages after TF-IDF vectorisation
#print(y.head()) y is the target or ouput variable for our machine learning model , it tells the model whether each email is spam or ham


#Split data into training and testing sets 20% for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training messages:", X_train.shape[0])#Purpose: The model learns patterns from this data.
print("Testing messages:", X_test.shape[0])#It is a set of emails (X_test) that the model has never seen before


model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))





