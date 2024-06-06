import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string  # Import the string module

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to preprocess the text
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r'[^\w\s]', '', message)
    lemmatizer = WordNetLemmatizer()
    message = ' '.join([lemmatizer.lemmatize(word) for word in message.split()])
    return message

# Title
st.title("Spam Detection App")

# Sample dataset to train the model
sample_data = {
    'text': [
        'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)',
        'Nah I dont think he goes to usf, he lives around here though',
        'WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461.',
        'I have a date on Sunday with Will!!',
        'Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030',
        'I\'m gonna be home soon and i don\'t want to talk about this stuff anymore tonight, k? I\'ve cried enough today.',
        'SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ T&C apply www.9000cash.com',
        'I HAVE A DATE ON SUNDAY WITH WILL!!',
        'XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap.xxxmobilemovieclub.com?n=QJKGIGHJJGCBL',
        'Oh k...i\'m watching here:)',
    ],
    'target': [
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0
    ]
}

# Creating DataFrame
df = pd.DataFrame(sample_data)

# Text Preprocessing
ps = nltk.PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

df['transformed_text'] = df['text'].apply(transform_text)

# Model Training
X = df['transformed_text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

model.fit(X_train, y_train)

# Input for new messages
st.header("Enter a message to classify")
new_message = st.text_area("Message")

if st.button("Classify"):
    if new_message:
        preprocessed_message = transform_text(new_message)
        prediction = model.predict([preprocessed_message])[0]
        st.write("Prediction:", "Spam" if prediction == 1 else "Ham")
    else:
        st.write("Please enter a message to classify.")
