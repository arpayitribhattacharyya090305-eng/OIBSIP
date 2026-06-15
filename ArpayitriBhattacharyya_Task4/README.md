# Email Spam Detection

This is a simple machine learning project that detects whether a message is spam or not. It uses a trained text classification model and a Streamlit interface where users can type or paste a message and instantly see the prediction.

The project was created as part of the Oasis Infobyte internship task on email spam detection.

## About the Project

Spam messages are common in emails and SMS. They often contain fake offers, urgent warnings, suspicious links, or requests for personal information. The aim of this project is to build a basic spam detector that can separate normal messages from spam messages using machine learning.

The model is trained on a labeled dataset containing two classes:

- `ham` for normal messages
- `spam` for unwanted or suspicious messages

After training, the saved model is used inside a Streamlit web app to classify new messages.

## What This Project Does

- Takes an email or SMS message as input
- Predicts whether the message is spam or ham
- Shows spam and ham probability scores
- Displays the confidence of the prediction
- Highlights common suspicious patterns in the message
- Shows model accuracy and evaluation details

## Tools and Libraries Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Project Files

```text
Email Spam Detection/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- spam.csv
|   `-- .gitkeep
|-- model/
|   |-- spam_detector.joblib
|   `-- metrics.joblib
`-- .streamlit/
```

`train_model.py` is used to train the machine learning model.

`app.py` runs the Streamlit web application.

`data/spam.csv` contains the dataset used for training.

`model/` stores the trained model and evaluation metrics.

## Dataset

The dataset should be named `spam.csv` and placed inside the `data` folder.

The expected format is:

```csv
v1,v2
ham,Your normal message here
spam,Your spam message here
```

In the code, `v1` is treated as the label and `v2` is treated as the message text.

## How to Run the Project

First, install the required libraries:

```powershell
pip install -r requirements.txt
```

Then train the model:

```powershell
python train_model.py
```

After the model is trained, start the Streamlit app:

```powershell
streamlit run app.py
```

Streamlit will open the app in your browser. If it does not open automatically, copy the local URL from the terminal and open it manually.

## How the Model Works

The project uses a machine learning pipeline with two main parts:

1. `TfidfVectorizer`

   This converts message text into numerical values so that the machine learning model can understand it. It also uses words and short word combinations to capture patterns in the message.

2. `MultinomialNB`

   This is a Naive Bayes classifier. It is commonly used for text classification problems because it is fast, simple, and works well with word-based features.

The dataset is split into training and testing data. The model learns from the training data, and its performance is checked using the testing data.

## App Features

The Streamlit app includes:

- A message input box
- Sample messages for quick testing
- Spam or ham prediction
- Confidence score
- Spam and ham probabilities
- Suspicious signal detection
- Model evaluation report
- Confusion matrix

If the trained model files are missing, the app will train the model automatically before making predictions.

## Example Messages

Spam-like message:

```text
Congratulations! You have won a free prize. Claim now by clicking the link.
```

Normal message:

```text
Hi, can we move tomorrow's project meeting to 3 PM?
```

## Output

When a message is scanned, the app shows whether it looks like spam or a legitimate message. It also shows how confident the model is, which helps the user understand how strong the prediction is.

## Conclusion

This project demonstrates how machine learning can be used for basic spam detection. It covers dataset loading, text feature extraction, model training, evaluation, saving the model, and building a small user interface for predictions.

The model is useful for learning purposes, but real-world email security systems would need much larger datasets, regular updates, and more advanced filtering techniques.
