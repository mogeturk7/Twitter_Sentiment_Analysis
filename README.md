# Twitter Sentiment Analysis Application
This project consists of a Dash application that predicts the sentiment of tweets. The model was trained on a subset of a [Kaggle dataset of 1.6 million tweets.](https://www.kaggle.com/kazanova/sentiment140) The application includes a tweet randomizer for both positive and negative predicted tweets as well as a text box for a user to test out tweet sentiments. More information about the model can be found on the "model" tab of the application. Please follow the instructions below to successfully run the app. I plan to add new features to this app and host it on a public domain soon. Stay tuned!

## Installing Necessary Packages
All required packages are in the requirements.txt file. Simply go to your terminal and run **pip install -r requirements.txt** to install the required packages.

## Running the application
To run the application, simply run the **application.py** file. You will get a response stating that the application is "Running on http://127.0.0.1:8000/". Go to this server address and you should be able to view the application. 

## Other files
While you only need to run application.py in order to run the app, it could be helpful to know what the other files do.
* **sentiment_predictor.py:** Contains the actual sentiment prediction model using a TFIDF Vectorizer and LinearSVC. Go to the bottom and uncomment the last few lines if you want to train the model and apply labels again. Make sure to comment out these lines when you run application.py so that it doesn't have to train the model every time the app is run.
* **home.py:** Contains the home page layout of the app, using bootstrap css classes to make the styling of html components easy.
* **model.py:** Contains the model page layout of the app. This page explains the training dataset and model training in more detail and provides a confusion matrix.
* **navigation.py:** Contains the navigation of the app so that the user can easily navigate from the home page to the model page and vice versa.
