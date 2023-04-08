# Importing Library
import os
from flask import Flask, flash, request, redirect, url_for, render_template, Markup, jsonify
from werkzeug.utils import secure_filename
from flask import send_from_directory
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
from function_script import cleansing
import pickle

MAX_SEQUENCE_LENGTH = 64
with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

app = Flask(__name__, template_folder='templates')
app.secret_key = 'bagas_data_science'

def model_cnn_file(file):
    df_new = pd.read_csv(file, encoding='latin-1')
    df_new['tweet_clean'] = df_new['tweets'].apply(cleansing)
    df_new.drop(columns=['labels'], inplace=True)

    sentences = df_new['tweet_clean'].to_list()

    loaded_model = load_model("sentiment_analysis_model_CNN_challenge.h5")

    X_new = tokenizer.texts_to_sequences(sentences)
    X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

    # lakukan prediksi pada data baru
    y_prob = loaded_model.predict(X_new)
    y_pred = y_prob.argmax(axis=-1)

    # konversi nilai prediksi menjadi label sentimen
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    df_new['labels'] = [labels[pred] for pred in y_pred]
    df_new = df_new.to_dict(orient='records')

    return df_new

def model_lstm_file(file):
    df_new = pd.read_csv(file, encoding='latin-1')
    df_new['tweet_clean'] = df_new['tweets'].apply(cleansing)
    df_new.drop(columns=['labels'], inplace=True)

    sentences = df_new['tweet_clean'].to_list()

    loaded_model = load_model("sentiment_analysis_model_challenge.h5")

    X_new = tokenizer.texts_to_sequences(sentences)
    X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

    # lakukan prediksi pada data baru
    y_prob = loaded_model.predict(X_new)
    y_pred = y_prob.argmax(axis=-1)

    # konversi nilai prediksi menjadi label sentimen
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    df_new['labels'] = [labels[pred] for pred in y_pred]
    df_new = df_new.to_dict(orient='records')

    return df_new

def model_ffnn_file(file):
    df_new = pd.read_csv(file, encoding='latin-1')
    df_new['tweet_clean'] = df_new['tweets'].apply(cleansing)
    df_new.drop(columns=['labels'], inplace=True)

    sentences = df_new['tweet_clean'].to_list()

    loaded_model = load_model("sentiment_analysis_feedForward_neuralNetwork.h5")

    X_new = tokenizer.texts_to_sequences(sentences)
    X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

    # lakukan prediksi pada data baru
    y_prob = loaded_model.predict(X_new)
    y_pred = y_prob.argmax(axis=-1)

    # konversi nilai prediksi menjadi label sentimen
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    df_new['labels'] = [labels[pred] for pred in y_pred]
    df_new = df_new.to_dict(orient='records')

    return df_new

## Function Predict untuk Upload dan Download File ##
def predict_lstm_download(file):
        df_new = pd.read_csv(file, encoding='latin-1')
        df_new['tweet_clean'] = df_new['tweets'].apply(cleansing)
        df_new.drop(columns=['labels'], inplace=True)

        sentences = df_new['tweet_clean'].to_list()

        loaded_model = load_model("sentiment_analysis_model_challenge.h5")

        X_new = tokenizer.texts_to_sequences(sentences)
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        df_new['labels'] = [labels[pred] for pred in y_pred]
        return df_new

def predict_cnn_download(file):
        df_new = pd.read_csv(file, encoding='latin-1')
        df_new['tweet_clean'] = df_new['tweets'].apply(cleansing)
        df_new.drop(columns=['labels'], inplace=True)

        sentences = df_new['tweet_clean'].to_list()

        loaded_model = load_model("sentiment_analysis_model_CNN_challenge.h5")

        X_new = tokenizer.texts_to_sequences(sentences)
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        df_new['labels'] = [labels[pred] for pred in y_pred]
        return df_new

def predict_ffnn_download(file):
        df_new = pd.read_csv(file, encoding='latin-1')
        df_new['tweet_clean'] = df_new['tweets'].apply(cleansing)
        df_new.drop(columns=['labels'], inplace=True)

        sentences = df_new['tweet_clean'].to_list()

        loaded_model = load_model("sentiment_analysis_feedForward_neuralNetwork.h5")

        X_new = tokenizer.texts_to_sequences(sentences)
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        df_new['labels'] = [labels[pred] for pred in y_pred]
        return df_new

## Text ##

def clean_user_text(tweet):
    clean = cleansing(tweet)
    result = pd.DataFrame({'origin_text' : [tweet],
                           'clean' : [clean]},
                          index=[0])
    result['origin_text'] = result['origin_text'].to_list()
    result['clean'] = result['clean'].to_list()
    return result

def clean_user_text_swgr(tweet):
        clean = cleansing(tweet)
        result = [clean]
        result = pd.DataFrame({'origin_text' : [tweet],
                               'clean' : [clean]},
                               index=[0])
        result['clean'] = result['clean'].to_list()
        return result

def predict_input_text_lstm(tweet):
    X_new = tokenizer.texts_to_sequences(tweet['clean'])
    X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

    loaded_model = load_model("sentiment_analysis_model_challenge.h5")

    # lakukan prediksi pada data baru
    y_prob = loaded_model.predict(X_new)
    y_pred = y_prob.argmax(axis=-1)
    # konversi nilai prediksi menjadi label sentimen
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    tweet['labels'] = [labels[pred] for pred in y_pred]
    tweet = tweet.to_dict(orient='records')

    return tweet

def predict_input_text_cnn(tweet):
    X_new = tokenizer.texts_to_sequences(tweet['clean'])
    X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

    loaded_model = load_model("sentiment_analysis_model_CNN_challenge.h5")

    # lakukan prediksi pada data baru
    y_prob = loaded_model.predict(X_new)
    y_pred = y_prob.argmax(axis=-1)
    # konversi nilai prediksi menjadi label sentimen
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    tweet['labels'] = [labels[pred] for pred in y_pred]
    tweet = tweet.to_dict(orient='records')

    return tweet

def predict_input_text_ffnn(tweet):
    X_new = tokenizer.texts_to_sequences(tweet['clean'])
    X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

    loaded_model = load_model("sentiment_analysis_feedForward_neuralNetwork.h5")

    # lakukan prediksi pada data baru
    y_prob = loaded_model.predict(X_new)
    y_pred = y_prob.argmax(axis=-1)
    # konversi nilai prediksi menjadi label sentimen
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    tweet['labels'] = [labels[pred] for pred in y_pred]
    tweet = tweet.to_dict(orient='records')

    return tweet