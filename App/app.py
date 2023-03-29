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

# Muat Tokenizer yang telah dilatih
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Swagger
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

app = Flask(__name__, template_folder='templates')
app.secret_key = 'bagas_data_science'

##### home interface as .html
@app.route("/", methods=['GET'])
def home():
    return render_template('home_full.html')

##### UPLOAD FILE #####
@app.route("/data_after_cleansing", methods=["GET", "POST"])
def upload_file():
        if request.method == 'POST':
            file = request.files['file']
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

            return jsonify(df_new)

  # If the request method is "GET", render the form template
        return render_template("file.html")

##### UPLOAD FILE FOR CNN #####

@app.route("/data_after_cleansing_CNN", methods=["GET", "POST"])
def upload_file_cnn():
        if request.method == 'POST':
            file = request.files['file']
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

            return jsonify(df_new)

  # If the request method is "GET", render the form template
        return render_template("file_cnn.html")

##### UPLOAD FILE FOR FFNN #####
@app.route("/data_after_cleansing_ffnn", methods=["GET", "POST"])
def upload_file_ffnn():
        if request.method == 'POST':
            file = request.files['file']
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

            return jsonify(df_new)

  # If the request method is "GET", render the form template
        return render_template("file_ffnn.html")


##### UPLOAD FILE CSV, CLEAN IT AUTOMATICALLY, AND DOWNLOAD IT #####
app.config['UPLOAD_FOLDER'] = ''
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload_download_file', methods=['GET', 'POST'])
def upload_download_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        directory_path = request.form.get("directory_path")
        filename = request.form.get("filename")

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if not filename:
                filename = secure_filename(file.filename)
        else:
            filename = secure_filename(filename)
        if file and allowed_file(file.filename):
            if directory_path:
                app.config['UPLOAD_FOLDER'] = directory_path
            else:
                app.config['UPLOAD_FOLDER'] = "C:/Users/Acer/Downloads"
            if not filename:
                filename = secure_filename(file.filename)
            else:
                filename = secure_filename(filename)

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
            df_new.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "data_clean.csv"), index=False, encoding='latin-1')

            flash('The file has been uploaded and cleaned data is saved to the directory {} as data_clean.csv'.format(app.config['UPLOAD_FOLDER']))
            return redirect(url_for('upload_download_file', name=df_new))
    return render_template('download_file.html')

##### UPLOAD FILE CSV, CLEAN IT AUTOMATICALLY, AND DOWNLOAD IT FOR CNN #####
app.config['UPLOAD_FOLDER'] = ''
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file_cnn(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload_download_file_CNN', methods=['GET', 'POST'])
def upload_download_file_cnn():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        directory_path = request.form.get("directory_path")
        filename = request.form.get("filename")

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if not filename:
                filename = secure_filename(file.filename)
        else:
            filename = secure_filename(filename)
        if file and allowed_file(file.filename):
            if directory_path:
                app.config['UPLOAD_FOLDER'] = directory_path
            else:
                app.config['UPLOAD_FOLDER'] = "C:/Users/Acer/Downloads"
            if not filename:
                filename = secure_filename(file.filename)
            else:
                filename = secure_filename(filename)

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
            df_new.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "data_clean.csv"), index=False, encoding='latin-1')

            flash('The file has been uploaded and cleaned data is saved to the directory {} as data_clean.csv'.format(app.config['UPLOAD_FOLDER']))
            return redirect(url_for('upload_download_file_cnn', name=df_new))
    return render_template('download_file_cnn.html')

##### UPLOAD FILE CSV, CLEAN IT AUTOMATICALLY, AND DOWNLOAD IT FOR FFNN #####
app.config['UPLOAD_FOLDER'] = ''
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file_cnn(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload_download_file_ffnn', methods=['GET', 'POST'])
def upload_download_file_ffnn():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        directory_path = request.form.get("directory_path")
        filename = request.form.get("filename")

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if not filename:
                filename = secure_filename(file.filename)
        else:
            filename = secure_filename(filename)
        if file and allowed_file(file.filename):
            if directory_path:
                app.config['UPLOAD_FOLDER'] = directory_path
            else:
                app.config['UPLOAD_FOLDER'] = "C:/Users/Acer/Downloads"
            if not filename:
                filename = secure_filename(file.filename)
            else:
                filename = secure_filename(filename)

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
            df_new.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "data_clean.csv"), index=False, encoding='latin-1')

            flash('The file has been uploaded and cleaned data is saved to the directory {} as data_clean.csv'.format(app.config['UPLOAD_FOLDER']))
            return redirect(url_for('upload_download_file_ffnn', name=df_new))
    return render_template('download_file_ffnn.html')

##### PREPROCESSING TEXT (INPUT TEXT) #####
@app.route("/predict_sentiment", methods=['GET', 'POST'])
def clean():
    if request.method == 'POST':
        tweet = request.form['tweet']
        clean = cleansing(tweet)
        result = [tweet]
        result = pd.DataFrame({'origin_text' : [tweet],
                               'clean' : [clean]},
                               index=[0])
        result['origin_text'] = result['origin_text'].to_list()
        result['clean'] = result['clean'].to_list()

        X_new = tokenizer.texts_to_sequences(result['clean'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("sentiment_analysis_model_challenge.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        result['labels'] = [labels[pred] for pred in y_pred]
        result = result.to_dict(orient='records')
        return jsonify(result)

    return render_template("input_text.html")

##### PREPROCESSING TEXT (INPUT TEXT) FOR CNN #####
@app.route("/predict_sentiment_cnn", methods=['GET', 'POST'])
def clean_cnn():
    if request.method == 'POST':
        tweet = request.form['tweet']
        clean = cleansing(tweet)
        result = [tweet]
        result = pd.DataFrame({'origin_text' : [tweet],
                               'clean' : [clean]},
                               index=[0])
        result['origin_text'] = result['origin_text'].to_list()
        result['clean'] = result['clean'].to_list()

        X_new = tokenizer.texts_to_sequences(result['clean'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("sentiment_analysis_model_CNN_challenge.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        result['labels'] = [labels[pred] for pred in y_pred]
        result = result.to_dict(orient='records')
        return jsonify(result)

    return render_template("input_text_cnn.html")

##### PREPROCESSING TEXT (INPUT TEXT) FOR FFNN #####
@app.route("/predict_sentiment_ffnn", methods=['GET', 'POST'])
def clean_ffnn():
    if request.method == 'POST':
        tweet = request.form['tweet']
        clean = cleansing(tweet)
        result = [tweet]
        result = pd.DataFrame({'origin_text' : [tweet],
                               'clean' : [clean]},
                               index=[0])
        result['origin_text'] = result['origin_text'].to_list()
        result['clean'] = result['clean'].to_list()

        X_new = tokenizer.texts_to_sequences(result['clean'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("sentiment_analysis_feedForward_neuralNetwork.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        result['labels'] = [labels[pred] for pred in y_pred]
        result = result.to_dict(orient='records')
        return jsonify(result)

    return render_template("input_text_ffnn.html")



##### -------------------------------------SWAGGER---------------------------------------- #####


app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling'),
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flagger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)

from function_script import cleansing
MAX_SEQUENCE_LENGTH = 64

# Muat Tokenizer yang telah dilatih
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

##### UPLOADING FILE TO CLEAN THE DATA, PREDICT THE LABLES, THEN SEE THE RESULTS AS JSON ON SWAGGER#####
@swag_from("./templates/swag_clean.yaml", methods=['POST'])
@app.route('/Upload File to Clean and Predict The Sentiment Using LSTM Model', methods=['POST'])
def upload_file_swgr_json():
        if request.method == 'POST':
            file = request.files['file']
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
            df_new = jsonify(df_new)

  # If the request method is "GET", render the form template
        return df_new


###################################################################################################

##### UPLOADING FILE TO CLEAN THE DATA, PREDICT THE LABLES, THEN SEE THE RESULTS AS JSON ON SWAGGER FOR CNN #####
@swag_from("./templates/swag_clean_cnn.yaml", methods=['POST'])
@app.route('/Upload File to Clean and Predict The Sentiment Using CNN Model', methods=['POST'])
def upload_file_swgr_json_cnn():
        if request.method == 'POST':
            file = request.files['file']
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
            df_new = jsonify(df_new)

  # If the request method is "GET", render the form template
        return df_new


###################################################################################################

##### UPLOADING FILE TO CLEAN THE DATA, PREDICT THE LABLES, THEN SEE THE RESULTS AS JSON ON SWAGGER FOR FFNN #####
@swag_from("./templates/swag_clean_ffnn.yaml", methods=['POST'])
@app.route('/Upload File to Clean and Predict The Sentiment Using FFNN Model', methods=['POST'])
def upload_file_swgr_json_ffnn():
        if request.method == 'POST':
            file = request.files['file']
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
            df_new = jsonify(df_new)

  # If the request method is "GET", render the form template
        return df_new


##### UPLOADING FILE TO CLEAN THE DATA, PREDICT THE SENTIMENT LABELS, THEN DOWNLOAD IT #####
app.config['UPLOAD_FOLDER'] = ''
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@swag_from("./templates/swag_clean.yaml", methods=['POST'])
@app.route('/Upload File, Clean The Text, Predict The Sentiment Using LSTM Model, and Download The Result', methods=['POST'])
def upload_file_swgr_download():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        directory_path = request.form.get("directory_path")
        filename = request.form.get("filename")

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if not filename:
                filename = secure_filename(file.filename)
        else:
            filename = secure_filename(filename)
        if file and allowed_file(file.filename):
            if directory_path:
                app.config['UPLOAD_FOLDER'] = directory_path
            else:
                app.config['UPLOAD_FOLDER'] = "C:/Users/Acer/Downloads"
            if not filename:
                filename = secure_filename(file.filename)
            else:
                filename = secure_filename(filename)

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
            df_new_json = df_new.to_dict(orient='records')
            df_new_json = jsonify(df_new_json)
            df_new.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "data_clean.csv"), index=False, encoding='latin-1')

            flash('The file has been uploaded and cleaned data is saved to the directory {} as data_clean.csv'.format(app.config['UPLOAD_FOLDER']))
        table = df_new_json
        return redirect(url_for('upload_download_file', name=df_new))
    return table

##### UPLOADING FILE TO CLEAN THE DATA, PREDICT THE SENTIMENT LABELS, THEN DOWNLOAD IT FOR CNN#####
app.config['UPLOAD_FOLDER'] = ''
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file_cnn(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@swag_from("./templates/swag_clean_cnn.yaml", methods=['POST'])
@app.route('/Upload File, Clean The Text, Predict The Sentiment with CNN Model, and Download The Result', methods=['POST'])
def upload_file_swgr_download_cnn():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        directory_path = request.form.get("directory_path")
        filename = request.form.get("filename")

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if not filename:
                filename = secure_filename(file.filename)
        else:
            filename = secure_filename(filename)
        if file and allowed_file(file.filename):
            if directory_path:
                app.config['UPLOAD_FOLDER'] = directory_path
            else:
                app.config['UPLOAD_FOLDER'] = "C:/Users/Acer/Downloads"
            if not filename:
                filename = secure_filename(file.filename)
            else:
                filename = secure_filename(filename)

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
            df_new_json = df_new.to_dict()
            df_new_json = jsonify(df_new_json)
            df_new.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "data_clean.csv"), index=False, encoding='latin-1')

            flash('The file has been uploaded and cleaned data is saved to the directory {} as data_clean.csv'.format(app.config['UPLOAD_FOLDER']))
        table = df_new_json
        return redirect(url_for('upload_download_file', name=df_new))
    return table


##### UPLOADING FILE TO CLEAN THE DATA, PREDICT THE SENTIMENT LABELS, THEN DOWNLOAD IT FOR FFNN #####
app.config['UPLOAD_FOLDER'] = ''
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file_cnn(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@swag_from("./templates/swag_clean_ffnn.yaml", methods=['POST'])
@app.route('/Upload File, Clean The Text, Predict The Sentiment with FFNN Model, and Download The Result', methods=['POST'])
def upload_file_swgr_download_ffnn():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        directory_path = request.form.get("directory_path")
        filename = request.form.get("filename")

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if not filename:
                filename = secure_filename(file.filename)
        else:
            filename = secure_filename(filename)
        if file and allowed_file(file.filename):
            if directory_path:
                app.config['UPLOAD_FOLDER'] = directory_path
            else:
                app.config['UPLOAD_FOLDER'] = "C:/Users/Acer/Downloads"
            if not filename:
                filename = secure_filename(file.filename)
            else:
                filename = secure_filename(filename)

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
            df_new_json = df_new.to_dict()
            df_new_json = jsonify(df_new_json)
            df_new.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "data_clean.csv"), index=False, encoding='latin-1')

            flash('The file has been uploaded and cleaned data is saved to the directory {} as data_clean.csv'.format(app.config['UPLOAD_FOLDER']))
        table = df_new_json
        return redirect(url_for('upload_download_file', name=df_new))
    return table

@swag_from("./templates/text_clean.yaml", methods=['POST'])
@app.route('/Clean and Predict The Sentiment From Your Text Using LSTM Model', methods=['POST'])
def text_cleansing_swgr():
    if request.method == 'POST':
        text = request.form.get('text')
        result = cleansing(text)
        result = [result]
        result = pd.DataFrame({'original_text' : text,
                               'clean' : result})
        result['clean'] = result['clean'].to_list()

        X_new = tokenizer.texts_to_sequences(result['clean'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("sentiment_analysis_model_challenge.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        result['labels'] = [labels[pred] for pred in y_pred]
        result = result.to_dict()
        result = jsonify(result)
    
    return result

@swag_from("./templates/text_clean_cnn.yaml", methods=['POST'])
@app.route('/Clean and Predict The Sentiment with CNN Model From Your Text', methods=['POST'])
def text_cleansing_swgr_cnn():
    if request.method == 'POST':
        text = request.form.get('text')
        result = cleansing(text)
        result = [result]
        result = pd.DataFrame({'original_text' : text,
                               'clean' : result})
        result['clean'] = result['clean'].to_list()

        X_new = tokenizer.texts_to_sequences(result['clean'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("sentiment_analysis_model_CNN_challenge.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        result['labels'] = [labels[pred] for pred in y_pred]
        result = result.to_dict()
        result = jsonify(result)
    
    return result

@swag_from("./templates/text_clean_ffnn.yaml", methods=['POST'])
@app.route('/Clean and Predict The Sentiment with FFNN Model From Your Text', methods=['POST'])
def text_cleansing_swgr_ffnn():
    if request.method == 'POST':
        text = request.form.get('text')
        result = cleansing(text)
        result = [result]
        result = pd.DataFrame({'original_text' : text,
                               'clean' : result})
        result['clean'] = result['clean'].to_list()

        X_new = tokenizer.texts_to_sequences(result['clean'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("sentiment_analysis_model_CNN_challenge.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        result['labels'] = [labels[pred] for pred in y_pred]
        result = result.to_dict()
        result = jsonify(result)
    
    return result

if __name__ == '__main__':
    app.run(debug=True)