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
from model_script import model_cnn_file, model_lstm_file, model_ffnn_file
from model_script import predict_lstm_download, predict_cnn_download, predict_ffnn_download
from model_script import clean_user_text, clean_user_text_swgr, predict_input_text_lstm, predict_input_text_cnn, predict_input_text_ffnn

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
        df_new = model_lstm_file(file)
        return jsonify(df_new)

  # If the request method is "GET", render the form template
    return render_template("file.html")

##### UPLOAD FILE FOR CNN #####

@app.route("/data_after_cleansing_CNN", methods=["GET", "POST"])
def upload_file_cnn():
    if request.method == 'POST':
        file = request.files['file']
        df_new = model_cnn_file(file)
        return jsonify(df_new)
    
    # If the request method is "GET", render the form template
    return render_template("file_cnn.html")


##### UPLOAD FILE FOR FFNN #####
@app.route("/data_after_cleansing_ffnn", methods=["GET", "POST"])
def upload_file_ffnn():
    if request.method == 'POST':
        file = request.files['file']
        df_new = model_ffnn_file(file)
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

            df_new = predict_lstm_download(file)

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

            df_new = predict_cnn_download(file)
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

            df_new = predict_ffnn_download(file)
            df_new.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "data_clean.csv"), index=False, encoding='latin-1')

            flash('The file has been uploaded and cleaned data is saved to the directory {} as data_clean.csv'.format(app.config['UPLOAD_FOLDER']))
            return redirect(url_for('upload_download_file_ffnn', name=df_new))
    return render_template('download_file_ffnn.html')

##### PREPROCESSING TEXT (INPUT TEXT) #####
@app.route("/predict_sentiment", methods=['GET', 'POST'])
def clean():
    if request.method == 'POST':
        tweet = request.form['tweet']
        result = clean_user_text(tweet)
        result = predict_input_text_lstm(result)
        return jsonify(result)

    return render_template("input_text.html")

##### PREPROCESSING TEXT (INPUT TEXT) FOR CNN #####
@app.route("/predict_sentiment_cnn", methods=['GET', 'POST'])
def clean_cnn():
    if request.method == 'POST':
        tweet = request.form['tweet']
        result = clean_user_text(tweet)
        result = predict_input_text_cnn(result)
        return jsonify(result)

    return render_template("input_text_cnn.html")

##### PREPROCESSING TEXT (INPUT TEXT) FOR FFNN #####
@app.route("/predict_sentiment_ffnn", methods=['GET', 'POST'])
def clean_ffnn():
    if request.method == 'POST':
        tweet = request.form['tweet']
        result = clean_user_text(tweet)
        result = predict_input_text_ffnn(result)
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
            df_new = model_lstm_file(file)
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
            df_new = model_cnn_file(file)
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
            df_new = model_ffnn_file(file)
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

            df_new = predict_lstm_download(file)
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

            df_new = predict_cnn_download(file)
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

            df_new = predict_ffnn_download(file)
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
        tweet = request.form.get('tweet')
        result = clean_user_text_swgr(tweet)
        result = predict_input_text_lstm(result)
        result = jsonify(result)
    
    return result

@swag_from("./templates/text_clean.yaml", methods=['POST'])
@app.route('/Clean and Predict The Sentiment From Your Text Using CNN Model', methods=['POST'])
def text_cleansing_swgr_cnn():
    if request.method == 'POST':
        tweet = request.form.get('tweet')
        result = clean_user_text_swgr(tweet)
        result = predict_input_text_cnn(result)
        result = jsonify(result)
    
    return result

@swag_from("./templates/text_clean.yaml", methods=['POST'])
@app.route('/Clean and Predict The Sentiment From Your Text Using FFNN Model', methods=['POST'])
def text_cleansing_swgr_ffnn():
    if request.method == 'POST':
        tweet = request.form.get('tweet')
        result = clean_user_text_swgr(tweet)
        result = predict_input_text_ffnn(result)
        result = jsonify(result)
    
    return result

if __name__ == '__main__':
    app.run(debug=True)