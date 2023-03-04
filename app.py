##### FROM LINE 5 UNTIL 316 IS JUST A FLASK #####

##### FROM LINE 323 UNTIL 586 IS FLASK + SWAGGER UI. The URL : http://127.0.0.1:5000/docs/ #####

# Importing Library
import sqlite3
import os
from flask import Flask, flash, request, redirect, url_for, render_template, Markup, jsonify
from werkzeug.utils import secure_filename
from flask import send_from_directory
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras.models import load_model

MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
word_index = tokenizer.word_index

# Remove Stopwords
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.corpus.stopwords.words('indonesian')

# Swagger
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

app = Flask(__name__, template_folder='templates')
app.secret_key = 'bagas_data_science'

# Function to cleaning the data
def cleansing(text):
    # Make sentence being lowercase
    text = text.lower()

    # Remove user, rt, \n, retweet, \t, url, xd
    pattern_1 = r'(user|retweet|\\t|\\r|url|xd|orang|kalo)'
    text = re.sub(pattern_1, '', text)

    # Remove mention
    pattern_2 = r'@[^\s]+'
    text = re.sub(pattern_2, '', text)

    # Remove hashtag
    pattern_3 = r'#([^\s]+)'
    text = re.sub(pattern_3, '', text)

    # Remove general punctuation, math operation char, etc.
    pattern_4 = r'[\,\@\*\_\-\!\:\;\?\'\.\"\)\(\{\}\<\>\+\%\$\^\#\/\`\~\|\&\|]'
    text = re.sub(pattern_4, ' ', text)

    # Remove single character
    pattern_5 = r'\b\w{1,3}\b'
    text = re.sub(pattern_5, '', text)

    # Remove emoji
    pattern_6 = r'\\[a-z0-9]{1,5}'
    text = re.sub(pattern_6, '', text)

    # Remove digit character
    pattern_7 = r'\d+'
    text = re.sub(pattern_7, '', text)

    # Remove url start with http or https
    pattern_8 = r'(https|https:)'
    text = re.sub(pattern_8, '', text)

    # Remove (\); ([); (])
    pattern_9 = r'[\\\]\[]'
    text = re.sub(pattern_9, '', text)

    # Remove character non ASCII
    pattern_10 = r'[^\x00-\x7f]'
    text = re.sub(pattern_10, '', text)

    # Remove character non ASCII
    pattern_11 = r'(\\u[0-9A-Fa-f]+)'
    text = re.sub(pattern_11, '', text)

    # Remove multiple whitespace
    pattern_12 = r'(\s+|\\n)'
    text = re.sub(pattern_12, ' ', text)

    # Remove "wkwkwk"
    pattern_13 = r'\bwk\w+'
    text = re.sub(pattern_13, '', text)
    
    # Remove whitespace at the first and end sentences
    text = text.rstrip()
    text = text.lstrip()
    return text

def replaceThreeOrMore(text):
    # Pattern to look for three or more repetitions of any character, including newlines.
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", text)

indo_stop_words = stopwords.words("indonesian")

def remove_stopwords(text):
    return ' '.join([word for word in word_tokenize(text) if word not in indo_stop_words])

##### home interface as .html
@app.route("/", methods=['GET'])
def home():
    return render_template('home.html')

##### READING FILE, SHOW DATAFRAME .HTML #####
@app.route("/data_before_cleansing", methods=["GET", "POST"])
def read_file_to_html():
    conn = sqlite3.connect("D:\github branch bagas\challenge_platinum_binar_academy\\binar_platinum_challenge.db")
    cursor = conn.cursor()

    if request.method == 'POST':
        csv_file = request.files.get("file")
        if not csv_file or not csv_file.filename.endswith('.csv'):
            return 'Invalid file'

    # Read the .csv file into a Pandas dataframe
        df = pd.read_csv(csv_file, encoding='latin-1')

        conn = sqlite3.connect("D:\github branch bagas\challenge_platinum_binar_academy\\binar_platinum_challenge.db")
        cursor = conn.cursor()
        table = df.to_sql('challenge', conn, if_exists='replace') # to prove that this code is running well, drop the "upload_and_download_csv_file" table first from the database via the app_sqlite.py file
        conn.commit()
        conn.close()

        df = df.to_html(index=False, justify='left')

        return Markup(df)

    # If the request method is "GET", render the form template
    return render_template("file.html")


##### UPLOAD FILE #####

@app.route("/data_after_cleansing", methods=["GET", "POST"])
def upload_file():
        if request.method == 'POST':
            file = request.files['file']
            df_new = pd.read_csv(file, encoding='latin-1')
            df_new['tweet_clean'] = df_new['tweets'].apply(cleansing)
            df_new.drop(columns=['labels'], inplace=True)

            sentences = df_new['tweet_clean'].to_list()

            loaded_model = load_model("XXX")

            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(sentences)
            X_new = tokenizer.texts_to_sequences(sentences)
            X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

            # lakukan prediksi pada data baru
            y_prob = loaded_model.predict(X_new)
            y_pred = y_prob.argmax(axis=-1)
            
            # konversi nilai prediksi menjadi label sentimen
            labels = {0: "negative", 1: "neutral", 2: "positive"}
            df_new['labels'] = [labels[pred] for pred in y_pred]

            table = df_new.to_html()
            return Markup(table)

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

            loaded_model = load_model("D:\github branch bagas\challenge_platinum_binar_academy\sentiment_analysis_model_CNN_challenge.h5")

            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(sentences)
            X_new = tokenizer.texts_to_sequences(sentences)
            X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

            # lakukan prediksi pada data baru
            y_prob = loaded_model.predict(X_new)
            y_pred = y_prob.argmax(axis=-1)
            
            # konversi nilai prediksi menjadi label sentimen
            labels = {0: "negative", 1: "neutral", 2: "positive"}
            df_new['labels'] = [labels[pred] for pred in y_pred]

            table = df_new.to_html()
            return Markup(table)

  # If the request method is "GET", render the form template
        return render_template("file_cnn.html")


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

            loaded_model = load_model("XXX")

            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(sentences)
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

            loaded_model = load_model("D:\github branch bagas\challenge_platinum_binar_academy\sentiment_analysis_model_CNN_challenge.h5")

            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(sentences)
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


##### CLEANSING DATA BY INDEX NUMBER OF 'TWEET' COLUMN ##### 
@app.route("/cleansing_tweet_column", methods=['GET', 'POST'])
def index():
    conn = sqlite3.connect("D:\github branch bagas\challenge_platinum_binar_academy\\binar_platinum_challenge.db")
    cursor = conn.cursor()
    if request.method == 'POST':
        # get the value of the 'row' field from the form data
        before = request.form.get('before')

        # convert the 'row' value to an integer
        before = int(before)

        # select the row using the 'before' value, then apply cleansing function
        cursor.execute('''SELECT * FROM challenge''')
        df = pd.read_sql_query('''SELECT * FROM challenge''', conn)
        conn.commit()
        values_data = df[['Tweet']].iloc[before].apply(cleansing)

        # apply replaceThreeOrMore function to variable values_data
        values_data = values_data.apply(replaceThreeOrMore)

        # define stopwords in Indonesian
        indo_stop_words = stopwords.words("indonesian")

        # Function to remove stopwords
        def remove_stopwords(text):
            return ' '.join([word for word in word_tokenize(text) if word not in indo_stop_words])

        # apply the function to all string columns in the dataframe
        table = values_data.apply(remove_stopwords)

        # format the values_data to list
        values_str = table.to_list()

        # select the row using the 'row' value
        cursor.execute('''SELECT * FROM challenge''')
        df = pd.read_sql_query('''SELECT * FROM challenge''', conn)
        conn.commit()
        before_data = df[['Tweet']].iloc[before]

        # format the values to list
        before_pre = before_data.to_list()
        conn.close()

        return redirect(url_for("by_index", clean=values_str, before=before_pre))

    return render_template("index_2.html")

@app.route("/by_index", methods=['GET'])
def by_index():
    clean = request.args.get('clean')
    before = request.args.get('before')
    return f'''
    TWEET BEFORE PREPROCESSING (CLEANSING): <br> <br> {before} <br> <br> <br> <br> <br>
    TWEET AFTER PREPROCESSING (CLEANSING): <br> <br> {clean}
    '''

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

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(result['clean'])
        X_new = tokenizer.texts_to_sequences(result['clean'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("D:\Binar Academy - Data Science\challenge_platinum\challenge_platinum_binar_academy\sentiment_analysis_model_challenge.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        result['labels'] = [labels[pred] for pred in y_pred]
        table = result.to_html()
        return Markup(table)

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

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(result['clean'])
        X_new = tokenizer.texts_to_sequences(result['clean'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("D:\github branch bagas\challenge_platinum_binar_academy\sentiment_analysis_model_CNN_challenge.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        result['labels'] = [labels[pred] for pred in y_pred]
        table = result.to_html()
        return Markup(table)

    return render_template("input_text_cnn.html")



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

def cleansing(text):
    # Make sentence being lowercase
    text = text.lower()

    # Remove user, rt, \n, retweet, \t, url, xd
    pattern_1 = r'(user|retweet|\\t|\\r|url|xd|orang|kalo)'
    text = re.sub(pattern_1, '', text)

    # Remove mention
    pattern_2 = r'@[^\s]+'
    text = re.sub(pattern_2, '', text)

    # Remove hashtag
    pattern_3 = r'#([^\s]+)'
    text = re.sub(pattern_3, '', text)

    # Remove general punctuation, math operation char, etc.
    pattern_4 = r'[\,\@\*\_\-\!\:\;\?\'\.\"\)\(\{\}\<\>\+\%\$\^\#\/\`\~\|\&\|]'
    text = re.sub(pattern_4, ' ', text)

    # Remove single character
    pattern_5 = r'\b\w{1,3}\b'
    text = re.sub(pattern_5, '', text)

    # Remove emoji
    pattern_6 = r'\\[a-z0-9]{1,5}'
    text = re.sub(pattern_6, '', text)

    # Remove digit character
    pattern_7 = r'\d+'
    text = re.sub(pattern_7, '', text)

    # Remove url start with http or https
    pattern_8 = r'(https|https:)'
    text = re.sub(pattern_8, '', text)

    # Remove (\); ([); (])
    pattern_9 = r'[\\\]\[]'
    text = re.sub(pattern_9, '', text)

    # Remove character non ASCII
    pattern_10 = r'[^\x00-\x7f]'
    text = re.sub(pattern_10, '', text)

    # Remove character non ASCII
    pattern_11 = r'(\\u[0-9A-Fa-f]+)'
    text = re.sub(pattern_11, '', text)

    # Remove multiple whitespace
    pattern_12 = r'(\s+|\\n)'
    text = re.sub(pattern_12, ' ', text)

    # Remove "wkwkwk"
    pattern_13 = r'\bwk\w+'
    text = re.sub(pattern_13, '', text)
    
    # Remove whitespace at the first and end sentences
    text = text.rstrip()
    text = text.lstrip()
    return text

def replaceThreeOrMore(text):
    # Pattern to look for three or more repetitions of any character, including newlines.
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", text)

indo_stop_words = stopwords.words("indonesian")

def remove_stopwords(text):
    return ' '.join([word for word in word_tokenize(text) if word not in indo_stop_words]) 


##################################################################################################################


##### UPLOADING FILE TO CLEAN THE DATA, PREDICT THE LABLES, THEN SEE THE RESULTS AS JSON ON SWAGGER#####
@swag_from("./templates/swag_clean.yaml", methods=['POST'])
@app.route('/Upload File to Clean and Predict The Sentiment', methods=['POST'])
def upload_file_swgr_json():
        if request.method == 'POST':
            file = request.files['file']
            df_new = pd.read_csv(file, encoding='latin-1')
            df_new['tweet_clean'] = df_new['tweets'].apply(cleansing)
            df_new.drop(columns=['labels'], inplace=True)

            sentences = df_new['tweet_clean'].to_list()

            loaded_model = load_model("D:\Binar Academy - Data Science\challenge_platinum\challenge_platinum_binar_academy\sentiment_analysis_model_challenge.h5")

            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(sentences)
            X_new = tokenizer.texts_to_sequences(sentences)
            X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

            # lakukan prediksi pada data baru
            y_prob = loaded_model.predict(X_new)
            y_pred = y_prob.argmax(axis=-1)
            
            # konversi nilai prediksi menjadi label sentimen
            labels = {0: "negative", 1: "neutral", 2: "positive"}
            df_new['labels'] = [labels[pred] for pred in y_pred]

            table = df_new.to_json()

  # If the request method is "GET", render the form template
        return table


###################################################################################################

##### UPLOADING FILE TO CLEAN THE DATA, PREDICT THE LABLES, THEN SEE THE RESULTS AS JSON ON SWAGGER FOR CNN #####
@swag_from("./templates/swag_clean.yaml", methods=['POST'])
@app.route('/Upload File to Clean and Predict The Sentiment For CNN', methods=['POST'])
def upload_file_swgr_json_cnn():
        if request.method == 'POST':
            file = request.files['file']
            df_new = pd.read_csv(file, encoding='latin-1')
            df_new['tweet_clean'] = df_new['tweets'].apply(cleansing)
            df_new.drop(columns=['labels'], inplace=True)

            sentences = df_new['tweet_clean'].to_list()

            loaded_model = load_model("D:\github branch bagas\challenge_platinum_binar_academy\sentiment_analysis_model_CNN_challenge.h5")

            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(sentences)
            X_new = tokenizer.texts_to_sequences(sentences)
            X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

            # lakukan prediksi pada data baru
            y_prob = loaded_model.predict(X_new)
            y_pred = y_prob.argmax(axis=-1)
            
            # konversi nilai prediksi menjadi label sentimen
            labels = {0: "negative", 1: "neutral", 2: "positive"}
            df_new['labels'] = [labels[pred] for pred in y_pred]

            table = df_new.to_json()

  # If the request method is "GET", render the form template
        return table


###################################################################################################


##### UPLOADING FILE TO CLEAN THE DATA, PREDICT THE SENTIMENT LABELS, THEN DOWNLOAD IT #####
app.config['UPLOAD_FOLDER'] = ''
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@swag_from("./templates/swag_clean.yaml", methods=['POST'])
@app.route('/Upload File, Clean The Text, Predict The Sentiment, and Download The Result', methods=['POST'])
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

            loaded_model = load_model("D:\Binar Academy - Data Science\challenge_platinum\challenge_platinum_binar_academy\sentiment_analysis_model_challenge.h5")

            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(sentences)
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
        table = df_new.to_json()
        return redirect(url_for('upload_download_file', name=df_new))
    return table


@swag_from("./templates/text_clean.yaml", methods=['POST'])
@app.route('/Clean and Predict The Sentiment From Your Text', methods=['POST'])
def text_cleansing_swgr():
    if request.method == 'POST':
        text = request.form.get('text')
        result = cleansing(text)
        result = [result]
        result = pd.DataFrame({'original_text' : text,
                               'clean' : result})
        result['clean'] = result['clean'].to_list()

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(result['clean'])
        X_new = tokenizer.texts_to_sequences(result['clean'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("D:\Binar Academy - Data Science\challenge_platinum\challenge_platinum_binar_academy\sentiment_analysis_model_challenge.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        result['labels'] = [labels[pred] for pred in y_pred]
    
    return result.to_json()


################################################################################################
##### UPLOADING FILE TO CLEAN THE DATA, PREDICT THE SENTIMENT LABELS, THEN DOWNLOAD IT FOR CNN#####
app.config['UPLOAD_FOLDER'] = ''
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file_cnn(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@swag_from("./templates/swag_clean.yaml", methods=['POST'])
@app.route('/Upload File, Clean The Text, Predict The Sentiment with CNN, and Download The Result', methods=['POST'])
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

            loaded_model = load_model("D:\github branch bagas\challenge_platinum_binar_academy\sentiment_analysis_model_CNN_challenge.h5")

            tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            tokenizer.fit_on_texts(sentences)
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
        table = df_new.to_json()
        return redirect(url_for('upload_download_file', name=df_new))
    return table


@swag_from("./templates/text_clean.yaml", methods=['POST'])
@app.route('/Clean and Predict The Sentiment with CNN From Your Text', methods=['POST'])
def text_cleansing_swgr_cnn():
    if request.method == 'POST':
        text = request.form.get('text')
        result = cleansing(text)
        result = [result]
        result = pd.DataFrame({'original_text' : text,
                               'clean' : result})
        result['clean'] = result['clean'].to_list()

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(result['clean'])
        X_new = tokenizer.texts_to_sequences(result['clean'])
        X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

        loaded_model = load_model("D:\github branch bagas\challenge_platinum_binar_academy\sentiment_analysis_model_CNN_challenge.h5")

        # lakukan prediksi pada data baru
        y_prob = loaded_model.predict(X_new)
        y_pred = y_prob.argmax(axis=-1)
        # konversi nilai prediksi menjadi label sentimen
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        result['labels'] = [labels[pred] for pred in y_pred]
    
    return result.to_json()


################################################################################################


# Clean dataframe by index, then show the results as a JSON
@swag_from("./templates/swagger_index.yaml", methods=['POST'])
@app.route("/Clean dataframe by index. Choose 0 - 13168", methods=['GET','POST'])
def index_swgr():
    conn = sqlite3.connect('database/challenge_level_3.db')
    cursor = conn.cursor()
    if request.method == 'POST':
        # get the value of the 'row' field from the form data
        index = int(request.form.get('index'))

        # select the row using the 'before' value, then apply cleansing function
        cursor.execute('''SELECT * FROM challenge''')
        df = pd.read_sql_query('''SELECT * FROM challenge''', conn)
        conn.commit()
        values_data = df[['Tweet']].iloc[index].apply(cleansing)

        # apply replaceThreeOrMore function to variable values_data
        values_data = values_data.apply(replaceThreeOrMore)

        # apply the function to all string columns in the dataframe
        table = values_data.apply(remove_stopwords)

        # format the values_data to list
        values_str = table.to_list()

        # select the row using the 'row' value
        cursor.execute('''SELECT * FROM challenge''')
        df = pd.read_sql_query('''SELECT * FROM challenge''', conn)
        conn.commit()
        before_data = df[['Tweet']].iloc[index]

        # format the values to list
        before_pre = before_data.to_list()
        conn.close()

    return jsonify(clean=values_str, before=before_pre)

##### UPLOAD FILE TO CLEAN IT, AND SHOW IT AS JSON #####
@swag_from("./templates/swag_clean.yaml", methods=['POST'])
@app.route("/data_before_cleansing_swagger", methods=["GET", "POST"])
def read_file_to_json():
    conn = sqlite3.connect('database/challenge_level_3.db')
    cursor = conn.cursor()
    if request.method == 'POST':
        csv_file = request.files.get("file")
        if not csv_file or not csv_file.filename.endswith('.csv'):
            return 'Invalid file'

        df = pd.read_csv(csv_file, encoding='latin-1')

        # Replace the existing table with the cleaned data
        table = df.to_sql('challenge', conn, if_exists='replace') # to prove that this code is running well, drop the "challenge_cleaned_flask_swagger" table from the database via the app_sqlite.py file
        conn.commit()

    # Convert the dataframe as JSON
        table = df.to_json()
        conn.close()
        return table

if __name__ == '__main__':
    app.run(debug=True)