from os import makedirs, listdir
from os.path import expanduser, isdir, isfile, join
import sys, getopt
import re
import requests
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import flask

HOME_DIR = expanduser('~')
DATA_DIR = HOME_DIR + "/.publia/data/sentiment-analysis"

MODEL_URL = "https://drive.google.com/uc?export=download&id=1Wgx2K2aIB9oztqYWMkwREBDRtKrpAg-8"
MODEL_NAME = 'model_de.h5'
MODEL_FILE = DATA_DIR + "/" + MODEL_NAME

TOKENIZER_URL = "https://drive.google.com/uc?export=download&id=13wYposDP3TqbiDM-Y72n9cllOvHVszQh"
TOKENIZER_NAME = 'tokenizer_de.pickle'
TOKENIZER_FILE = DATA_DIR + "/" + TOKENIZER_NAME

MAX_TEXT_LENGTH = 400

# Initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
tokenizer = None

def load_model_from_google_drive():
    """ Load saved model. """
    global model

    # Create data and model dir
    if not isdir(DATA_DIR):
        print('Creating data directory: ' + DATA_DIR)
        makedirs(DATA_DIR)

    if not isfile(MODEL_FILE):
        print('Downloading model {0} from {1}'.format(MODEL_NAME, MODEL_URL))
     
        r = requests.get(MODEL_URL, timeout=10)
        if r.status_code == 200:
            data = r.content
            if r.cookies:
                confirm = re.search(r'confirm=(.{4})', data.decode('utf-8'))
                if confirm:
                    confirmed_url = MODEL_URL + '&confirm=' + confirm.group(1)
                    r2 = requests.get(confirmed_url, cookies=r.cookies)
                    data = r2.content

            with open(MODEL_FILE, 'wb') as f:                
                f.write(data)
        else:
            print('Error ({0}) loading model from {1}'.format(r.status_code, MODEL_URL))

    print('Loading model {0}'.format(MODEL_FILE))
    model = load_model(MODEL_FILE)

def load_tokenizer_from_google_drive():
    """ Load saved tokenizer. """
    global tokenizer

    # Create data and model dir
    if not isdir(DATA_DIR):
        print('Creating data directory: ' + DATA_DIR)
        makedirs(DATA_DIR)

    if not isfile(TOKENIZER_FILE):
        print('Downloading tokenizer {0} from {1}'.format(TOKENIZER_NAME, TOKENIZER_URL))
        r = requests.get(TOKENIZER_URL, timeout=10)
        if r.status_code == 200:
            data = r.content
            if r.cookies:
                confirm = re.search(r'confirm=(.{4})', data.decode('utf-8'))
                if confirm:
                    confirmed_url = TOKENIZER_URL + '&confirm=' + confirm.group(1)
                    r2 = requests.get(confirmed_url, cookies=r.cookies)
                    data = r2.content

            with open(TOKENIZER_FILE, 'wb') as f:                
                f.write(data)
        else:
            print('Error ({0}) loading tokenizer from {1}'.format(r.status_code, TOKENIZER_URL))

    print('Loading tokenizer {0}'.format(TOKENIZER_FILE))
    with open(TOKENIZER_FILE, 'rb') as handle:
        tokenizer = pickle.load(handle)

def prepare_texts(texts):
    """
    Create the input sequences and do the padding of the vector.

    Arguments:
        texts -- an array of texts to prepare for prediction.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded_texts = pad_sequences(sequences, maxlen=MAX_TEXT_LENGTH)
    return np.array(padded_texts)

@app.route("/predict", methods=["POST"])
def predict():
    data = { "success": False }

    if flask.request.method == "POST":
        if flask.request.json["texts"]:
            # Pead the texts
            texts = flask.request.json["texts"]

            # Preprocess the texts and prepare them for classification
            prepared_texts = prepare_texts(texts)

            # Classify the input texts and then initialize the list
            # of predictions to return to the client
            results = model.predict(prepared_texts)
            data["predictions"] = []

            # Loop over the results and add them to the list of
            # returned predictions
            for i in range(len(results)):
                r = { "text": texts[i], "probability": float(results[i][0]) }
                data["predictions"].append(r)

            # Indicate that the request was a success
            data["success"] = True

    # Return the data dictionary as a JSON response
    return flask.jsonify(data)

def main(argv):
    host = None
    port = None

    try:
        opts, args = getopt.getopt(argv, "hp:", ["host=", "port="])
        for opt, arg in opts:
            if opt == "-h":
                print('Usage: rest-server.py --host=<host> --port=<port>')
                sys.exit()
            elif opt in ("--host"):
                host = arg
            elif opt in ("-p", "--port"):
                port = int(arg)

        print("Staring server...")
        load_model_from_google_drive()
        load_tokenizer_from_google_drive()
        app.run(host, port)

    except getopt.GetoptError:
        print('Usage: rest-server.py --host=<host> --port=<port>')
        sys.exit(2)
    

# If this is the main thread of execution first load the model and tokenizer and
# then start the server
# To kill the server if port is still in use after Ctrl+C: ps aux | grep rest_server
if __name__ == "__main__":
    main(sys.argv[1:])
    
