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
DATA_DIR = HOME_DIR + '/.ipublia/data/sentiment-analysis/'

REMOTE_DATA_URL = 'https://www.ipublia.com/data/'

MODEL_NAME = 'sa_model_de_v1.0.0.h5'
MODEL_URL = REMOTE_DATA_URL + MODEL_NAME
MODEL_FILE = DATA_DIR + MODEL_NAME

TOKENIZER_NAME = 'sa_tokenizer_de_v1.0.0.pickle'
TOKENIZER_URL = REMOTE_DATA_URL + TOKENIZER_NAME
TOKENIZER_FILE = DATA_DIR + TOKENIZER_NAME

MAX_TEXT_LENGTH = 400

# Initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
tokenizer = None

def load_remote(source_url, target_file):
    print('Downloading {0} to {1}'.format(source_url, target_file))
    
    if not isdir(DATA_DIR):
        print('Creating data directory: ' + DATA_DIR)
        makedirs(DATA_DIR)

    r = requests.get(source_url, timeout=10)
    if r.status_code == 200:
        data = r.content
        with open(target_file, 'wb') as f:                
            f.write(data)
        return 1
    else:
        print('Error ({0}) loading from {1}'.format(r.status_code, source_url))
        return -1
    return 0

def load_model_and_tokenizer():
    global model
    global tokenizer

    load_remote(MODEL_URL, MODEL_FILE)
    print('Loading model {0}'.format(MODEL_FILE))
    model = load_model(MODEL_FILE)

    load_remote(TOKENIZER_URL, TOKENIZER_FILE)
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
                print('Usage: sa-rest-server.py --host=<host> --port=<port>')
                sys.exit()
            elif opt in ("--host"):
                host = arg
            elif opt in ("-p", "--port"):
                port = int(arg)

        load_model_and_tokenizer()
        print("Staring server...")
        app.run(host, port)

    except getopt.GetoptError:
        print('Usage: sa-rest-server.py --host=<host> --port=<port>')
        sys.exit(2)
    

# If this is the main thread of execution first load the model and tokenizer and
# then start the server
# To kill the server if port is still in use after Ctrl+C: ps aux | grep rest_server
if __name__ == "__main__":
    main(sys.argv[1:])
    
