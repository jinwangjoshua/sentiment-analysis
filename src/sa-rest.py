import os
import os.path
import sys, getopt
import re
import requests
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import flask

HOME_DIR = os.path.expanduser('~')
DATA_DIR = os.path.join(HOME_DIR, '.ipublia', 'data', 'sentiment-analysis')
REMOTE_DATA_URL = 'https://www.ipublia.com/data/sentiment-analysis'
MAX_TEXT_LENGTH = 400

# Initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
tokenizer = None

def load_remote(source_url, target_file):
    if not os.path.isdir(DATA_DIR):
        print('Creating data directory: ' + DATA_DIR)
        os.makedirs(DATA_DIR)

    if not os.path.isfile(target_file):
        print('Downloading {0} to {1}'.format(source_url, target_file))
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

def load_model_and_tokenizer(model_id):
    global model
    global tokenizer

    model_name = 'model_' + model_id + '.h5'
    model_file = os.path.join(DATA_DIR, model_name)
    model_url = REMOTE_DATA_URL + '/' + model_name

    load_remote(model_url, model_file)
    print('Loading model {0}'.format(model_file))
    model = load_model(model_file)

    tokenizer_name = 'tokenizer_' + model_id + '.pickle'
    tokenizer_file = os.path.join(DATA_DIR, tokenizer_name)
    tokenizer_url = REMOTE_DATA_URL + '/' + tokenizer_name

    load_remote(tokenizer_url, tokenizer_file)
    print('Loading tokenizer {0}'.format(tokenizer_file))
    with open(tokenizer_file, 'rb') as handle:
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

@app.route('/predict', methods=['POST'])
def predict():
    data = { 'success': False }

    if flask.request.method == 'POST':
        if flask.request.json['texts']:
            # Pead the texts
            texts = flask.request.json['texts']

            # Preprocess the texts and prepare them for classification
            prepared_texts = prepare_texts(texts)

            # Classify the input texts and then initialize the list
            # of predictions to return to the client
            results = model.predict(prepared_texts)
            data["predictions"] = []

            # Loop over the results and add them to the list of
            # returned predictions
            for i in range(len(results)):
                r = { 'text': texts[i], 'probability': float(results[i][0]) }
                data['predictions'].append(r)

            # Indicate that the request was a success
            data["success"] = True

    # Return the data dictionary as a JSON response
    return flask.jsonify(data)

def main(argv):
    host = None
    port = None
    model_id = 'de_1.0.0'

    try:
        opts, args = getopt.getopt(argv, 'hp:', ['model=', 'host=', 'port='])
        for opt, arg in opts:
            if opt == '-h':
                print('Usage: sa-rest.py --model=<model> --host=<host> --port=<port>')
                sys.exit()
            elif opt in ('-m', '--model'):
                model_id = arg
            elif opt in ('--host'):
                host = arg
            elif opt in ('-p', '--port'):
                port = int(arg)
        
        load_model_and_tokenizer(model_id)
        print('Staring server...')
        app.run(host, port)

    except getopt.GetoptError:
        print('Usage: sa-rest.py --model=<model> --host=<host> --port=<port>')
        sys.exit(2)
    

# If this is the main thread of execution first load the model and tokenizer and
# then start the server
# To kill the server if port is still in use after Ctrl+C: ps aux | grep sa-rest
if __name__ == '__main__':
    main(sys.argv[1:])
    
