import os
import os.path
import sys, getopt
import requests
import json
import pickle
import itertools

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
models = {}
tokenizers = {}

def load_remote_file(source_url, target_file):
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

def register(model_ids, detect_lang):
    global detect_lang_url
    global lang_registry

    detect_lang_url = detect_lang
    lang_registry = {}
    model_ids = [m.strip() for m in model_ids.split(',')]

    for model_id in model_ids:
        lang = model_id.split('_')[0]

        model_name = 'model_' + model_id + '.h5'
        model_file = os.path.join(DATA_DIR, model_name)
        model_url = REMOTE_DATA_URL + '/' + model_name

        # Load an register model
        load_remote_file(model_url, model_file)
        print('Loading model {0}'.format(model_file))
        model = load_model(model_file)

        # Load and register tokenizer
        tokenizer_name = 'tokenizer_' + model_id + '.pickle'
        tokenizer_file = os.path.join(DATA_DIR, tokenizer_name)
        tokenizer_url = REMOTE_DATA_URL + '/' + tokenizer_name

        load_remote_file(tokenizer_url, tokenizer_file)
        print('Loading tokenizer {0}'.format(tokenizer_file))
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)

        lang_registry[lang] = { 'model': model, 'tokenizer': tokenizer }

def prepare_texts(tokenizer, texts):
    """
    Create the input sequences and do the padding of the vector.

    Arguments:
        texts -- an array of texts to prepare for prediction.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded_texts = pad_sequences(sequences, maxlen=MAX_TEXT_LENGTH)
    return np.array(padded_texts)

def probability_to_string(probability):
    if probability < 1/2:
        return 'negativ'
    else:
        return 'positiv'

@app.route('/predict', methods=['POST'])
def predict():
    data = { 'success': False }

    if flask.request.method == 'POST':
        if flask.request.json['texts']:
            texts = flask.request.json['texts']
            if detect_lang_url:
                headers = {'content-type': 'application/json'}
                response = requests.post(detect_lang_url, headers=headers, json=flask.request.json)
                detected = response.json()
                # Group texts by language
                by_lang = {}
                for lang_response in detected['predictions']:
                    by_lang.setdefault(lang_response['lang']['label'], []).append(lang_response)
                
                for lang, values in by_lang.items():
                    if lang in lang_registry:
                        model = lang_registry[lang]['model']
                        tokenizer = lang_registry[lang]['tokenizer']
                        
                        # Get the texts as array for prediction
                        texts = [d['text'] for d in values]
                        prepared_texts = prepare_texts(tokenizer, texts)
                        predictions = model.predict(prepared_texts)

                        # Mixin the results
                        for p, v in zip(predictions, values):
                            p = float(p[0])
                            v['sentiment'] = {
                                'label': probability_to_string(p),
                                'probability': p
                            }

                # Remove language grouping
                data['predictions'] = list(itertools.chain.from_iterable(by_lang.values()))
                data["success"] = True
                return flask.jsonify(data)

            else:
                lang = next (iter (lang_registry.keys()))
                model = lang_registry[lang]['model']
                tokenizer = lang_registry[lang]['tokenizer']
                
                prepared_texts = prepare_texts(tokenizer, texts)
                predictions = model.predict(prepared_texts)
                data["predictions"] = []

                for p, t in zip(predictions, texts):
                    p = float(p[0])
                    pred =  {
                                'lang': lang,
                                'sentiment': {
                                    'label': probability_to_string(p),
                                    'probability': p
                                },
                                'text': t
                            }

                    data['predictions'].append(pred)

                data['success'] = True
                return flask.jsonify(data);

def main(argv):
    host = None
    port = None
    model_ids = 'en_1.0.0'
    detect_lang = None

    try:
        opts, args = getopt.getopt(argv, 'hp:', ['models=', 'host=', 'port=', 'detect_lang='])
        for opt, arg in opts:
            if opt == '-h':
                print('Usage: sa-rest.py --models=<models> --host=<host> --port=<port> --detect_lang=<url>')
                sys.exit()
            elif opt in ('-m', '--models'):
                model_ids = arg
            elif opt in ('--host'):
                host = arg
            elif opt in ('-p', '--port'):
                port = int(arg)
            elif opt in ('-dlu', '--detect_lang'):
                detect_lang = arg
        
        register(model_ids, detect_lang)
        print('Staring server...')
        app.run(host, port)

    except getopt.GetoptError:
        print('Usage: sa-rest.py --models=<models> --host=<host> --port=<port> --detect_lang=<url>')
        sys.exit(2)
    

# If this is the main thread of execution first load the model and tokenizer and
# then start the server
# To kill the server if port is still in use after Ctrl+C: ps aux | grep sa-rest
if __name__ == '__main__':
    main(sys.argv[1:])
    
