# Sentiment Analysis

Multilingual sentiment analysis for English, German, French and Italian. The machine learning model predicts whether the sentiment of a text is positive or negative.

The machine learning model was implemented with [Keras](https://keras.io) and trained with 50'000 english, 30'000 german, 50'000 french and 30'000 italian film reviews. The models are tested with [TensorFlow](https://www.tensorflow.org/) as backend.

## Accuracy of the Predictions

The measured accuracy of the predictions is 88.6% for English, 81.3% for German, 84.4% for French and 81.6% for Italian.

The model consists of an embedding layer as input, a hidden layer with 8 LSTM units and an output layer with 1 unit. The maximum length of an input text is 400 words. For the word embedding the [pre-trained word vectors](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) of *FacebookResearch* were used.

## Scope of Application

The model was trained with film reviews in English, German, French and Italian. We also tested the model at random with product reviews and reader comments in those languages. The results were generally good. However, we recommend testing when using the models in new domains.

Several texts can be posted to the REST service at once for analysis. You can start the sentiment analysis service to access a [language detection](https://github.com/ipublia/language-detection) service. The service then first determines the language of the texts and then passes them on to the corresponding sentiment analysis model (see the examples below).

## Installation

The REST service was tested with [Python 3.6](https://www.python.org), [Keras 2.1](https://keras.io) and [Tensorflow 1.5](https://tensorflow.org). Run the following command to install the dependencies:

```bash
pip3 install -r ./requirements.txt
```

## Starting the REST Service

To start the REST service in a terminal:

```bash
sa-rest.py -h
Usage: sa-rest.py --model=<model> --host=<host> --port=<port>
```

model: id composed of the language and the version (examples: en_1.0.1, de_1.0.1, fr_1.0.1, it_1.0.1) of the model to load. Per default the newest versions of all models are loaded.

_Note_: The first time you start the service, the model (including the trained weights) and the tokenizer are loaded from ipublia's website (using https). They are stored in the directory ~/.ipublia/data/sentiment-analysis.

### Examples

Loads all language models in the newest version:

```bash
python3 ./src/sa-rest.py --host=127.0.0.1 --port=5000
```

For French:

```bash
python3 ./src/sa-rest.py --model=fr_1.0.1 --host=127.0.0.1 --port=5000
```

For German:

```bash
python3 ./src/sa-rest.py --model=en_1.0.1 --host=127.0.0.1 --port=5000
```

## Querying the REST Service

Queries use the following JSON format:

```json
{
  "texts": [
    "This film is exciting from the beginning to the end! The actors are really good!",
    "This movie is boring from the beginning to the end! The actors are moderate to bad!"
  ]
}
```

### Example Call with Curl

Example of a curl call in a terminal:

```bash
curl -H "Content-Type: application/json" -X POST -d '{"texts": ["This film is exciting from the beginning to the end! The actors are really good!","This movie is boring from the beginning to the end! The actors are moderate to bad!"]}' http://127.0.0.1:5000/predict
```

The answer will look something like this:

```json
{
  "predictions": [
    {
      "lang": "en", 
      "sentiment": {
        "label": "positiv", 
        "probability": 0.9012099504470825
      }, 
      "text": "This film is exciting from the beginning to the end! The actors are really good!"
    }, 
    {
      "lang": "en", 
      "sentiment": {
        "label": "negativ", 
        "probability": 0.01829037070274353
      }, 
      "text": "This movie is boring from the beginning to the end! The actors are moderate to bad!"
    }
  ], 
  "success": true
}

```

A probability towards 0 means *negative*, one towards 1 means *positive* sentiment.

## Starting the REST Service with Language Detection

You can start the *sentiment analysis service* to access a [language detection](https://github.com/ipublia/language-detection) service. The service then first determines the language of the texts and then passes them on to the corresponding sentiment analysis model.

To start the REST service with language detection in a terminal:

```bash
sa-rest.py -h
Usage: sa-rest.py --model=<model> --host=<host> --port=<port> --detect_lang=<url>
```

model: id composed of the language and the version (examples: en_1.0.1, de_1.0.1, fr_1.0.1, it_1.0.1) of the model to load. Per default the newest versions of all models are loaded.

detect_lang: url of the language detection service (example: http://127.0.0.1:5001/predict).

### Example

First, start a language detection service:

```bash
python3 ./src/ld-rest.py --host=127.0.0.1 --port=5001
```

Then start the sentiment analysis service with a reference to the language detection service:

```bash
python3 ./src/sa-rest.py --host=127.0.0.1 --port=5000 --detect_lang=http://127.0.0.1:5001/predict
```

### Example Call with Curl

Example of a curl call in a terminal (note the third text in English):

```bash
curl -H "Content-Type: application/json" -X POST -d '{"texts": ["I found this movie really hard to sit through, my attention kept wandering off the tv.","Dieser Film ist vom Anfang bis am Ende spannend! Die Schauspieler sind wirklich gut!","Dieser Film ist vom Anfang bis am Ende langweilig! Die Schauspieler sind mässig bis schlecht!","J\u0027aime ce film. Les acteurs jouent vraiment bien!","Non mi piace affatto questo film. Gli attori sono cattivi e la storia è noiosa!"]}' http://127.0.0.1:5000/predict
```

The answer will look like this:

```json
{
  "predictions": [
    {
      "lang": {
        "label": "en", 
        "probability": {
          "de": 0.15291942656040192, 
          "en": 0.386442095041275, 
          "fr": 0.1524139940738678, 
          "it": 0.15353836119174957, 
          "rm": 0.1546860933303833
        }
      }, 
      "sentiment": {
        "label": "negativ", 
        "probability": 0.44230401515960693
      }, 
      "text": "I found this movie really hard to sit through, my attention kept wandering off the tv."
    }, 
    {
      "lang": {
        "label": "de", 
        "probability": {
          "de": 0.4036977291107178, 
          "en": 0.1490136682987213, 
          "fr": 0.1489626169204712, 
          "it": 0.14897985756397247, 
          "rm": 0.14934605360031128
        }
      }, 
      "sentiment": {
        "label": "positiv", 
        "probability": 0.9132363200187683
      }, 
      "text": "Dieser Film ist vom Anfang bis am Ende spannend! Die Schauspieler sind wirklich gut!"
    }, 
    {
      "lang": {
        "label": "de", 
        "probability": {
          "de": 0.40277570486068726, 
          "en": 0.1491391956806183, 
          "fr": 0.14907844364643097, 
          "it": 0.1491205096244812, 
          "rm": 0.1498860865831375
        }
      }, 
      "sentiment": {
        "label": "negativ", 
        "probability": 0.0641002506017685
      }, 
      "text": "Dieser Film ist vom Anfang bis am Ende langweilig! Die Schauspieler sind mässig bis schlecht!"
    }, 
    {
      "lang": {
        "label": "fr", 
        "probability": {
          "de": 0.14898933470249176, 
          "en": 0.1492728441953659, 
          "fr": 0.40348654985427856, 
          "it": 0.14923687279224396, 
          "rm": 0.14901447296142578
        }
      }, 
      "sentiment": {
        "label": "positiv", 
        "probability": 0.6126847267150879
      }, 
      "text": "J'aime ce film. Les acteurs jouent vraiment bien!"
    }, 
    {
      "lang": {
        "label": "it", 
        "probability": {
          "de": 0.15224331617355347, 
          "en": 0.15012496709823608, 
          "fr": 0.15062405169010162, 
          "it": 0.39602142572402954, 
          "rm": 0.15098623931407928
        }
      }, 
      "sentiment": {
        "label": "negativ", 
        "probability": 0.20823518931865692
      }, 
      "text": "Non mi piace affatto questo film. Gli attori sono cattivi e la storia è noiosa!"
    }
  ], 
  "success": true
}
```