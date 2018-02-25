# Sentiment Analysis

Sentiment analysis of german texts. The machine learning model predicts whether the sentiment of a text is positive or negative.

The machine learning model was implemented with [Keras](https://keras.io) and trained with about 30'000 German film reviews. The measured accuracy of the predictions is 81.82%. The model consists of an embedding layer as input, a hidden layer with 400 LSTM units and an output layer with 1 unit. The maximum length of an input text is 400 words. For the word embedding the [pre-trained word vectors](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) of *FacebookResearch* were used.

Several texts can be posted to the REST service at once for analysis (see the examples below).

## Scope of Application

The model was trained with film reviews in German language. We also tested the model at random with German product reviews and reader comments. The results were generally promising (but need further investigation).

## Installation

The REST service was tested with [Python 3.6](https://www.python.org), [Keras 2.1](https://keras.io) and [Tensorflow 1.5](https://tensorflow.org). Run the following command to install the dependencies:

```bash
pip3 install -r ./requirements.txt
```

## Starting the REST Service

To start the REST server in a terminal:

```bash
sa-rest.py -h
Usage: sa-rest.py --model=<model> --host=<host> --port=<port>
```

model: id composed of the language and the version (examples: de_1.0.0, en_1.0.0) of the model to load.

_Note_: The first time you start the service, the model (including the trained weights) and the tokenizer are loaded from ipublia's website (using https). They are stored in the directory ~/.ipublia/data/sentiment-analysis.

### Example

```bash
python3 ./src/sa-rest.py --model=de_1.0.0 --host=127.0.0.1 --port=5000
```

## Querying the REST Service

Queries use the following JSON format:

```json
{
    "texts": [
        "Dieser Film ist vom Anfang bis am Ende spannend! Die Schauspieler sind super!",
        "Dieser Film ist vom Anfang bis am Ende langweilig! Die Schauspieler sind mässig bis schlecht!"
    ]
}
```

### Example Call with Curl

Example of a curl call in a terminal:

```bash
curl -H "Content-Type: application/json" -X POST -d '{"texts": ["Dieser Film ist vom Anfang bis am Ende spannend! Die Schauspieler sind super!","Dieser Film ist vom Anfang bis am Ende langweilig! Die Schauspieler sind mässig bis schlecht!"]}' http://127.0.0.1:5000/predict
```

The answer will look something like this:

```json
{
    "predictions": [
    {
        "prediction": 0.988837718963623,
        "sentiment": "positiv",
        "text": "Dieser Film ist vom Anfang bis am Ende spannend! Die Schauspieler sind super!"
    },
    {
        "prediction": 0.005242485553026199,
        "sentiment": "negativ",
        "text": "Dieser Film ist vom Anfang bis am Ende langweilig! Die Schauspieler sind mässig bis schlecht!"
    }
  ],
  "success": true
}
```

A probability towards 0 means *negative*, one towards 1 means *positive* sentiment.
