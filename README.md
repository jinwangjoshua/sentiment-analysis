# Sentiment Prediction
Sentiment analysis of german texts. The machine learning model tries to predict whether a text is positive or negative.

The machine learning model was implemented with [Keras](https://keras.io) and trained for about 7 hours with about 30'000 German film reviews. The measured accuracy of the predictions is 81.82%. The model consists of an embedding layer as input, a hidden layer with 400 LSTM units and an output layer with 1 unit. The maximum length of an input text is 400 words.
For the word embedding the [pre-trained word vectors](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) of *FacebookResearch* were used.
Several texts can be posted to the REST service at once for analysis (see the examples below).

## Installation
The REST service was tested with [Python 3.6](https://www.python.org/) and [Tensorflow 1.5](https://tensorflow.org). The following Python packages must be installed:

```
pip3 install pip --upgrade
pip3 install tensorflow --upgrade
pip3 install keras --upgrade
pip3 install numpy  --upgrade
pip3 install pickle --upgrade
pip3 install flask --upgrade
```

## Starting the REST Service
To start the REST server in a terminal:

```
Usage: rest-server.py --host=<host> --port=<port>
```

### Example

```
python3 ./src/rest-server.py --port=5001
```

## Querying the REST Service
Queries use the following JSON format:

```
{"texts": [
  "Dieser Film ist vom Anfang bis am Ende spannend! Die Schauspieler sind super!",
  "Dieser Film ist vom Anfang bis am Ende langweilig! Die Schauspieler sind mässig bis schlecht!"
]}
```

### Example with Curl
Example of a curl call in a terminal:

```
curl -H "Content-Type: application/json" -X POST -d '{"texts": ["Dieser Film ist vom Anfang bis am Ende spannend! Die Schauspieler sind super!","Dieser Film ist vom Anfang bis am Ende langweilig! Die Schauspieler sind mässig bis schlecht!"]}' http://127.0.0.1:5000/predict
```

### Example Response

```
{
  "predictions": [
    {
      "probability": 0.988837718963623, 
      "text": "Dieser Film ist vom Anfang bis am Ende spannend! Die Schauspieler sind super!"
    }, 
    {
      "probability": 0.005242485553026199, 
      "text": "Dieser Film ist vom Anfang bis am Ende langweilig! Die Schauspieler sind mässig bis schlecht!"
    }
  ], 
  "success": true
}
```

A probability towards 0 means *negative*, one towards 1 means *positive*.
