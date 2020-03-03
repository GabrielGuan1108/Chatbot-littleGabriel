import tensorflow as tf
import random
import json

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs = []

for intent in data['intents']:
    for patern in intent[patterns]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

    if intent['tag'] not in labels:
        labels.append(intent['tag'])
