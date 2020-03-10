"""
A deeplearning  chatbot implemented  by @GabrielGuan, based on tensorflow v1.13 and nltk.
This robot can help users to make query to dataset with natural language rather than complex pandas comments.
"""


import nltk
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import tflearn

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open('data.pickle', "rb") as f:
        words,labels,training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            print('wrds:', wrds)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            print('tag',intent["tag"])
            labels.append(intent["tag"])

    print('words: ', words)
    words =[stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    print(words)

    labels = sorted(labels)
    print(labels)
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        #print([stemmer.stem(w.lower()) for w in doc])

        wrds =[stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = np.array(training)
    output = np.array(output)
    #print(training)
    #print(output)
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output),f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower())  for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    print("Start talking with little Gabriel! ")
    """
    dataset=pd.read_excel('tmp002.xls')
    cov19 = pd.read_excel('nCov-19-China.xlsx')
    vocation = pd.read_excel('vocation.xlsx')
    """
    while True:
        inp = input("You: ")
        if inp.lower() =="quit":
            break
        results = model.predict([bag_of_words(inp,words)])
        results_index = np.argmax(results)
        tag = labels[results_index]
        print(np.max(results))
        if np.max(results) <0.7:
            print("I don't understand what are you talking, try to ask another question.")
        elif tag =='no_payment':
            print('no payment')
        else:
            for tg in data["intents"]:
                if  tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))



chat()
