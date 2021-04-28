import os,sys,re,csv
import numpy as np
import scipy
import math
import random
import nltk
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


tweets=[]
emojis=[]
tweets_dict={}
predictions=[]

def load_data(filename):
    #load date into two list, tweets and emojis
    global tweets_dict
    global tweets
    global emojis
    f = open(filename, "r", encoding="utf8")
    rows =f.read().split("\n")
    for each in rows:
        row=each.split("\t")
        if len(row)==2:
            tweet=row[0]
            tweets.append(tweet)
            emoji=row[1]
            emojis.append(emoji)
            tweets_dict[tweet]=emoji

def calculate_most_frequent_emoji(emojis):
    emoji_dict={}
    for emoji in emojis:
        if emoji not in emoji_dict:
            emoji_dict[emoji]=1
        else:
            emoji_dict[emoji]+=1
    emoji_list=sorted(emoji_dict.items(), key=lambda d: d[1],reverse=True)
    print(emoji_list)
    for each in emoji_list:
        print(each)
    return emoji_list[0][0]

def predict(filename,emoji):
    #print(emoji)
    global predictions
    origins=[]
    f = open(filename, "r", encoding="utf8")
    rows =f.read().split("\n")
    #print(len(rows))
    for each in rows:
        row=each.split("\t")
        if len(row)==2:
            origins.append(row[1])
            predictions.append(emoji)
    print("f1_score:",f1_score(origins, predictions,average='micro'))
    print("precision:",precision_score(origins, predictions,average='micro'))
    print("recall:",recall_score(origins, predictions,average='micro'))

load_data("20_train")
most_frequent_emoji=calculate_most_frequent_emoji(emojis)
#load_data("20_validation")
#most_frequent_emoji=calculate_most_frequent_emoji(emojis)
predict("20_test",most_frequent_emoji)
