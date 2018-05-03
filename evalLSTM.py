import sys
import json
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt
from keras.models import load_model
from preprocessMisc import readNorwFastText, classlessTypes, generateClassPriorFromURLKey, addClass, normalizeActiveTime, dateSpltArticles




def makeHRplot():
    pass

def makeMRRplot():
    pass
    
def makeRecallplot():
    pass


def sampleTimeSeqData(artvec, art, evtusr):
    for 




def main(args):
    model_type=args[0]
    model_name=args[1]
    article_fname=args[2]
    evt_fname=args[3]
    with open(article_fname) as f:
        articles=json.load(f)
    with open(evt_fname) as f:
        eventUser=json.load(f)
    fasttxt=readNorwFastText('./wiki.no.vec')
    classPriors= generateClassPriorFromURLKey(classlessTypes(articles, num=10), articles) 
    art1=addClass(articles, classPriors, addNoise=True, loc=0, scale=0.001)
    eventUser=normalizeActiveTime(eventUser)
    coldart, hotart= dateSpltArticles(art1, last_days=5, train_test_gap=5)
    if model_type == 'time':
        
