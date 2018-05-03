
# coding: utf-8

# In[5]:

import sys
import json
import os
from copy import deepcopy as copy
from collections import OrderedDict


# reading in the preprocessed article json file to dictionary

# code to import norweigen fasttext weight matrix from facebook, found here: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

# In[1]:


artclVec={}
CLASS={'arts-and-entertainment':0,'automotive':1,'business':2,'careers':3,'education':4,'family-and-parenting':5,'food-and-drink':6,
 'health-and-fitness':7,'hobbies-and-interests':8,'home-and-garden':9,'law-government-and-politics':10,'personal-finance':11,
     'pets':12,'real-estate':13,'religion-and-spirituality':14,'science':15,'shopping':16,'society':17,'sports':18,'style-and-fashion':19,
 'technology-and-computing':20,'travel':21,'uncategorized':22}

import numpy as np
def readNorwFastText(directory):
    indexDict={}
    matrix=np.array([])
    i=0
    j=0
    for line in open(directory):
        if i == 0:
            nrow=int(line.split()[0])
            ncol=int(line.rstrip().split()[1])
            matrix=np.zeros((nrow, ncol))
            i +=1
        try:
            indexDict[line.split()[0]]=j
            matrix[j,:]=np.array([float(n) for n in line.rstrip().split()[1:]])
            j +=1
        except:
            continue
    return {"index":indexDict, "matrix":matrix}

def addClassVector(classDict, wordVec, c_threshold=0):
    classVec=np.zeros((len(CLASS)))
    for c in classDict.keys():
        if c == 'uncategorized' or classDict[c] < c_threshold:
            classVec[CLASS[c]] = 0
        else:
            classVec[CLASS[c]]= classDict[c]
    return np.hstack((classVec, wordVec))

def findWord(word, wordDict, numMat):
    if len(word.split()) == 1:
        try:
            return numMat[wordDict[word],:]
        except:
            return 0
    else:
        if wordDict.get("".join(word.split())):
            return numMat[wordDict["".join(word.split())],:]
        elif wordDict.get("+".join(word.split())):
            return numMat[wordDict["+".join(word.split())],:]
        elif wordDict.get("_".join(word.split())):
            return numMat[wordDict["_".join(word.split())],:]
        elif wordDict.get("/".join(word.split())):
            return numMat[wordDict["/".join(word.split())],:]
        elif wordDict.get('\\'.join(word.split())):
            return numMat[wordDict['\\'.join(word.split())],:]
        else:
            sep=[]
            for w in word.split():
                vec=findWord(w, wordDict, numMat)
                if np.all(vec) != 0:
                    sep.append(vec)
            if not sep:
                return 0
            else:
                return np.mean(sep, axis=0)

def articles2vec(articles, wordDict, numMat, c_threshold=0,keyword=False, title=False):
    artclVec={}
    for doc in articles.keys():
        temp=[]
        for cpt in articles[doc]['concepts'].keys():
            val=findWord(cpt, wordDict, numMat)
            if np.all(val):
                temp.append(val*articles[doc]['concepts'][cpt])
        for cpt in articles[doc]['newsloc'].keys():
            val=findWord(cpt, wordDict, numMat)
            if np.all(val):
                temp.append(val*articles[doc]['newsloc'][cpt])
        for cpt in articles[doc]['person'].keys():
            val=findWord(cpt, wordDict, numMat)
            if np.all(val):
                temp.append(val*articles[doc]['person'][cpt])
        for cpt in articles[doc]['entities'].keys():
            val=findWord(cpt, wordDict, numMat)
            if np.all(val):
                temp.append(val*articles[doc]['entities'][cpt])
        
        if not temp:
            for cpt in articles[doc]['keywords']:
                val=findWord(cpt.lower(), wordDict, numMat)
                if np.all(val):
                    temp.append(val)
            for cpt in articles[doc]['title']:
                val=findWord(cpt.lower(), wordDict, numMat)
                if np.all(val):
                    temp.append(val)
        else:   
            if title:
                for cpt in articles[doc]['keywords']:
                    val=findWord(cpt.lower(), wordDict, numMat)
                    if np.all(val):
                        temp.append(val)
            if keyword:
                for cpt in articles[doc]['title']:
                    val=findWord(cpt.lower(), wordDict, numMat)
                    if np.all(val):
                        temp.append(val)
        if temp:
            temp=np.mean(temp, axis=0)
            artclVec[doc]=addClassVector(articles[doc]['classification'], temp, c_threshold=c_threshold)
    return artclVec


# Code to see how many types of articles are there that don't have actual classification based on url content:

# In[2]:


# Generates the prior of the classes for the keys extracted from urls using **classlessTypes** function

# In[211]:


def urlKeyword(key):
    ind=key.split('/')[3]
    if ind == 'nyheter' and '.ece' not in key.split('/')[4] and '.html' not in key.split('/')[4]:
        ind=key.split('/')[4]
    if ind == 'pluss' and 'nyheter' not in key.split('/')[4] and '20' not in key.split('/')[4]:
        ind=key.split('/')[4]
    if '100s' in key or 'sport' in key or 'fotball' in key:
        ind='sport'
    if ind == 'nyheter' or ind == 'pluss' or 'ece' in ind or 'html' in ind or ind.startswith('20'):
        ind='uncategorized'
    return ind

def classlessTypes(articles, num=None):
    classifications={}
    i=0
    for key in articles.keys():
        #if not articles[key]['classification']:
            #if classes in "uncategorized":
        ind=urlKeyword(key)
        try:
            classifications[ind] +=1
        except:
            classifications[ind]=1
    
    if num:
        classes=list(classifications.keys())
        for key in classes:
            if classifications[key] <= num:
                classifications['uncategorized'] += classifications[key]
                del classifications[key]
                
    return classifications

def generateClassPriorFromURLKey(noclass, articles):
    noclassPrior={}
    for key in noclass.keys():
        if key == 'uncategorized':
            continue
        noclassPrior[key]={}
        for art in articles.keys():
            if articles[art]['classification']:
                if key == urlKeyword(art):
                    for c in articles[art]['classification'].keys():
                        if noclassPrior[key].get(c):
                            noclassPrior[key][c] += articles[art]['classification'][c]
                        else:
                            noclassPrior[key][c]= articles[art]['classification'][c]
    for key in noclassPrior.keys():
        sum=0
        for c in noclassPrior[key].keys():
            sum +=noclassPrior[key][c]
        for c in noclassPrior[key].keys():
            noclassPrior[key][c]=noclassPrior[key][c]/sum
    return noclassPrior

def addClass(articles, classPrior, addNoise=True, loc=0, scale=0.01):
    for key in articles.keys():
        if not articles[key]['classification']:
            ind=urlKeyword(key)
            if ind == 'uncategorized' or not classPrior.get(ind):
                articles[key]['classification']['uncategorized']=1
            else:
                if addNoise:
                    articles[key]['classification']={c:classPrior[ind][c]+float(np.random.normal(size=1, loc=loc, scale=scale)) 
                                                 for c in classPrior[ind].keys()}
                else:
                    articles[key]['classification']={c:classPrior[ind][c] for c in classPrior[ind].keys()}
    return articles

# The activeTime for each article the user read throughout the 3 month is normalized by the largest activeTime of each user
def normalizeActiveTime(userevt):
    for usr in userevt:
        total=[]
        for evt in userevt[usr]:
            total.append(evt['activeTime'])
        for evt in userevt[usr]:
            evt['activeTime'] = float(evt['activeTime'])/max(total)
    return userevt
    


# In[246]:
def dateSpltArticles(artcl, last_days=7, train_test_gap=3): # last time point is 1490997602 in all data; split between 
    testArticlesTime=1490997602-last_days*24*60*60
    trainArticlesEndTime=1490997602-(last_days+train_test_gap)*24*60*60
    testArts=set()
    trainArts=set()
    for art in artcl:
        if artcl[art]['utime'] >= testArticlesTime:
            testArts.add(art)
        elif artcl[art]['utime'] <= trainArticlesEndTime:
            trainArts.add(art)
    testArts=testArts.difference(trainArts)
    coldartcl={}
    hotartcl={}
    for art in artcl:
        if art in testArts:
            coldartcl[art]=artcl[art]
        elif art in trainArts:
            hotartcl[art]=artcl[art]
    return coldartcl, hotartcl
    

# randomly split the articles into 2 dictionary, one for training the other for testing the coldstart problem
# This method is actually not coldstart as later learned, since it only used unseen targets but not unseen features
# the coldartcl dictionary here were later used as targets for prediction which was wrong.
def randomSplitArticle(artcl, coldstart=0.01):
    coldartcl={}
    hotartcl={}
    artcls=list(artcl.keys())
    for key in artcls:
        if np.random.uniform(high=1, low=0) <= coldstart:
            coldartcl[key]=artcl[key]
        else:
            hotartcl[key]=artcl[key]
    return coldartcl, hotartcl


### Still under progress and work:: generateTimeSeqData
# generate training data by time length: num_hours
# given a period of time use all articles read in that time to predict the final article at the end of that time period
# samples might have different length of timesteps depending on how much the user read in the time period
# once a time period surpass the allowed length, a new time period is started with the next article read
# by default the sampling starts with the first article the user read and begins the first time period there
# Thus by starting with later articles that were read, it is possible to generate different training samples
def generateTimeSeqData(artcl, usrevt, num_hours=48, minsize=1, batch=False, max_batchsize=30, firstind=[0]):
    Xdata=[]
    Tdata=[]
    time=num_hours*24*60*60
    for usr in usrevt:
        for ind in firstind:
            starttime=usrevt[usr][ind]['utime']
            events=[]
            i=ind
            for evt in usrevt[usr][ind:]:
                if np.any(artcl.get(evt['url'])):
                    if evt['utime'] - starttime <= time:
                        events.append(np.hstack((artcl[evt['url']], evt['activeTime'])))
                    else:
                        if len(events) > minsize:
                            Xdata.append(np.asarray(events[:-1]))
                            Tdata.append(events[-1][23:-1])
                        events=[]
                        starttime=evt['utime']
                    if i == len(usrevt[usr])-1 and len(events) > minsize:
                        Xdata.append(np.asarray(events[:-1]))
                        Tdata.append(events[-1][23:-1])
            
                        
    if batch:
        xtup=[(a,b) for a, b in zip(Xdata,Tdata)]
        xtup.sort(key=lambda x: len(x[0]))
        #xt, tt = (np.array(x) for x in zip(*xtup))
        Xdata=[]
        Tdata=[]
        curr_len=len(xtup[0][0])
        batchX=[] 
        batchT=[]
        i=0
        for x, t in xtup:
            if i == max_batchsize or len(x) > curr_len:
                Xdata.append(np.array(batchX))
                Tdata.append(np.array(batchT))
                curr_len=len(x)
                batchX, batchT=[], []
                i=0
            batchX.append(x.tolist())
            batchT.append(t.tolist())
            i +=1
        Xdata, Tdata=np.array(Xdata), np.array(Tdata)
        print('batched seq data length and shape is: '+str(len(Xdata))+str(Xdata[-1].shape)) 
        return {'X':Xdata, 'T':Tdata}
    else:
        Xdata, Tdata=[np.array([x]) for x in Xdata], [np.array([t]) for t in Tdata]
        arrX, arrT = np.empty(len(Xdata), dtype=object), np.empty(len(Tdata), dtype=object)
        arrX[:], arrT[:] = Xdata, Tdata
        print('fixed seq data length and shape is: '+str(len(Xdata))) 
        return {'X':arrX, 'T':arrT}

def generateFixedSeqData(artcl, usrevt, size=20, stride=1):
    Xdata=[]
    Tdata=[]
    for usr in usrevt:
        start=0
        events=[]
        for evt in usrevt[usr]:
            if np.any(artcl.get(evt['url'])):
                events.append(np.hstack((artcl[evt['url']], evt['activeTime'])))                
        while start+size < len(events)-1:
            #sequence=np.asarray(events[start:start+size])
            #sequence[:,-1]=sequence[:,-1]/np.sum(sequence[:,-1])
            Xdata.append(np.asarray(events[start:start+size]))
            Tdata.append(events[start+size+1][23:-1])
            start +=stride
            if start + size >= len(events)-1:
                finalstride=start+size-len(events)+2
                start=start-stride+finalstride
    print('fixed seq data length and shape is: '+str(len(Xdata))+str(Xdata[0].shape))
    return {'X':np.asarray(Xdata), 'T':np.asarray(Tdata)}

## Under progress:: generateColdStartTest
# can generate coldstart test data for fixed length sequencial data
# generating coldstart test data for time period sequencial data requires further coding
def generateColdStartTest(coldart, artVec, userevt, byhour=False, num_hours=48, size=20):
    coldart_names=set(coldart.keys())
    Xdata=[]
    Tdata=[]
    time=num_hours*24*60*60
    if byhour:
        for usr in userevt:
            currtime=userevt[usr][0]['utime']
            events=[]
            for evt in userevt[usr]:
                if np.any(artVec.get(evt['url'])):
                    if evt['utime'] == currtime:
                        events.append(np.hstack((artVec[evt['url']], evt['activeTime'])))
                        if evt['url'] in coldart_names:
                            pass
        
    else:
        j=0
        for usr in userevt:
            i=0
            events=[]
            for evt in userevt[usr]:
                if np.any(artVec.get(evt['url'])):
                    events.append(np.hstack((artVec[evt['url']], evt['activeTime'])))
                    if evt['url'] in coldart_names:
                        j +=1
                        if i == 0:
                            i +=1
                            continue
                        elif len(events) < size+1:
                            Xdata.append(np.asarray(events[:i]))
                            Tdata.append(np.asarray(events[i][23:-1]))
                        else:
                            Xdata.append(np.asarray(events[i-size:i]))
                            Tdata.append(np.asarray(events[i][23:-1]))
                    i +=1
#        print(j)
    return {'X':np.asarray(Xdata), 'T':np.asarray(Tdata)}
    
                
                
                    

def printOutDataSetStats(coldart, hotart, artcl, evtusr, last_days=5):
    cold=set(list(coldart.keys()))
    hot=set(list(hotart.keys()))
    i=0
    j=0
    k=0
    h=0
    for usr in evtusr:
        for evt in evtusr[usr]:
            if evt['utime'] <= 1490997602-last_days*24*60*60:
                h+=1
                continue
            else:
                i +=1
                if evt['url'] in cold:
                    j+=1
                elif evt['url'] in hot:
                    k+=1
    print('last '+str(last_days)+'days\ntotal reads: '+str(i)+'\ntotal reads in cold start data'+str(j)+'\ntotal reads in training data'+str(k))
    print('total cold-start articles: '+str(len(cold)))
    print('total training articles: '+str(len(hot)))
    print('total articles: '+str(len(artcl)))
    print('total number of reads in training: '+str(h))
# #### Main Function

# In[7]:


def main(args):
    article_fname=args[0]
    evt_fname=args[1]
    with open(article_fname) as f:
        articles=json.load(f)
    with open(evt_fname) as f:
        eventUser=json.load(f)
    fasttxt=readNorwFastText('./wiki.no.vec')
    noclass=classlessTypes(articles, num=10)
    classPriors= generateClassPriorFromURLKey(noclass, articles) 
    art1=copy(articles)
    art1=addClass(art1, classPriors, addNoise=True, loc=0, scale=0.001)
    eventUser=normalizeActiveTime(eventUser)
    coldart, hotart= dateSpltArticles(art1, last_days=5, train_test_gap=5)
    printOutDataSetStats(coldart, hotart, art1, eventUser)
    # the cold article vectors of the last 7 days
    coldartvec = articles2vec(coldart, fasttxt['index'], fasttxt['matrix'],keyword=True, title=True)
    hotartvec = articles2vec(hotart, fasttxt['index'], fasttxt['matrix'],keyword=True, title=True)
    
    time_batch=generateTimeSeqData(hotartvec, eventUser, batch=True, num_hours=3, minsize=1, firstind=[0])
    np.save('time_Xbatch.npy', time_batch['X'])
    np.save('time_Tbatch.npy', time_batch['T'])
    
    # generated unsorted time sequence data for non batch training, samples were fed 1 by 1 into model
#    time_1by1=generateTimeSeqData(hotartvec, eventUser, num_hours=12, minsize=1, firstind=[0])
#    np.save('time_X1by1.npy', time_1by1['X'])
#    np.save('time_T1by1.npy', time_1by1['T'])

    d_fixed=generateFixedSeqData(hotartvec, eventUser, size=30, stride=15)
    #cold_data_timed=generateColdStartTest(coldart, hotart, eventUser) # generated random cold start data which was wrong, 
    np.save('d30_Xfixed.npy', d_fixed['X'])
    np.save('d30_Tfixed.npy', d_fixed['T'])
    
    cold_fixed=generateFixedSeqData(coldartvec, eventUser,  size=30, stride=15)
    np.save('d30_Xcoldfixed.npy', cold_fixed['X'])
    np.save('d30_Tcoldfixed.npy', cold_fixed['T'])

    cold_time=generateTimeSeqData(coldartvec, eventUser, num_hours=3, minsize=1, firstind=[0])
    np.save('time_Xcold.npy', cold_time['X'])
    np.save('time_Tcold.npy', cold_time['T']) 

# In[40]:


if __name__ == "__main__":
    main(sys.argv[1:])

