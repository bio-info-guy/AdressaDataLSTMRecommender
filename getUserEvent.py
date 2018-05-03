
# coding: utf-8

# In[85]:


import json
from urllib.parse import unquote_plus
import pickle
import os
import time as T
import logging
from collections import Counter
import sys


def removeAllapostrophe(string):
    string=string.replace('.','')
    string=string.replace(',','')
    string=string.replace('?','')
    string=string.replace('"','')
    string=string.replace('/','')
    string=string.replace('\\','')
    string=string.replace('`','')
    string=string.replace(';','')
    string=string.replace('!','')
    string=string.replace(':',' ')
    string=string.replace("'",'')
    string=string.replace("»",'')
    string=string.replace("#",'')
    string=string.replace("«",'')
    string=string.replace("%",'')
    string=string.replace("+",'')
    string=string.replace("//",'') 
    return string


# In[57]:



def getAdressaSubUser(filename, subusers):
    for line in open(filename):
        data=line.rstrip()
        if "/pluss/" in line:
            d=json.loads(data)
            subusers.add(d['userId'])
    return subusers


# In[86]:


def simplifyAdressa(filename, userEvent, subusers, session):
    for line in open(filename):
        data=line.rstrip()
        d=json.loads(data)
        if d['userId'] in subusers:
            newd={}
            loc=[]
            utime=''
            time=''
            activeTime=0
            if 'time' in d.keys():
                utime=int(d['time'])
                time=T.strftime('%Y-%m-%d %H:%M:%S', T.localtime(utime))
            if 'activeTime' in d.keys():
                activeTime=d['activeTime']
            if session.get(d['userId']) == None:
                session[d['userId']]=0
            else:
                if d['sessionStart']:
                    session[d['userId']]+=1
                    
            if d['url'] == 'http://adressa.no':
                continue
            elif 'ece' not in d['url'] and 'html' not in d['url']:
                continue
            if 'country' in d.keys():
                loc.append(d['country'])
                if 'city' in d.keys():
                    loc.append(d['city'])
                    if 'region' in d.keys():
                        loc.append(d['region'])
            else:
                loc='NA'
            newd['session']=session[d['userId']]
            newd['utime']=utime
            newd['time']=time
            newd['activeTime']=activeTime
            newd['location']="_".join(loc)
            newd['url']=d['url']
            try:
                userEvent[d['userId']].append(newd)
            except:
                userEvent[d['userId']]=[newd]
    return userEvent, session


# In[ ]:

logging.basicConfig(filename="test.log", level=logging.DEBUG)
def main(args):
    if len(args) > 0:
        with open(args[0],'rb') as f:
            subusr=pickle.load(f)
    else:
        subusr=set()
        for filename in os.listdir('./'):
            if filename.startswith('2017'):
                logging.debug(filename)
                subusr=getAdressaSubUser(filename, subusr)
        with open('subusers.pkl', 'wb') as f:
            pickle.dump(subusr, f)
    logging.debug("total amount of subscribing users: "+str(len(subusr)))
    files=[]
    userevt={}
    sessionNum={}
    for filename in os.listdir('./'):
        if filename.startswith('2017'):
            files.append(filename)
    files.sort()
    for filename in files:
            logging.debug(filename)
            userevt, sessionNum=simplifyAdressa(filename, userevt, subusr, sessionNum)

    
    with open('eventUser.pkl','w') as f:
        json.dump(userevt, f)

if __name__ == '__main__':
    main(sys.argv[1:])
