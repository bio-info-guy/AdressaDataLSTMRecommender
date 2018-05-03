import json
from urllib.parse import unquote_plus
import pickle
import os
import time as T
import logging
from collections import Counter
import sys
import datetime
logging.basicConfig(filename="test1.log", level=logging.DEBUG) 
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




def extractArticles(filename, articles):
    articlenum=set()
    for line in open(filename):
        data=line.rstrip()
        d=json.loads(data)
        if d['url'] == 'http://adressa.no':
            continue                                                                                                                                                                                                                     
        elif 'ece' not in d['url'] and 'html' not in d['url']:
            continue
        newd={}
        articlenum.add(d['url'])
        if articles.get(d['url']):
            oldd=articles[d['url']]
            keywords=oldd['keywords']
            concepts=oldd['concepts'] 
            entities=oldd['entities'] 
            classification=oldd['classification']
            person=oldd['person']
            utime=oldd['utime']
            time=oldd['time']
            titlekeys=oldd['title']
            location=oldd['newsloc']
        else:
            keywords=[]
            concepts={}
            entities={}
            location={}
            classification={}
            person={}
            utime=''
            time=''
            urltitle=unquote_plus(d['url'].split("/")[-1])
            titlekeys=urltitle.split("-")[:-1] 

        if 'publishtime' in d.keys():
            time=d['publishtime']
            b=tuple(time.replace("T", "-").replace(":","-").split(".")[0].split("-"))
            b=[int(i) for i in b]
            utime=int(T.mktime(datetime.datetime(b[0], b[1], b[2], b[3], b[4], b[5]).timetuple()))

        elif 'time' in d.keys() and not utime:
            utime=int(d['time'])
            time=T.strftime('%Y-%m-%d %H:%M:%S', T.localtime(utime)) 
            
        if 'title' in d.keys():
            title=unquote_plus(d['title'])
            title=removeAllapostrophe(title)
            titlekeys=title.split(' ')
            
        if 'keywords' in d.keys():
            keywords=d['keywords'].split(',')
        if 'profile' in d.keys():
            for prof in d['profile']:
                for grp in prof['groups']:
                    item=unquote_plus(prof['item'])
                    if grp['group'] =='concept':
                        concepts[item]=grp['weight']
                    elif grp['group'] =='entities':
                        entities[item]=grp['weight']
                    elif grp['group'] =='location':
                        location[item]=grp['weight']
                    elif grp['group'] =='classification':
                        classification[item]=grp['weight']
                    elif grp['group'] =='person':
                        person[item]=grp['weight']
        newd['utime']=utime
        newd['time']=time
        newd['concepts']=concepts
        newd['entities']=entities
        newd['newsloc']=location
        newd['classification']=classification
        newd['person']=person
        newd['title']=titlekeys
        newd['keywords']=keywords
        articles[d['url']]=newd
#        print(newd)
#        logging.debug(len(articles)) 
    logging.debug(len(articlenum))
    return articles


# In[ ]:

def main():
    
    artcl={}
    for filename in os.listdir('./'):
        if filename.startswith('2017'):
            logging.debug(filename)
            artcl=extractArticles(filename, artcl)
            logging.debug(len(artcl))
    articles=list(artcl.keys())
    with open('articles_unfiltered.json','w') as f:
        json.dump(artcl, f)
    """
    with open('articles_unfiltered.json') as f:
        artcl=json.load(f)
    articles=list(artcl.keys())
    """
    
    for art in articles:
        empty=False
        for key in ['title']:
            if artcl[art][key]:
                continue
            elif artcl[art]['keywords']:
                continue
            elif artcl[art]['concepts']:
                continue
            elif artcl[art]['person']:
                continue
            elif artcl[art]['entities']:
                continue
            elif artcl[art]['newsloc']:
                continue
            else:
                empty=True
        if empty:
            del artcl[art]
    logging.debug(len(artcl)) 
    with open('articles_filtered.json','w') as f:
        json.dump(artcl, f)

if __name__ == '__main__':
    main()
