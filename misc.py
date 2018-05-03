import json
import sys
import logging

logging.basicConfig(filename="misc.log", level=logging.DEBUG) 

def filterbymonth(month, userevt):
    if month == 'jan' or int(month) == 1:
        start, end = (1483225200, 1485903602)
    elif month == 'feb' or int(month) == 2:
        start, end = (1485903602, 1488322802)
    elif month == 'mar' or int(month) == 3:
        start, end = (1488322802, 1490997602)
    else:
        assert type(month) == 'tuple', 'must supply tuple or month name is argument for month'
        start, end=month
    for user in userevt.keys():
        for event in userevt[user]:
            if int(event['utime']) <= start:
                continue
            if int(event['utime']) <= end and int(event['utime']) > start:
                continue
            else:
                del userevt[user][user.index(event):]
                break
    return userevt

def filterbynumevt(num,userevt):
    newdata={}
    for user in userevt.keys():
        if len(userevt[user]) >= int(num):
            newdata[user] = userevt[user]
    return newdata


def filterbyactiveTime(time, userevt):
    time=int(time)
    for user in userevt.keys():
        newevt=[]
        logging.debug(str(len(userevt[user])))
        for event in userevt[user]:
            if int(event['activeTime']) > time:
                newevt.append(event)
        userevt[user]=newevt
        logging.debug(str(len(userevt[user])))
    return userevt

def filterbyarticle(artcl, userevt):
    for user in userevt.keys():
        newevt=[]
        for event in userevt[user]:
            if artcl.get(event['url']):
                newevt.append(event)
        userevt[user]=newevt
    return userevt

def urlonly(userevt):
    for user in userevt.keys():
        for event in userevt[user]:
            del event['classification']
            del event['concepts']
            del event['entities']
            del event['keywords']
            del event['newsloc']
            del event['person']
            del event['title']
    return userevt


def main(arg):
    filename=arg[0]
    condition=arg[1]
    with open(filename) as f:
        userevent=json.load(f)
    
    if 'month' in condition:
        userevent=filterbymonth(condition.split('=')[1], userevent)
        cond=condition.split('=')[0]+condition.split('=')[1]
    elif 'num' in condition:
        userevent=filterbynumevt(condition.split('=')[1], userevent) 
        cond=condition.split('=')[0]+condition.split('=')[1]
    elif 'activeTime' in condition:
        userevent=filterbyactiveTime(condition.split('=')[1], userevent)
        cond=condition.split('=')[0]+condition.split('=')[1]
    elif 'url_only' in condition:
        userevent=urlonly(userevent)
        cond='url_only'
    elif 'article' in condition:
        with open(condition.split('=')[1]) as f:
            article=json.load(f)
        userevent=filterbyarticle(article, userevent)
        cond=condition.split('=')[1].split('.')[0]
    with open(cond+"_"+filename, 'w') as f:
        json.dump(userevent, f)

if __name__=='__main__':
    main(sys.argv[1:])
