from io import BytesIO
from datetime import datetime
from datetime import timedelta
from geopy import geocoders
import json
import requests

class Data(object):
    
    def __init__(self, cdr_id, auth, pwd):
        self.cdr_id = cdr_id
        self.auth = auth
        self.pwd = pwd
        
    def get_target_ad(self):
        query = json.dumps({
            "query": {
                "match": {
                    "_id": self.cdr_id
                }
            }
        })
        r = requests.get('https://cdr-es.istresearch.com:9200/memex-domains/escorts/_search?&pretty=true', auth=(self.auth, self.pwd), data=query)
        print r.status_code
        if r.status_code == 200 : 
            jout = json.loads(r.text)
            for akey in jout['hits']['hits']:
                #print akey
                if (akey.get('_source').get('extractions').get('title')) and (akey.get('_source').get('extractions').get('text')):
                    
                    #print '_id:', akey['_id']
                    
                    #if (akey.get('_source').get('extractions').get('name')):
                    #    names = akey['_source']['extractions']['name']['results']
                        #print 'names:', names
                    #else: names = []
                    
                    #if (akey.get('_source').get('extractions').get('weight')):
                    #    weight = akey['_source']['extractions']['weight']['results']
                    #    print 'weight:', weight
                    
                    title = akey['_source']['extractions']['title']['results'][0].encode('ascii', 'ignore')
                    #print 'title:', title
                        
                    text = akey['_source']['extractions']['text']['results'][0].encode('ascii', 'ignore')
                    #print 'text:', text
                    
                    postid = akey['_source']['extractions']['sid']['results'][0].encode('ascii', 'ignore')
                    #print 'Post ID:', postid
                        
                    extract_time = akey['_source']['crawl_data']['context']['timestamp']
                    #print 'extract time:', extract_time
        
        if "backpage.com" in title: title = title.replace('backpage.com','')
        if "escorts -" in title: title = title.replace('escorts -','')
        if "body rubs -" in title: title = title.replace('body rubs -','')
        if "strippers and strip clubs -" in title: title = title.replace('strippers and strip clubs -','')
        if "domination and BDSM services -" in title: title = title.replace('domination and BDSM services -','')
        if "transsexual escorts -" in title: title = title.replace('transsexual escorts -','')
        if "Adult web site directory -" in title: title = title.replace('male escorts -','')
        if "adult jobs -" in title: title = title.replace('adult jobs -','')
          
        return title, text, extract_time
    
    def get_comparison_set(self, extract_time):
        comparison_set = {}
        query = json.dumps({
            "query": {
                "range": {
                    "crawl_data.context.timestamp": {
                       "gte" : extract_time - (44640 * 60),
                       "lte" : extract_time
                    }
                }
            }
        })
        r = requests.get('https://cdr-es.istresearch.com:9200/memex-domains/escorts/_search?&pretty=true', auth=(self.auth, self.pwd), data=query)
        #print r.status_code
        if r.status_code == 200 : 
            jout = json.loads(r.text)
            for akey in jout['hits']['hits']:
                if (akey.get('_source').get('extractions').get('title')) and (akey.get('_source').get('extractions').get('text')) and (akey.get('_source').get('extractions').get('sid')):
                    
                    _id = akey['_id']
                    #print '_id:', akey['_id']
                    
                    #if (akey.get('_source').get('extractions').get('name')):
                    #    names = akey['_source']['extractions']['name']['results']
                        #print 'names:', names
                    #else: names = []
                    
                    title = akey['_source']['extractions']['title']['results'][0].encode('ascii', 'ignore')
                    #print 'title:', title
                        
                    text = akey['_source']['extractions']['text']['results'][0].encode('ascii', 'ignore')
                    #print 'text:', text
                    
                    postid = akey['_source']['extractions']['sid']['results'][0].encode('ascii', 'ignore')
                    #print 'Post ID:', postid
                        
                    extract_time = akey['_source']['crawl_data']['context']['timestamp']
                    #print 'extract time:', extract_time
                    
                    #data = (title, text, names)
                    if "backpage.com" in title: title = title.replace('backpage.com','')
                    if "escorts -" in title: title = title.replace('escorts -','')
                    if "body rubs -" in title: title = title.replace('body rubs -','')
                    if "strippers and strip clubs -" in title: title = title.replace('strippers and strip clubs -','')
                    if "domination and BDSM services -" in title: title = title.replace('domination and BDSM services -','')
                    if "transsexual escorts -" in title: title = title.replace('transsexual escorts -','')
                    if "Adult web site directory -" in title: title = title.replace('male escorts -','')
                    if "adult jobs -" in title: title = title.replace('adult jobs -','')
         
                    data = (title, text)
                    comparison_set[_id] = data
            
        return comparison_set
    

if __name__ == '__main__':

    #d = Data('19B14B8FC8D451DA77E9D5C7E82D0F9BADBAE36E26EB36515B3125FC5933466C')
    d = Data('F6B69F2CD3E2701373745DB9F86459AD438DE74A25CA1F4B9DACFCF078BBEEEC',auth,pwd)
    t_title, t_text, t_extract_time = d.get_target_ad()
    comparison_set = d.get_comparison_set(t_extract_time)
    for k,v in comparison_set.iteritems(): print k, v
    #print comparison_set
                                             
    
    
    

