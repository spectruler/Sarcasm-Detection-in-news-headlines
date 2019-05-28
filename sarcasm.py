import numpy as np #for numpy array
import pandas as pd #to read json
import json 
import spacy #nlp library 
import nltk #nlp library 

nltk.download("vader_lexicon") #for sentiment analysis 
nltk.download('averaged_perceptron_tagger') #for tags
nltk.download('punkt')# for punctuatuion

from nltk.sentiment.vader import SentimentIntensityAnalyzer # for sentiment analysis
from senticnet.senticnet import SenticNet #for opinion based and concept based sentiment analysis

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report #for measure performance
from sklearn.svm import LinearSVC, SVC 
from sklearn.model_selection import train_test_split

#####MULTITHREADING##########
#############################
import threading 
import time
import gc #garbage collector

class Sarcasm:

    def __init__(self,*args,**kwargs):
        # loading necessaries 
        self.nlp = spacy.load("en_core_web_sm")
        self.senti = SenticNet()
        self.sid = SentimentIntensityAnalyzer()
        #loading dataset
        self.df = pd.read_json("./Sarcasm_Headlines_Dataset.json",lines=True)
        self.df = self.df[:15000]
        self.df.drop(columns="article_link",inplace=True) #dropping unnessary attribute
        #storing nlp data in headlines variable 
        self.headlines = []
        self.uni_gram = set()
        self.uni_feature = []
        self.y_ = []
        for i in self.df['headline']:
            self.headlines.append(self.nlp(i))
        
    
    def w_score(self,w):
        """
        input: word
        Finding word score based on nltk's vader_lexicon sentiment analysis
        and Senticnet sentiment analysis
        """
        ss = self.sid.polarity_scores(w)['compound']
        try:
            sn = self.senti.polarity_intense(w)
            sn = float(sn)
            if ss == 0 :
                return sn
            else:
                return (sn+ss)/2
                
        except:
            #not found in sn find for only ss or concepts
            if ss != 0:
                return ss
            elif ss == 0: #find for the concepts
                return ss

    def sentimentScore(self,sent):
        """
        input: sentence
        Return if contradiction occurs 
        or not 
        """
        sum_pos_score = 0
        sum_neg_score = 0
        for w in sent:
            if w.lemma_ == '-PRON-':
                score = self.w_score(w.text)
            else:
                score = self.w_score(w.lemma_)
            if score > 0:
                sum_pos_score += score
            else:
                sum_neg_score += score
        if sum_pos_score > 0 and sum_neg_score < 0:
            return ("contradict",sum_pos_score,sum_neg_score)
        else:
            return ("anything",sum_pos_score,sum_neg_score)
        
    
    def coherence(self,s1,s2):
        '''
        Input sentence1, sentence2 using nlp
        Rule1:- Pronoun match feature - including reflexive, personal, and possessive pronouns.
        Rule2:- String match feature - ignore stop words
        Rule3:- Definite noun phrase - w2 starts with the word 'the'
        Rule4:- Demonstrative noun phrase feature - w2 starts with the "this", "that", "these" and "those"
        Rule5:- Both proper names features - w1 and w2 are both named entities
        '''
        # subject and object of s1 and s2
        sub1 = ""
        sub2 = ""
        obj1 = ""
        obj2 = ""
        
        for i in s1.noun_chunks:
            if i.root.dep_ == 'nsubj':
                sub1 = i.root
            if i.root.dep == 'pobj':
                obj1 = i.root
        for j in s2.noun_chunks:
            if j.root.dep_ == 'nsubj':
                if type(sub1) != type("") and sub1.pos_ == 'PRON' and j.root.pos_ == 'PRON':
                    if sub1.text.lower() == j.root.text.lower():
                        return "coherent"
                # rule 4:-
                
                if j[0].text.lower() == 'the':
                    return "coherent"
                if j[0].text.lower() in ['this','that','these','those']:
                    return "coherent"
            if j.root.dep_ == 'pobj':
                if type(obj1) != type("") and obj1.pos_ == 'PRON' and j.root.pos_ == 'PRON':
                    if obj1.text.lower() == j.root.text.lower():
                        return "coherent"
        return "Not coherent"
    
    def to_string_from_list(self,l):
        st = ""
        for i in l:
            st += i + ' '
        return st.rstrip()

    def n_gram_feature(self,text,n):
        """
        Input: headline in nlp
        Finding n grams of given text
        """
        one_list = []
        for tok in text:
            if not tok.is_punct:
                if tok.lemma_ !=  '-PRON-':
                    one_list.append(tok.lemma_)
                else:
                    one_list.append(tok.text)
        try:
            one_list.remove(' ')
        except:
            pass
        #convert it to n-gram
        _list = []
        for i,t in enumerate(one_list):
            if len(one_list[i:n+i]) >= n:
                _list.append(self.to_string_from_list(one_list[i:n+i]))
        return set(_list)

    def contradiction_feature(self,headline):
        '''
        Contradiction feature 
        input: nlp processed 
        '''
        #for single sentence headline
        if len(list(headline.sents)) == 1:
            if self.sentimentScore(headline)[0] == 'contradict':
                return(1,0)
            else:
                return(0,0)
        #for multisentence headline
        else:
            if self.sentimentScore(headline)[0] == 'contradict':
                sent = list(headline.sents)
                i = 0
                while i < len(sent)-1: 
                    # number of sentece
                    if self.coherence(sent[i],sent[i+1]) is not "coherent":
                        return(0,0)
                    i += 1
                return (0,1)
            
            else:
                return(0,0)

    def baseline3(self):
        '''
        Use of sentiment analysis + coherence
        '''
        predictions = []
        for i in self.headlines:
            get = self.contradiction_feature(i)
            if get == (1,0) or get == (0,1):
                predictions.append(1)
            else:
                predictions.append(0)
        return(confusion_matrix(self.df['is_sarcastic'],predictions),
        classification_report(self.df['is_sarcastic'],predictions),
        accuracy_score(self.df['is_sarcastic'],predictions))
    

    def baseline1(self):
        predictions = []
        for p in self.headlines:
            co,_,_ = self.sentimentScore(p)
            if(co == 'contradict'):
                predictions.append(1)
            else:
                predictions.append(0)
        return(confusion_matrix(self.df['is_sarcastic'],predictions),
        classification_report(self.df['is_sarcastic'],predictions),
        accuracy_score(self.df['is_sarcastic'],predictions))

    def uni_gram_features(self,start,end,n=1):
        self.uni_gram = list(self.uni_gram)
        self.uni_gram = sorted(self.uni_gram)
        index = start
        for p in self.headlines[start:end]:
            uni = [0 for i in range(len(self.uni_gram))]
            for i,j in enumerate(p):
                temp = [] #temp
                if len(p[i:n+i]) >= n:
                    for k in range(n):

                        if p[i+k].lemma_ != '-PRON-':
                            temp.append(p[i+k].lemma_)
                        else:
                            temp.append(p[i+k].text)

                    temp = self.to_string_from_list(temp)
                    if temp in self.uni_gram:
                        uni[self.uni_gram.index(temp)] = 1
            self.y_.append(self.df['is_sarcastic'][index])
            index += 1
            self.uni_feature.append(uni)

            
    def baseline2(self,n = 1):
        #unigram features
        self.uni_gram = set()
        self.uni_feature = []
        self.y_ = []
        for p in self.headlines:
            self.uni_gram = self.uni_gram.union(self.n_gram_feature(p,n))

        #now find 
        length = len(self.headlines)
        t1 = threading.Thread(target = self.uni_gram_features, name='t1',args=(0,int(length/4),n))
        t2 = threading.Thread(target = self.uni_gram_features, name='t2',args=(int(length/4),int(length/2),n))
        t3 = threading.Thread(target = self.uni_gram_features, name='t3',args=(int(length/2),int(3*length/4),n))
        t4 = threading.Thread(target = self.uni_gram_features, name='t4',args=(int(3*length/4),length,n))
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        st = time.time()
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        print(f'time taken: {time.time()-st}')
        X_train, X_test, y_train,y_test = train_test_split(self.uni_feature,self.y_,test_size=0.33,random_state=42)
        return self.findLINEARSVCResult(X_train,X_test,y_train,y_test)

    def findLINEARSVCResult(self,X_train,X_test,y_train,y_test):
        '''
         Training data using LinearSVC model
        '''
        svc_model = LinearSVC()
        svc_model.fit(X_train,y_train)
        predictions = svc_model.predict(X_test)
        return (confusion_matrix(y_test,predictions),
        classification_report(y_test,predictions),
        accuracy_score(y_test,predictions))