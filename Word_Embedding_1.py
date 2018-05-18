import xml.etree.cElementTree as et
from sklearn.utils import shuffle
import re
import numpy as np
import pandas as pd

def get_stop_words():
    file = open("data_1/stopword.txt", "r")
    stop = []
    while True :
        line = file. readline()
        if (line == ""):
            break
        word = line.split()[0]
        stop.append(word)
    return stop

def tokenizer(text, stop):
    #text = str(text2)
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text)
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^|][0-9]+', ' ', text)
    text = re.sub(r'[\s!/,\\.?¡¿"“”:/();]+', ' ', text) #”
    text = re.sub(r'[W]+', ' ', text)
    text = re.sub(r'[\s]+', ' ', text.lower())
    text = re.sub(r'\b(gg+\b)+', 'por', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\b(d*xd+x*[xd]*\b|\ba*ha+h[ha]*\b|\ba*ja+j*[ja]*|o?l+o+l+[ol])\b', 'ja', text)
    text = re.sub('^\s', '', text)
    text = re.sub('@([a-z0-9_]+)', '@user', text)
    text = re.sub(r'\b(x\b)+', 'por', text)
    text = re.sub(r'\b(q\b)+|\b(k\b)+|\b(ke\b)+|\b(qe\b)+|\b(khe\b)+|\b(kha\b)+', 'que', text)
    text = re.sub(r'\b(xk\b)+|\b(xq\b)+', 'porque', text)
    text = re.sub(r'\b(xd\b)+|\b(xq\b)+', 'porque', text)
    text = re.sub(r'\b(ai+uda\b)+', 'ayuda', text)
    text = re.sub(r'\b(hostia\b)+', 'ostia', text)
    text = re.sub(r'\b(d\b)+', 'de', text)
    text = re.sub(r'\b(d\b)+', 'de', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\b-\b', ' ', text)
    text = re.findall(r'[^\s!,.?¡¿"“”:;]+', text)
    text = [w for w in text if w not in stop]
    return text

def getvalueofnode(node):
    """ return node text or None """
    return node.text if node is not None else None

def loadData_2_in(x_dir, g_dir):
    parsedXML = et.parse( x_dir )
    dfcols = ['content', 'polarity']
    df_xml = pd.DataFrame(columns=dfcols)

    for node in parsedXML.getroot():
        content = node.find('content')
        polarity = node.find('sentiment/polarity/value')
        if getvalueofnode(polarity) is None:
            polarity = node.find('sentiments/polarity/value')
        if getvalueofnode(polarity) != "NONE":
            df_xml = df_xml.append(
                pd.Series([getvalueofnode(content), getvalueofnode(polarity)], index=dfcols), ignore_index=True)

    parsedXML = et.parse( g_dir )
    for node in parsedXML.getroot():
        content = node.find('content')
        polarity = node.find('sentiment/polarity/value')
        if getvalueofnode(polarity) is None:
            polarity = node.find('sentiments/polarity/value')
        #if (getvalueofnode(content) is not None) * (getvalueofnode(polarity) is not None) * (getvalueofnode(polarity) != "NONE"):
        if (getvalueofnode(content) is not None) * (getvalueofnode(polarity) is not None):
            df_xml = df_xml.append(
                pd.Series([getvalueofnode(content), getvalueofnode(polarity)], index=dfcols), ignore_index=True)
    return df_xml

def loadData_1_in(x_dir):
    parsedXML = et.parse( x_dir )
    dfcols = ['content', 'polarity']
    df_xml = pd.DataFrame(columns=dfcols)

    for node in parsedXML.getroot():
        content = node.find('content')
        polarity = node.find('sentiment/polarity/value')
        if getvalueofnode(polarity) is None:
            polarity = node.find('sentiments/polarity/value')
        if getvalueofnode(polarity) != "NONE":
            df_xml = df_xml.append(
                pd.Series([getvalueofnode(content), getvalueofnode(polarity)], index=dfcols), ignore_index=True)
    return df_xml

def loadDataTesting(x_dir, y_dir):
    parsedXML = et.parse( x_dir )
    dfcols = ['content', 'polarity']
    df_xml = pd.DataFrame(columns=dfcols)

    file = open(y_dir, "r")
    answer = {}
    while True :
        line = file. readline()
        if (line == ""):
            break
        tweetid, polarity = line.split()
        answer[str(tweetid)] = polarity
        #print(polarity)

    for node in parsedXML.getroot():
        content = node.find('content')
        tweetid = node.find('tweetid')
        polarity = answer.get(str(getvalueofnode(tweetid)))
        if polarity != "NONE":
            df_xml = df_xml.append(
                pd.Series([getvalueofnode(content), polarity], index=dfcols), ignore_index=True)
    return df_xml

def loadChannels():
    #FACEBOOK FASTTEXT
    file = open("data_1/FastText.vec", "r")
    FastText = {}
    while True :
        line = file. readline()
        if (line == ""):
            break
        word = line.split()[0]
        vec = [float(i) for i in line.split()[1:51]]
        FastText[word] = vec

    #MIKOLOV WORD2VEC
    file = open("data_1/Word2Vec.txt", "r")
    Word2Vec = {}
    while True :
        line = file. readline()
        if (line == ""):
            break
        word = line.split()[0]
        vec = [float(i) for i in line.split()[1:51]]
        Word2Vec[word] = vec

    #STANFORD GLOVE
    file = open("data_1/GloVe.txt", "r")
    GloVe = {}
    while True :
        line = file. readline()
        if (line == ""):
            break
        word = line.split()[0]
        vec = [float(i) for i in line.split()[1:51]]
        GloVe[word] = vec

    return FastText, Word2Vec, GloVe

def extract_data(data, FastText, Word2Vec, GloVe):
    #y_d = np.zeros((len(data), 4)) # etiqueta [N, NEU, P, NONE]
    stop = get_stop_words()
    y_d = np.ones((len(data), 4))
    max_mat = 0
    text = []
    k = 0
    for i in range(len(data)):
        temp = tokenizer(data.at[i, "content"], stop)
        #temp = data.at[i, "content"].split()
        temp_2 = []
        for j in range(len(temp)):
            temp[j] = temp[j].lower()
            if (temp[j] in FastText)*(temp[j] in Word2Vec)*(temp[j] in GloVe):
                temp_2.append(temp[j])
        if (len(temp_2) > 2):
            max_mat = max(max_mat, len(temp_2))
            text.append(temp_2)
            x = data.at[i, "polarity"]
            y_d[k,:] = [(x =='N')*1, (x =='NEU')*1, (x =='P')*1, (x =='NONE')*1]
            #y_d[k,:] = [(x =='N')*1, (x =='NEU')*1, (x =='P')*1]
            k = k + 1
    #y_d = y_d[~np.all(y_d == 0, axis=1)]
    y_d = y_d[~np.all(y_d == 1, axis=1)]
    return text, y_d, max_mat

def decode_we(x, max_mat, FastText, Word2Vec, GloVe):
    #x_d = np.zeros((len(x), max_mat, 50, 3))
    x_d = np.ones((len(x), max_mat, 50, 3))
    for i in range(len(x)):
        for j in range(len(x[i])):
            x_d[i, j, :, 0] = FastText[x[i][j]]
            x_d[i, j, :, 1] = Word2Vec[x[i][j]]
            x_d[i, j, :, 2] = GloVe[x[i][j]]
    return x_d

def get_minibatch(n, it, x, y):
    offset = (it * n) % (len(x) - n)
    x_ = x[offset:(offset + n),:,:,:]
    y_ = y[offset:(offset + n),:]
    return x_, y_

def cross_validation(it, n, x, y):
    temp = int(len(y)/n)
    x_test = x[it*temp:(it+1)*temp,:,:,:]
    y_test = y[it*temp:(it+1)*temp,:]
    x_train = np.delete(x, np.s_[it*temp:(it+1)*temp], 0)
    y_train = np.delete(y, np.s_[it*temp:(it+1)*temp], 0)
    return x_train, y_train, x_test, y_test
