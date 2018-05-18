import xml.etree.cElementTree as et
from sklearn.utils import shuffle
import re
import numpy as np
import pandas as pd

def get_stop_words():
    file = open("data_2/stopword.txt", "r")
    stop = []
    while True :
        line = file. readline()
        if (line == ""):
            break
        word = line.split()[0]
        stop.append(word)
    return stop

def tokenizer(text, stop):
    text = str(text)
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text)
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^|][0-9]+', ' ', text)
    text = re.sub(r'[\s!/,\\.?¡¿"“”#:/();]+', ' ', text) #”
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
    #text = re.sub(r'\b(d\b)+', 'de', text)
    text = re.sub(r'^/(?!jugador)\b(\w+)( \1\b)+', r'\1', text)
    #text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    #text = re.sub(r'\b-\b', ' ', text)
    text = re.findall(r'[^\s!,.?¡¿"“”:;]+', text)
    text = [w for w in text if w not in stop]
    return text

def getvalueofnode(node):
    """ return node text or None """
    return node.text if node is not None else None

def loadData(x_dir):
    s = []
    parsedXML = et.parse( x_dir )
    dfcols = ['content', 'polarity']
    aspects = []
    polarities = []
    for node in parsedXML.getroot():
        temp = []
        temp2 = []
        for sentiment in node.findall('sentiment'):
            temp.append(sentiment.get('aspect'))
            temp2.append(sentiment.get('polarity'))
            sentiment.text = sentiment.get('aspect')
            sentiment.text = ' ' + str(sentiment.text) + ' '
        aspects.append(temp)
        polarities.append(temp2)
        s.append(''.join(node.itertext()))
    return s, aspects, polarities

def loadChannels():
    #FACEBOOK FASTTEXT
    file = open("data_2/FastText.vec", "r")
    FastText = {}
    while True :
        line = file. readline()
        if (line == ""):
            break
        word = line.split()[0]
        vec = [float(i) for i in line.split()[1:51]]
        FastText[word] = vec

    #MIKOLOV WORD2VEC
    file = open("data_2/Word2Vec.txt", "r")
    Word2Vec = {}
    while True :
        line = file. readline()
        if (line == ""):
            break
        word = line.split()[0]
        vec = [float(i) for i in line.split()[1:51]]
        Word2Vec[word] = vec

    #STANFORD GLOVE
    file = open("data_2/GloVe.txt", "r")
    GloVe = {}
    while True :
        line = file. readline()
        if (line == ""):
            break
        word = line.split()[0]
        vec = [float(i) for i in line.split()[1:51]]
        GloVe[word] = vec

    return FastText, Word2Vec, GloVe

def get_data(data, wnd, FastText, Word2Vec, GloVe):
    # data: [0] sentences [1] aspects [2] polarity
    oraciones = data[0]*1
    aspectos = data[1]*1
    polaridad = data[2]*1
    text = []
    max_mat = 0
    stop = get_stop_words()
    labels = []
    it_lab = 0
    for i in range(len(oraciones)):
        oraciones[i] = tokenizer(oraciones[i], stop)
        temp = []
        for j in range(len(oraciones[i])):
            if (oraciones[i][j] in FastText)*(oraciones[i][j] in Word2Vec)*(oraciones[i][j] in GloVe):
                temp.append(oraciones[i][j])
        oraciones[i] = temp
        if len(oraciones[i]) > 1:
            for j in range(len(aspectos[i])):
                aspectos[i][j] = tokenizer(aspectos[i][j], stop)[0]
                if (aspectos[i][j] in FastText)*(aspectos[i][j] in Word2Vec)*(aspectos[i][j] in GloVe):
                    k = oraciones[i].index(aspectos[i][j])
                    k_min = max(0, k-wnd)
                    k_max = min(k+wnd+1, len(oraciones[i]))
                    temp = polaridad[i][j]
                    y = [(temp =='N')*1, (temp =='NEU')*1, (temp =='P')*1]
                    max_mat = max(k_max-k_min, max_mat)
                    if aspectos[i][j] not in labels:
                        labels.append(aspectos[i][j].lower())
                        text.append([0])
                        text[it_lab].append([oraciones[i][k_min:k_max], y])
                        text[it_lab].pop(0)
                        it_lab += 1
                    else:
                        text[labels.index(aspectos[i][j].lower())].append([oraciones[i][k_min:k_max], y])
    return text, max_mat

def decode_we(text, max_mat, FastText, Word2Vec, GloVe):
    x_d = []
    y_d = []
    for i in range(len(text)):
        x_d.append(np.ones((len(text[i]), max_mat, 50, 3)))
        y_d.append(np.ones((len(text[i]), 3)))

    #print(text)
    for i in range(len(text)): # etiquetas
        for j in range(len(text[i])): #num oraciones
            for k in range(len(text[i][j])): # oracion + etiqueta
                for ii in range(len(text[i][j][0])):
                    #print(i,' ',j,' ',k,' ',ii)
                    x_d[i][j, k, :, 0] = FastText[text[i][j][0][ii]]
                    x_d[i][j, k, :, 1] = Word2Vec[text[i][j][0][ii]]
                    x_d[i][j, k, :, 2] = GloVe[text[i][j][0][ii]]
                    y_d[i][j, :] = text[i][j][1]
    return x_d, y_d

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
