import csv
import scipy as sp
import numpy as np
import time
import pandas as pd
import re
import numpy as np
import mmap
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import AgglomerativeClustering,dbscan
#import multiprocessing as mp
import nltk
stemmer = nltk.stem.porter.PorterStemmer()
from scipy.cluster.hierarchy import dendrogram, linkage

#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.datasets import load_files
#nltk.download('stopwords')
#import pickle
#from nltk.corpus import stopwords

import processAttribute as at
import tokenization as tk

def saveDictToFile(myDict,fname):
    w = csv.writer(open(fname, "w"))
    for key, val in myDict.items():
        w.writerow([key, val])

def countFileLines(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines

def readFile(filename,nRows):
    df = pd.read_csv(filename,delimiter="\t",usecols=(0,1,5,7),nrows=nRows)
    return df

def readValidation(filename):
    #data = np.genfromtxt("mlchallenge_set_validation.tsv",delimiter="\t",usecols=(0,1,2,5,7),max_rows=1000)
    data = np.genfromtxt("mlchallenge_set_validation.tsv",delimiter="\t")
    print(data[:,-1])
    print(len(np.unique(data[:,:1])))
    return data

def printClusterAssign(df,labels):
    labelsList = np.unique(labels)
    count = 0
    empty = 0
    count1 = 0
    for i in labelsList:
        thisCluster = df[labels == i]
        x = thisCluster['title'].values

        if len(x)>1:
            print("New Cluster:" )
            print(x)
            print(thisCluster['attributes'].values)
            print("\n\n" )
            count += 1
        if len(x) == 0:
            empty+=1
        if len(x) == 1:
            count1+=1
    print(len(labelsList))
    print("More than1: " + str(count) + ", Empty: " + str(empty) + ", Only 1: " + str(count1))


def stemDataFrame(data):
    sepWord = []
    temp = data
    # Pre-process the data - contains : and , and spaces
    # Steps add spaces around colon and comma and then split by space and then stem
    # Finally merge , remove spaces so that we store info where colons and commas
    # were there and still split by all three punctuations
    data = data.replace(',',' , ')
    data = data.replace(':',' : ')

    for word in data.split(' '):
        #if i == 6:
        #    print(str(word) + '---' + str(stemmer.stem(word)))
        sepWord.append(stemmer.stem(word))
    data = ' '.join(sepWord)

    # Remove previously added spaces
    data = data.replace(' , ',',')
    data = data.replace(' : ',':')
    return data

def preProcess(df):
    #pool = mp.Pool(min(mp.cpu_count()-1,-1))
    for j in df.columns:
        #pool( #Using parallel pool
        if j in ['title', 'attributes']:
            thisDf = df[j].values
            for i in range(len(thisDf)):
                # Remove all special characters other than : and ,
                thisDf[i] = re.sub('[^,:,a-zA-Z0-9 \n\.]', ' ', thisDf[i])

                # remove all single characters
                thisDf[i] = re.sub(r'\s+[a-zA-Z]\s+', '', thisDf[i])

                # Convert multiple commas to single comma
                thisDf[i] = re.sub(r'(,){2,}', ',', thisDf[i])

                # Convert multiple spaces to single space
                thisDf[i] = re.sub(r'(\s){2,}', ' ', thisDf[i])

                # Convert to lower case and strip extra spaces
                thisDf[i] = thisDf[i].lower().strip()
                temp = thisDf[i]
                # Use nltk stemmer to create stemmed words
                #sepWord  = [stemmer.stem(word) for word in thisDf[i]]

                thisDf[i] = stemDataFrame(thisDf[i])
            # Assign the value back into dataframe
            df[j].value = thisDf
            #)  # End of parallel pool
    return df

# Start of Program
nRows = 1000#maxRows = 1002276
#cutOff = 5 # Cutoff weights -Ignores attributes repeated less than cutoff times

filename = "data.tsv"
#filename = "abc1.tsv"
start_time = time.time()

#print(countFileLines(filename))
#end_time = time.time() - start_time

df = readFile(filename,nRows)
#print(df.columns)
catList = np.array(df['category'].values)
catList = np.unique(catList)

indexes = np.array(df['index'].values)

# Pre-process the input file for simplification
df = preProcess(df)
print("Pre Processing Time Taken: " + str(time.time() - start_time))
# Get the attribute for processing
oldMaxLab=0
for catId in range(len(catList)):
    catbool = df['category']== catList[catId]
    sectDf = df[catbool]
    attribute = sectDf['attributes'].values
    masterDict = at.createAttributeList(attribute)
    #print(masterDict)
    saveDictToFile(masterDict,'MasterDictionary' + str(catList[catId]) + '.csv')
    #np.set_printoptions(threshold=sys.maxsize)
    dataMatrix = tk.tokenizeData(sectDf,masterDict)
    print("Starting Clustering for Group : " + str(catList[catId]) + "Time: " + str(time.time() - start_time))

    cluster = AgglomerativeClustering().fit(dataMatrix)
    cluster.fit_predict(dataMatrix)
    labels = cluster.labels_
    cluster,labels = dbscan(dataMatrix,eps=0.1,min_samples=1,n_jobs=4)
    labels = [x+oldMaxLab for x in labels]
    #print(labels)
   # plt.figure(figsize=(10, 7))  
    #plt.scatter(dataMatrix[:,0], dataMatrix[:,1], c=labels) 
    #plt.show()
    Z = linkage(dataMatrix, 'ward')
    #plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z,leaf_rotation=90.,    leaf_font_size=8., )
    plt.show()

    print("End of Clustering for Group : " + str(catList[catId]) + "Time: " + str(time.time() - start_time))

    clusterDict = dict()
 
    index1 = sectDf['index'].values
    for i in range(len(index1)):
        clusterDict[index1[i]] = labels[i]
    fname = 'CG_100k_' + str(catList[catId]) + '.csv'
    print("Printing file: " + fname)
    saveDictToFile(clusterDict,fname)


'''
    print(np.unique(labels))
    printClusterAssign(sectDf,labels)
    print(df['title'].values[4026])
    print(df['attributes'].values[4026])
    print('')
    print(df['title'].values[8064])
    print(df['attributes'].values[8064])
    print('')
'''
    #print(df['title'].values)

# Tokenization of values
#tokenDict = tk.getTokenDict(masterDict)


    
	

#print(np.array(df))
#print(np.unique(a))
print("Total Time Taken: " + str(time.time() - start_time))



#print(data[1:6])
#print(len(df))
