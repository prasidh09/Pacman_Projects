#id3.py

import util
import classificationMethod
import math
import sys
import time
from itertools import groupby

class Id3Classifier(classificationMethod.ClassificationMethod):
  """
  Note that the variable 'datum' in this code refers to a counter of features
  not to a raw samples.Datum).
  """
  f=[]
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "id3"
    self.prior=0
    self.DecisionTree=None

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    #print self.features
    #f[0]=util.Counter()
    #print trainingData,trainingLabels[1]
    self.DecisionTree = self.createTree(trainingData,trainingLabels,validationData,validationLabels,self.features)
    #print self.DecisionTree

  def createTree(self,trainingData,trainingLabels,validationData,validationLabels,features):
    count=util.Counter()
    Nlabels=0
    children=[]
    for label in trainingLabels:
        count[label]+=1
        Nlabels+=1
    #print count,Nlabels
    for i in self.legalLabels:
        #print i,count[i]
        factor = count[i]/float(Nlabels)
        #print factor
        if not factor == 0:
            self.prior-=factor*math.log(factor,2)
    IG=util.Counter()
    maxIG=float("-inf")
    currentRoot='root'
    for f in features:
        #calc information gain
        possible_values=[]
        outcome=[]
        for i in range(len(trainingData)):
            possible_values.append(trainingData[i][f])
            outcome.append(trainingLabels[i])
        #total_count=[len(list(group)) for key, group in groupby(possible_values)]
        PVset=set(possible_values)
        newPV=list(PVset)
        total = util.Counter()
        H = util.Counter()
        Ntotal=0.0
        #print newPV
        for item in newPV:
            count = util.Counter()
            for i in range(len(possible_values)):
                if item == possible_values[i]:
                    total[item]+=1
                    count[outcome[i]]+=1
            for i in range(len(outcome)):
                factor=count[outcome[i]]/float(total[item])
                if not factor==0:
                    H[item]-=factor*math.log(factor,2)
            Ntotal+=total[item]
        IG[f]=self.prior
        for item in newPV:
            IG[f]-=H[item]*total[item]/Ntotal
            #print IG[f]
        if maxIG<IG[f]:
            maxIG=IG[f]
            currentRoot=f
            children = newPV
    #print children
    tree=util.Counter()
    subtrees=[]
    features.remove(currentRoot)
    for child in children:
        tData=[]
        tLabels=[]
        for i in range(len(trainingData)):
            if trainingData[i][currentRoot]==child:
                tData.append(trainingData[i])
                tLabels.append(trainingLabels[i])
        flag=0
        if len(tLabels)>1:
            for i in range(len(tLabels)-1):
                if not tLabels[i]==tLabels[i+1]:
                    flag=1
                    break
        if flag==0 or len(tLabels)<=1:
            leaf=util.Counter()
            leaf['leaf']=tLabels[0]
            subtrees.append((child,leaf))
        else:
            subtrees.append((child,self.createTree(tData,tLabels,validationData,validationLabels,features)))
    tree[currentRoot]=subtrees
    return tree


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
        f=self.DecisionTree.keys()
        tree=self.DecisionTree[f[0]]
        #print f
        while not f==['leaf']:
            value=datum[f[0]]
            #print tree
            for child in tree:
                #print child
                if child[0]==value:
                    f=child[1].keys()
                    tree=child[1][f[0]]
                    #print f,tree
                    break
        guesses.append(tree)
    return guesses
      


    
      
