# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    self.prob={}
    self.temp={}
    self.prior={}
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

  import dataClassifier
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    #print trainingData, trainingLabels
    count=util.Counter()
    self.prior=util.Counter()
    Nlabels=0
    maxval=0
    maxprob = None
    for label in trainingLabels:
        #print label
        count[label]+=1
        Nlabels+=1
    for i in self.legalLabels:
        #print i,count[i]
        self.prior[i]=count[i]/float(Nlabels)
    #print self.prior
    count2 = {}
    for label in self.legalLabels:
        count2[label] = util.Counter()
    for i in range(len(trainingData)):
        for feature in self.features:
            count2[trainingLabels[i]][feature] = 0
    for i in range(len(trainingData)):
        for feature in self.features:
            if trainingData[i][feature] == 1:
                count2[trainingLabels[i]][feature] += 1
    for k in kgrid:
        prob = []
        for label in self.legalLabels:
            p = util.Counter()
            for feature in self.features:
                p[feature]=(count2[label][feature] + k)/float(count[label] + k)
            p.normalize()
            prob.append(p)
            #print prob
        self.prob=prob
        guesses=self.classify(validationData)
        #print guesses
        s=0
        for i in range(len(guesses)):
            if guesses[i] == validationLabels[i]:
                s += 1
        if s > maxval:
            maxval = s
            maxprob = prob
        #print k,maxval
    self.prob = maxprob


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    for l in self.legalLabels:
        logJoint[l]=0
        for f,v in datum.items():
            #print v,'------',self.prob[l][f]
            if v == 1:
                logJoint[l]+=math.log(self.prob[l][f])
        #print self.prior[l]
        logJoint[l]+=math.log(self.prior[l])
    #print logJoint
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
    ratio = util.Counter()
    for f in self.features:
        ratio[f] = (1.0* self.prob[label1][f]) / (1.0 * self.prob[label2][f])
    featuresOdds = ratio.sortedKeys()
    featuresOdds = featuresOdds[0:100]
    print featuresOdds
    return featuresOdds