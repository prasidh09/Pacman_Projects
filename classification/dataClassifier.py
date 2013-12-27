# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

import mostFrequent
import naiveBayes
import id3
import perceptron
import mira
import samples
import sys
import util

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70

def height(features):
  top=0
  bottom=0
  min_top=DIGIT_DATUM_HEIGHT
  max_bottom=0
  #print features
  #print ""
  for x in range(0,DIGIT_DATUM_WIDTH-1):
    flag=0
    for y in range(0,DIGIT_DATUM_HEIGHT-1):
      #print features[(x,y)]
      if features[(x, y)] ==1:
          top=y
          flag=1
          #print top,"got top"
          break
    for y in range(DIGIT_DATUM_HEIGHT-1,0,-1):
      if features[(x, y)] ==1:
          bottom=y
          #print bottom,"got bottom"
          break
    #print top,bottom
    if flag==1:
        if top<min_top:
          #print min_top,top
          min_top=top
        if bottom>max_bottom:
          max_bottom=bottom
  return max_bottom-min_top+1


def Width(features):
  left=0
  right=0
  min_left=DIGIT_DATUM_WIDTH
  max_right=0
  #print features
  #print ""
  for y in range(0,DIGIT_DATUM_HEIGHT-1):
    flag=0
    for x in range(0,DIGIT_DATUM_WIDTH-1):
      #print features[(x,y)]
      if features[(x, y)] ==1:
          left=x
          flag=1
          #print left,"got left"
          break
    for x in range(DIGIT_DATUM_WIDTH-1,0,-1):
      if features[(x, y)] ==1:
          right=x
          #print right,"got right"
          break
    #print left,right
    if flag==1:
        if left<min_left:
          #print min_left,left
          min_left=left
        if right>max_right:
          max_right=right
  return max_right-min_left+1

def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def addTopHeavyFeature(datum, features):
  topPixels = 0
  for x in range(DIGIT_DATUM_WIDTH):
      for y in range(DIGIT_DATUM_HEIGHT/2):
          if datum.getPixel(x, y) > 0:
              topPixels += 1
              
  bottomPixels = 0
  for x in range(DIGIT_DATUM_WIDTH):
      for y in range(DIGIT_DATUM_HEIGHT/2, DIGIT_DATUM_HEIGHT):
          if datum.getPixel(x, y) > 0:
              bottomPixels += 1
  
  if (topPixels - bottomPixels) > 0:
     features['top'] = 1
  else:
     features['top'] = 0
  weighFeature(features, 'top', 4)

def addSymmetricalFeature(datum, features):
    differences = []
    for y in range(DIGIT_DATUM_HEIGHT):
      leftPix = 0
      rightPix = 0
      for x in range(DIGIT_DATUM_WIDTH/2):
        if datum.getPixel(x, y) > 0:
          leftPix += 1
      for x in range(DIGIT_DATUM_WIDTH/2, DIGIT_DATUM_WIDTH):
        if datum.getPixel(x, y) > 0:
          rightPix += 1
          
      differences.append(abs(leftPix - rightPix))
      
    avg = float(sum(differences)) / len(differences)
    if avg > 2:
      features['symmetrical'] = 1
    else:
      features['symmetrical'] = 0

def weighFeature(features, name, weight):
  for i in xrange(weight):
    features[str(i) + name] = features[name]

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.
  
  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## DESCRIBE YOUR ENHANCED FEATURES HERE...
  1) 'top': We determine whether a digit has more 'ON' pixels
  in the top half of the picture than the bottom half.
  2) 'symmetrical': We determine the difference, across a given
  horizontal line in the picture, between the left side and the
  right side. If it's close to 0, we say that it is more symmetrical.
  3) 'loop': We determine whether there is a loop or not.
  ##
  """
  features =  basicFeatureExtractorDigit(datum)

  h=height(features)
  w=Width(features)
  #print h,w
  cellS=4
  cellh=h/cellS
  cellw=w/cellS
  count1=0
  count0=0
  for cell in range(cellS*cellS):
        count = 0
        startx=(cell%cellS)*cellw
        starty=(cell/cellS)*cellh
        stopx=startx+cellw
        if stopx>w-1:
            stopx=w
        stopy=starty+cellh
        if stopy>h-1:
            stopy=h
        for x in range(startx, stopx):
          for y in range(starty, stopy):
              if features[(x, y)] ==1:
                  count =count + 1
        #print stopx,startx,stopy,starty
        if count/float((stopx-startx)*(stopy-starty))>0.2:
            features['Cell'+str(cell)]=1
            #print features['Cell'+str(cell)]
            count1+=1
        else:
            features['Cell'+str(cell)]=0
            #print features['Cell'+str(cell)]
            count0+=1
  addTopHeavyFeature(datum, features)
  #xaddSymmetricalFeature(datum, features)
  
  masterLoop = False
  loopStart = False
  loopStarted = 0
  loopStartX = 0
  loopEnd = False
  loopEnded = 0
  loopEndX = 0
  prevGap = 0
  prevStart = 0
  prevEnd = 0
  for y in range(DIGIT_DATUM_HEIGHT):
    start = 0
    end = 0
    total = 0
    for x in range(DIGIT_DATUM_WIDTH):
      if datum.getPixel(x, y) > 0:
          total += 1
          if start == 0:
              start = x
          else:
              end = x
    
    if end == 0 and start != 0:
        end = start + 1          
    width = abs(start - end)
    if width > 1:
        width += 1
    gap = width - total
    if gap <= 0:
        gap = 0
    features['width' + str(y)] = width
    features['gap' + str(y)] = gap
    weighFeature(features, 'gap'+str(y), 2)
      
    features['start' + str(y)] = start
    weighFeature(features, 'start' + str(y), 4)
    features['end' + str(y)] = end
    weighFeature(features, 'end' + str(y), 4)
    
    if prevGap == 0 and gap != 0 and abs(prevStart - start) <= 2 and abs(prevEnd - end) <= 2:
        loopStart = True
        loopStarted += 1
        loopStartX = y
        #print 'loopStart: ', y
    
    if prevGap != 0:
        if gap == 0 and abs(prevStart - start) <= 2:
            loopEnd = True
            loopEnded += 1
            loopEndX = y
            if loopEndX > loopStartX + 1 and loopStarted == loopEnded:
                masterLoop = True
            #print 'loopEnd:', y
            
    prevGap = gap
    prevStart = start
    prevEnd = end
    
  if masterLoop:
      #print 'LOOPYLOOP'
      features['loop'] = 1
  else:
      features['loop'] = 0

  weighFeature(features, 'loop', 4)
  '''print features['Cell0'],features['Cell1'],features['Cell2'],features['Cell3']
  print features['Cell4'],features['Cell5'],features['Cell6'],features['Cell7']
  print features['Cell8'],features['Cell9'],features['Cell10'],features['Cell11']
  print features['Cell12'],features['Cell13'],features['Cell14'],features['Cell15']
  print '''''
  #print features
  #print count0,count1
  return features


def contestFeatureExtractorDigit(datum):
  """
  Specify features to use for the minicontest
  """
  features =  basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  features =  basicFeatureExtractorFace(datum)
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print "==================================="
          print "Mistake on example %d" % i 
          print "Predicted %d; truth is %d" % (prediction, truth)
          print "Image: "
          print rawTestData[i]
          break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print "new features:", pix
            continue
      print image  

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'id3', 'nb', 'naiveBayes','perceptron', 'mira', 'minicontest'], default='mostFrequent')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
  parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
  parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  # Set up variables according to the command line input.
  print "Doing classification"
  print "--------------------"
  print "data:\t\t" + options.data
  print "classifier:\t\t" + options.classifier
  if not options.classifier == 'minicontest':
    print "using enhanced features?:\t" + str(options.features)
  else:
    print "using minicontest feature extractor"
  print "training set size:\t" + str(options.training)
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit
    if (options.classifier == 'minicontest'):
      featureFunction = contestFeatureExtractorDigit
  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace      
  else:
    print "Unknown dataset", options.data
    print USAGE_STRING
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)
    
  if options.training <= 0:
    print "Training set size should be a positive integer (you provided: %d)" % options.training
    print USAGE_STRING
    sys.exit(2)
    
  if options.smoothing <= 0:
    print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
    print USAGE_STRING
    sys.exit(2)
    
  if options.odds:
    if options.label1 not in legalLabels or options.label2 not in legalLabels:
      print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
      print USAGE_STRING
      sys.exit(2)

  if(options.classifier == "mostFrequent"):
    classifier = mostFrequent.MostFrequentClassifier(legalLabels)
  elif(options.classifier == "id3"):
    classifier = id3.Id3Classifier(legalLabels)
  elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    classifier.setSmoothing(options.smoothing)
    if (options.autotune):
        print "using automatic tuning for naivebayes"
        classifier.automaticTuning = True
    else:
        print "using smoothing parameter k=%f for naivebayes" %  options.smoothing
  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  elif(options.classifier == "mira"):
    classifier = mira.MiraClassifier(legalLabels, options.iterations)
    if (options.autotune):
        print "using automatic tuning for MIRA"
        classifier.automaticTuning = True
    else:
        print "using default C=0.001 for MIRA"
  elif(options.classifier == 'minicontest'):
    import minicontest
    classifier = minicontest.contestClassifier(legalLabels)
  else:
    print "Unknown classifier:", options.classifier
    print USAGE_STRING
    
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
      
  # Load data  
  numTraining = options.training
  numTest = options.test

  if(options.data=="faces"):
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
  else:
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
    
  
  # Extract features
  print "Extracting features..."
  trainingData = map(featureFunction, rawTrainingData)
  validationData = map(featureFunction, rawValidationData)
  testData = map(featureFunction, rawTestData)
  
  # Conduct training and testing
  print "Training..."
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  print "Validating..."
  guesses = classifier.classify(validationData)
  correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
  print "Testing..."
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
  analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
  
  # do odds ratio computation if specified at command line
  if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
    label1, label2 = options.label1, options.label2
    features_odds = classifier.findHighOddsFeatures(label1,label2)
    if(options.classifier == "naiveBayes" or options.classifier == "nb"):
      string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
    else:
      string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)    
      
    print string3
    printImage(features_odds)

  if((options.weights) & (options.classifier == "perceptron")):
    for l in classifier.legalLabels:
      features_weights = classifier.findHighWeightFeatures(l)
      print ("=== Features with high weight for label %d ==="%l)
      printImage(features_weights)

if __name__ == '__main__':
  # Read input
  args, options = readCommand( sys.argv[1:] ) 
  # Run classifier
  runClassifier(args, options)

