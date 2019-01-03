"""
This script educates the usage of linear regression on insurance dataset 
"""
import numpy
import pandas as pd
import time

"""
This class executes the linear regression and provides the necessary infrastructure
to execute gradient descent.
"""
class LinearRegression:
    def __init__(self, X, Y, learningRate, toleranceLimit, maxIterations, L1Parameter, L2Parameter, verbose = False):
        self.features = numpy.ones((X.shape[0], (X.shape[1]+1)))
        self.features[:, 1:] = X                        
        self.weights = numpy.zeros((X.shape[1]+1, 1))
        self.tolerance = toleranceLimit
        self.Y = Y.reshape((Y.shape[0],1))
        self.maxIterations = maxIterations
        self.verbose = verbose
        self.learningRate = learningRate
        self.iteration = 0
        self.L1Parameter = L1Parameter
        self.L2Parameter = L2Parameter
    
    def getCost(self):
        estimateValues = self.features.dot(self.weights)
        cost = ((self.Y - estimateValues)*(self.Y - estimateValues)).sum()
        regularizedSum = self.L2Parameter * (numpy.nansum(self.weights * self.weights))
        regularizedSum = regularizedSum + self.L1Parameter * (numpy.nansum(numpy.fabs(self.weights)))
        return cost + regularizedSum
        
    def getCostGradient(self):
        estimateValues = self.features.dot(self.weights)
        cost = -(2*(self.Y - estimateValues))/self.Y.shape[0]
        product = cost * self.features
        gradientOfLikelihood = product.sum(axis = 0).reshape(self.weights.shape)
        regularizedL2Gradient = 2 * self.L2Parameter * (self.weights)
        regularizedL1Gradient = self.L1Parameter * (numpy.sign(self.weights))
        return gradientOfLikelihood + regularizedL2Gradient + regularizedL1Gradient
    
    def runGradientDescent(self):
        previousCost = 0
        currentCost = self.getCost()
        self.iteration = 1
        if self.verbose:
            print("Cost before GD = " + str(currentCost))
        while((numpy.fabs(currentCost - previousCost) > self.tolerance) and self.iteration < self.maxIterations):
            costGradient = self.getCostGradient()
            self.weights = self.weights - self.learningRate*costGradient
            previousCost = currentCost
            currentCost = self.getCost()
            if self.verbose:
                print("Cost after GD Step " + str(self.iteration) + " = " + str(currentCost))
            self.iteration = self.iteration + 1
        
"""
Function to scale feature .
X = X - Mean(X) / Std(X)
"""
def scaleFeature(feature):
    mean = feature.mean(axis = 0)
    std = feature.std(axis=0)
    return (feature-mean)/std

"""
Unscaling the weights according to the mechanism we used 
to scale the features
"""
def unscaleWeights(weights, unscaledFeature):
    mean = unscaledFeature.mean(axis = 0)
    std = unscaledFeature.std(axis=0)
    params = weights.T[0]/numpy.hstack((1,std))
    params[0] = params[0] - (mean*weights.T[0][1:]/std).sum()
    return params

if __name__ == "__main__":
    dataSet = pd.read_csv("data/insurance.csv")  
    dataFrame = pd.DataFrame(dataSet)
    dataFrame['region'] = dataFrame['region'].astype("category")
    dataFrame['smoker'] = dataFrame['smoker'].astype("category")
    dataFrame['sex'] = dataFrame['sex'].astype("category")
    cat_columns = dataFrame.select_dtypes(['category']).columns
    dataFrame[cat_columns] = dataFrame[cat_columns].apply(lambda x: x.cat.codes)
    X = dataFrame.loc[:,dataFrame.columns[0:6]].values

    Y = dataFrame['expenses'].values
    Xscaled = scaleFeature(X)
    startTime = time.time()
    linearRegression = LinearRegression(Xscaled, Y, 0.01, 0.0001, 2000, 0, 0, verbose=False)
    print("Initial Cost before GD : {}".format(linearRegression.getCost()))
    linearRegression.runGradientDescent()
    endTime = time.time()
    params = unscaleWeights(linearRegression.weights, X)
    print("Iterations required = " + str(linearRegression.iteration))
    print("Time taken for Classifier = " + str(endTime - startTime))
    paramsDict = {}
    paramsDict["intercept"] = params[0]
    for index in range(len(dataFrame.columns)-1):
        paramsDict[dataFrame.columns[index]] = params[index+1]
    print(paramsDict)
    print("Cost after GD : {}".format(linearRegression.getCost()))
    
