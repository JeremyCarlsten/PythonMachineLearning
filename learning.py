import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# do plotting inline
#%matplotlib inline

dataFile = pd.read_csv('./data/pima-data.csv')
shape = dataFile.shape
print(shape) # (768,10)

print(dataFile.head(5)) #first five rows
print("\n\n****************************************************************************\n\n")
print(dataFile.tail(5)) #last five rows
print("\n\n****************************************************************************\n\n")
print(dataFile.isnull().values.any()) #check for any null values

def plotCorr(dataFile, size=11):
    corr = dataFile.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)


# plotCorr(dataFile) #Show the corrilation table

del dataFile['skin']  #remove duplicated column data
print (dataFile.head(5)) # no more skin column....

####################################
# Change True to 1 and False to 0
####################################
print('Changing diabetes to 1s and 0s')

diabetesMap = {True : 1, False : 0}
dataFile['diabetes'] = dataFile['diabetes'].map(diabetesMap)

print (dataFile.head(5))



####################################
#checking percentages of true vs false
####################################

numTrue = len(dataFile.loc[dataFile['diabetes'] == True])
numFalse = len(dataFile.loc[dataFile['diabetes'] == False])

print("Number of true cases: {0} ({1:2.2f}%)".format(numTrue, (numTrue/(numTrue + numFalse)) * 100))
print("Number of false cases: {0} ({1:2.2f}%)".format(numFalse, (numFalse/(numTrue + numFalse)) * 100))

####################################
# split the data into test and training data. 70% training 30% testing
####################################



plt.show()