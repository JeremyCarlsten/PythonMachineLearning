import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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

featureColNames = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predictedClassNames = ['diabetes']

X = dataFile[featureColNames].values      #predictor feature columns (8 X m)
y = dataFile[predictedClassNames].values  #predicted class (1 = true, 0 = false) column (1 x m)
splitTestSize = 0.30

# test_size = 0.3 it's 30%, 42 is the answer to life, the universe, and everything
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitTestSize, random_state=42)

#check the training/test percentages matches 70/30 split
print("{0:0.2f}% in training set".format((len(X_train) / len(dataFile.index)) * 100 )) 
print("{0:0.2f}% in test set".format((len(X_test) / len(dataFile.index)) * 100 )) 

#verify the test/training data matches original true and false percentages
print("\n\nOriginal Number of true cases: {0} ({1:2.2f}%)".format(numTrue, (numTrue/(numTrue + numFalse)) * 100))
print("Original Number of false cases: {0} ({1:2.2f}%)".format(numFalse, (numFalse/(numTrue + numFalse)) * 100))

print("\nNumber of true test cases: {0} ({1:2.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1]) / len(y_train) * 100)))
print("Number of false test cases: {0} ({1:2.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0]) / len(y_train) * 100)))

print("\nNumber of true training cases: {0} ({1:2.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1]) / len(y_test) * 100)))
print("Number of false training cases: {0} ({1:2.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0]) / len(y_test) * 100)))

#########################################
#Find hidden missing values. 
#########################################

#Check for rows with zero values
print('\n\n### Checking for Hidden missing values')
print("# rows in dataframe {0}".format(len(dataFile)))
print("# rows missing glucose_conc: {0}".format(len(dataFile.loc[dataFile['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(len(dataFile.loc[dataFile['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(len(dataFile.loc[dataFile['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(dataFile.loc[dataFile['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(dataFile.loc[dataFile['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(len(dataFile.loc[dataFile['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(dataFile.loc[dataFile['age'] == 0])))

#insulin is the only one that might have a zero for a value. 
#impute - replce the empty fields with something reasonable
#since we aren't docs, lets use the mean.

#Imputer deprecated using SimpleImputer instead...
#fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

fill_0 = SimpleImputer(missing_values=0, strategy="mean") #SimpleImputer doesn't take axis?? issue? not sure. 
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


#########################################
# Start training.
#########################################

from sklearn.naive_bayes import GaussianNB

# create Gaussian NB model object and train it with the data
# Gaussian assumes the feature data is distributed on a gaussian (bell curve) with most of the data near the mean

nb_model = GaussianNB()

nb_model.fit(X_train, y_train.ravel()) # what does ravel() function do? works like groovy flatten(), from the numpy package.

# Performance on Training Data
from sklearn import metrics

nb_predict_test = nb_model.predict(X_test)

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))

plt.show()