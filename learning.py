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


plt.show()