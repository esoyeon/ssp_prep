import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_pickle("WM811K.pkl")

#list the field name of the structure
df.info()
# Select training and test data
trainIdx=df[df['trainTestLabel']=='Training'].index
testIdx=df[df['trainTestLabel']=='Test'].index

#show each failure type
trainFailureType=df.loc[trainIdx,'failureType']
testFailureType=df.loc[testIdx,'failureType']
uniqueType=df.loc[trainIdx,'failureType'].unique()
uniqueType.sort()
print(uniqueType)

#Plot a wafer map
idx=trainFailureType[trainFailureType==uniqueType[0]].index
exampleIdx=idx[0]
plt.imshow(df.iloc[exampleIdx]['waferMap'],)


