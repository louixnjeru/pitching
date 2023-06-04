import pybaseball as pb
import pandas as pd
import matplotlib.pyplot as plt

pitches = ['AB','AS','CH','CU','EP','FC','FF','FO','FS','FT','GY','IN','KC','KN','NP','PO','SC','SI','SL','UN']
pitchMaps = {p:i for i,p in enumerate(pitches)}
del pitches

def mapPitches(column):
    return [pitchMaps[p] for p in column.to_numpy()]

#k = pb.statcast_pitcher('2017-01-01', '2018-01-01', 477132)
k = pb.statcast_pitcher('2019-01-01', '2020-01-01', 543037)
k = k[k['plate_x'].notna()]
pd.options.display.max_rows = None
pd.options.display.max_columns = None
print(k[['plate_x','plate_z','pitch_type','release_speed','release_spin_rate']])
pitchMappings = {p:i for i,p in enumerate(list(set(k['pitch_type'].to_numpy())))}
print(pitchMappings)
k.plot(kind = 'scatter', x = 'plate_x', y = 'plate_z', c = mapPitches(k['pitch_type']), cmap='nipy_spectral')
k.plot(kind = 'scatter', x = 'release_speed', y = 'release_spin_rate', c = mapPitches(k['pitch_type']), cmap='nipy_spectral')

from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as met
import numpy as np
from sklearn.model_selection import train_test_split

def getPitchMap(pitch):
    return pitchMaps[pitch]

model1 = KNeighborsClassifier(n_neighbors=5)


kPruned = k[k['release_spin_rate'].notnull() & k['pitch_type'].notnull() & k['release_speed'].notnull()]
X,y = kPruned[['release_speed','release_spin_rate']],kPruned['pitch_type']

trainX, testX, trainY, testYAct = train_test_split(X,y,test_size=0.25)

model1.fit(trainX,trainY)

testYPred = model1.predict(testX)

accuracy = met.accuracy_score(testYAct,testYPred)*100
print(f'Accuracy: {accuracy}')

fig = plt.figure()
ax = fig.add_subplot(projection = '3d'i)
ax.scatter(k['plate_x'].to_numpy(), k['pfx_z'].to_numpy(), k['plate_z'].to_numpy(), c = mapPitches(k['pitch_type']), cmap = 'nipy_spectral')
ax.set_xlabel('Horizontal Plate Position')
ax.set_ylabel('Vertical Movement')
ax.set_zlabel('Vertical Plate Position')
fig.set_size_inches(16, 16)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(k['plate_x'].to_numpy(), k['plate_z'].to_numpy(), k['pfx_z'].to_numpy(), c = mapPitches(k['pitch_type']), cmap = 'nipy_spectral')
ax.set_xlabel('Horizontal Plate Position')
ax.set_ylabel('Vertical Plate Position')
ax.set_zlabel('Vertical Movement')
fig.set_size_inches(16, 16)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(k['release_speed'].to_numpy(), k['release_spin_rate'].to_numpy(), k['pfx_z'].to_numpy(), c = mapPitches(k['pitch_type']), cmap = 'nipy_spectral')
ax.set_xlabel('Speed (mph)')
ax.set_ylabel('Release Spin Rate (rpm)')
ax.set_zlabel('Vertical Movement (ft)')
fig.set_size_inches(16, 16)
plt.show()

import sklearn.metrics as met

cMatrix = met.confusion_matrix(testYAct, testYPred)
cmDisplay = met.ConfusionMatrixDisplay(confusion_matrix=cMatrix)
cmDisplay.plot()
plt.show()

