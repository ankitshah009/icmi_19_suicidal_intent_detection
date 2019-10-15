from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
X_data=np.load('X_manual_data.npy')
Y_labels=np.load('Y_labels.npy')
X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_labels,train_size=0.75,test_size=0.25)

tpot=TPOTClassifier(generations=100,population_size=25,verbosity=2,n_jobs=-1)
tpot.fit(X_train,Y_train)
print(tpot.score(X_test,Y_test))
tpot.export('suicide_100_25.py')
