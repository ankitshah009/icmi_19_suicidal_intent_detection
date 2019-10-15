from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
X_data=np.load('X_data.npy')
Y_labels=np.load('Y_labels.npy')
audio_feat_len=14
#acoustic_only = X_data[:, :audio_feat_len ]
verbal_feat_len=784
acoustic_and_verbal = X_data[:, : audio_feat_len + verbal_feat_len ]

X_train,X_test,Y_train,Y_test=train_test_split(acoustic_and_verbal,Y_labels,train_size=0.75,test_size=0.25)

tpot=TPOTClassifier(generations=40,population_size=25,verbosity=2,n_jobs=-1)
tpot.fit(X_train,Y_train)
print(tpot.score(X_test,Y_test))
tpot.export('suicide_40_25_acoustic_and_verbal.py')
