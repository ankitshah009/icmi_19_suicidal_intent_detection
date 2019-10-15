import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

X_data=np.load('X_data.npy')
Y_labels=np.load('Y_labels.npy')
audio_feat_len=14
verbal_feat_len=784
visual_and_acoustic = np.concatenate((X_data[:, :audio_feat_len ], X_data[:,  audio_feat_len + verbal_feat_len : ]), axis=1)
training_features,testing_features,training_target,testing_target=train_test_split(visual_and_acoustic,Y_labels,train_size=0.75,test_size=0.25)


# NOTE: Make sure that the class is labeled 'target' in the data file
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=0.0001, dual=False, penalty="l2")),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=2, max_features=0.05, min_samples_leaf=2, min_samples_split=8, n_estimators=100, subsample=0.65)),
    DecisionTreeClassifier(criterion="gini", max_depth=1, min_samples_leaf=4, min_samples_split=20)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print(classification_report(testing_target, results))

scores1 = cross_val_score(exported_pipeline, X_data, Y_labels, cv=5)
print("all features has 5 fold CV accuracy: " + str(scores1) + "   and mean 5 fold CV accuracy of: " + str(np.mean(scores1)))
scores1_p = cross_val_score(exported_pipeline, X_data, Y_labels, cv=5, scoring='precision')
print("all features has 5 fold CV prec: " + str(scores1_p) + "   and mean 5 fold CV prec of: " + str(np.mean(scores1_p)))
scores1_r = cross_val_score(exported_pipeline, X_data, Y_labels, cv=5, scoring='recall')
print("all features has 5 fold CV recall: " + str(scores1_r) + "   and mean 5 fold CV recall of: " + str(np.mean(scores1_r)))
scores1_f1 = (2 * scores1_p * scores1_r )/ (scores1_p + scores1_r)
print("all features has 5 fold CV f1: " + str(scores1_f1) + "   and mean 5 fold CV f1 of: " + str(np.mean(scores1_f1)))
scores1 = cross_val_score(exported_pipeline, X_data, Y_labels, cv=5, scoring='roc_auc')
print("all features has 5 fold CV AUC: " + str(scores1) + "   and mean 5 fold CV AUC of: " + str(np.mean(scores1)))
#### Best pipeline: DecisionTreeClassifier(GradientBoostingClassifier(LogisticRegression(input_matrix, C=0.0001, dual=False, penalty=l2), learning_rate=0.001, max_depth=2, max_features=0.05, min_samples_leaf=2, min_samples_split=8, n_estimators=100, subsample=0.6500000000000001), criterion=gini, max_depth=1, min_samples_leaf=4, min_samples_split=20)
##0.782608695652174
