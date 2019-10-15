import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# NOTE: Make sure that the class is labeled 'target' in the data file
X_data=np.load('X_data.npy')
Y_labels=np.load('Y_labels.npy')
audio_feat_len=14
verbal_feat_len=784
visual_only = X_data[:,  audio_feat_len + verbal_feat_len : ]
training_features,testing_features,training_target,testing_target=train_test_split(visual_only,Y_labels,train_size=0.75,test_size=0.25)
#tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
#features = tpot_data.drop('target', axis=1).values
#training_features, testing_features, training_target, testing_target = \
#            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.804395604396
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.2, min_samples_leaf=3, min_samples_split=11, n_estimators=100)),
    ZeroCount(),
    SelectFwe(score_func=f_classif, alpha=0.024),
    ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.65, min_samples_leaf=3, min_samples_split=9, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
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

### Best pipeline: GradientBoostingClassifier(Nystroem(PolynomialFeatures(input_matrix, degree=2, include_bias=False, interaction_only=False), gamma=0.1, kernel=cosine, n_components=5), learning_rate=0.5, max_depth=6, max_features=0.9500000000000001, min_samples_leaf=16, min_samples_split=17, n_estimators=100, subsample=0.8500000000000001)
### 0.782608695652174
