# -*- coding: utf-8 -*-




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataset = pd.read_csv('data.csv')

X = dataset.drop(labels=['label'], axis = 1)
y = dataset['label']

from sklearn.model_selection import train_test_split

X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0,stratify =y)

from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
knn = KNeighborsClassifier(n_neighbors=4)
from sklearn import svm
clf = svm.SVC(kernel='rbf',C=10,gamma=0.1)

sfs = SFS(clf, 
          k_features=(2,25), 
          forward=True, 
          floating=False,
          scoring='accuracy',
          cv=5)

sfs.fit(X_train,y_train)

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sfs.get_metric_dict(),kind='std_dev')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()    
plt.show()

sfs.subsets_

y1 = pd.DataFrame(X_train['230944_at'])
y2 = pd.DataFrame(X_train['1564229_at'])
y3 = pd.DataFrame(X_train['227925_at'])
y4 = pd.DataFrame(X_train['227919_at'])
y5 = pd.DataFrame(X_train['239113_at'])
y6 = pd.DataFrame(X_train['231954_at'])
y7 = pd.DataFrame(X_train['228839_s_at'])
y8 = pd.DataFrame(X_train['229090_at'])
y9 = pd.DataFrame(X_train['1558750_a_at'])
y10 = pd.DataFrame(X_train['240115_at'])
y11 = pd.DataFrame(X_train['216596_at'])
X1 = pd.concat([y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11],axis=1)



z1 = pd.DataFrame(X_test['230944_at'])
z2 = pd.DataFrame(X_test['1564229_at'])
z3 = pd.DataFrame(X_test['227925_at'])
z4 = pd.DataFrame(X_test['227919_at'])
z5 = pd.DataFrame(X_test['239113_at'])
z6 = pd.DataFrame(X_test['231954_at'])
z7 = pd.DataFrame(X_test['228839_s_at'])
z8 = pd.DataFrame(X_test['229090_at'])
z9 = pd.DataFrame(X_test['1558750_a_at'])
z10 = pd.DataFrame(X_test['240115_at'])
z11 = pd.DataFrame(X_test['216596_at'])
X2 = pd.concat([z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11],axis=1)
r2 = y_test

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score,recall_score

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
clf.fit(X1, y_train)
y_pred_RF = clf.predict(X2)
print(y_pred_RF)
print('Accuracy: ', accuracy_score(y_test, y_pred_RF))
print('report : ', classification_report(y_test,y_pred_RF))
print(confusion_matrix(y_test,y_pred_RF))
print(precision_score(y_test, y_pred_RF, average='macro'))
print(recall_score(y_test, y_pred_RF, average='macro'))
print(f1_score(y_test, y_pred_RF, average='macro'))

from sklearn import svm

clf2 = svm.SVC(kernel='rbf',C=10,gamma=1,probability=True)
clf2.fit(X1, y_train)
y_pred = clf2.predict(X2)
print(y_pred)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('report : ', classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='macro'))

clf4 = svm.SVC(kernel='linear', probability=True)
clf4.fit(X1, y_train)
y_pred = clf4.predict(X2)
print(y_pred)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('report : ', classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='macro'))

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = clf.predict_proba(X2)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# calculate the fpr and tpr for all thresholds of the classification
probs1 = clf2.predict_proba(X2)
preds1 = probs1[:,1]
fpr1, tpr1, threshold1 = metrics.roc_curve(y_test, preds1)
roc_auc1 = metrics.auc(fpr1, tpr1)


# calculate the fpr and tpr for all thresholds of the classification
probs2 = clf4.predict_proba(X2)
preds2 = probs2[:,1]
fpr2, tpr2, threshold2 = metrics.roc_curve(y_test, preds2)
roc_auc2 = metrics.auc(fpr2, tpr2)

roc_auc2

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristics')
plt.plot(fpr, tpr, 'c:', label = 'Random Forest' )
plt.plot(fpr1, tpr1, 'r:', label = 'SVM(rbf)' )
plt.plot(fpr2, tpr2, 'g:', label = 'SVM(linear)')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()















