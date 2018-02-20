


import warnings
warnings.filterwarnings('ignore')

#for data preprocessing 
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm

#for differnt classifier for testing 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#for validation 
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict 

#for visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


df = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv', header = None )

df.head()


df.info()


print('Size of traning data:' + str(len(df)))
print('Size of test data:' + str(len(test)))


cor = df.corr()

print (df.groupby(['buying_price','popularity'])['popularity'].count())


print (df.groupby(['maintainence_cost','popularity'])['popularity'].count())

print (df.groupby(['maintainence_cost','popularity'])['popularity'].count())


print (df.groupby(['number_of_doors','popularity'])['popularity'].count())


print (df.groupby(['number_of_seats','popularity'])['popularity'].count())


print (df.groupby(['luggage_boot_size','popularity'])['popularity'].count())

print (df.groupby(['safety_rating','popularity'])['popularity'].count())


X = np.array(df.drop(['popularity'],1))
#labeled data
Y = np.array(df['popularity'])  


scale = StandardScaler()
#fitting the scaler
X = scale.fit_transform(X)




X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.01,random_state= None)

classifiers = {
    'KKN' : KNeighborsClassifier(),
    'SVC': svm.SVC(),
    'DecisionTree':DecisionTreeClassifier(),
    'RandomForest':RandomForestClassifier(),
    'GradientBoosting':GradientBoostingClassifier(),
    'MLP':MLPClassifier(),
    'AdaBoost':AdaBoostClassifier(),
    'GaussianNB':GaussianNB(),
    'QuadraticDiscriminant':QuadraticDiscriminantAnalysis()}    


kfold = KFold(n_splits=10, random_state=22)
xyz=[]
accuracy=[]
std=[]
f1_s = []

print("Selecting classifiers")

for name,clf in classifiers.items():
    cv_result = cross_val_score(clf,X_train,Y_train, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
    
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    f1_s.append(f1_score(Y_test, Y_pred, average=None))
    
new_clf_dataframe=pd.DataFrame({'mean_accuracy':xyz,'std':std,'F1-scores':f1_s},index=classifiers)      

test = pd.read_csv('data/test.csv', header = None )
X_t = np.array(test)
X_t = scale.transform(X_t)


classifiers = {
    'KKN' : KNeighborsClassifier(),
    'SVC': svm.SVC(),
    'DecisionTree':DecisionTreeClassifier(),
    'RandomForest':RandomForestClassifier(),
    'GradientBoosting':GradientBoostingClassifier()}    


def avg_clf(data):
    cnt = np.zeros([len(data),4])
    for name,clf in classifiers.items():
        # X and Y are orginal traning data 
        clf.fit(X,Y)
        Y_pred = clf.predict(X_t)
        for i,x in enumerate(Y_pred):
            cnt[i][x-1] += 1
    out = []
    for i in  range(len(cnt)):
        my_list = cnt[i].tolist()
        max_value = max(my_list)
        max_index = my_list.index(max_value) + 1
        out.append(max_index)
    return out


# In[26]:


def tune():
    mx = 0
    CC = 1
    gg = .01
    kfold = KFold(n_splits=10, random_state=22)
    for c in range(1,5):
        for gm in (1,10):
            clf = svm.SVC(kernel='rbf',C=c,gamma=gm/10)
            cv_result = cross_val_score(clf,X_train,Y_train, cv = kfold,scoring = "accuracy")
            print(c,gm/10,cv_result.mean())
            if(mx<cv_result.mean()):
                mx = cv_result.mean()
                CC = c
                gg = gm/10
    return CC,gg,mx



Y_pred  = avg_clf(X_t)
out =pd.DataFrame(Y_pred)




clf = svm.SVC(kernel='rbf',C=25,gamma=.6)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_t)
out =pd.DataFrame(Y_pred)

print("Predicting....")

out.to_csv('predictions.csv',index=None,header=None)

