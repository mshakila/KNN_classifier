##############  ASSIGNMENT KNN using zoo dataset

########### Step 1. Data Preprocessing 

#### Importing the libraries.
import  pandas as pd
import numpy as np
import random # for setting seed 
from sklearn.model_selection import train_test_split # to split data 
from sklearn.neighbors import KNeighborsClassifier as KNC # to build knn model
import matplotlib.pyplot as plt # for visualizations

zoo = pd.read_csv('E:\\KNN\\Zoo.csv')

### Business problem : To classify the animals into categdories (Types)

zoo.shape
zoo.columns
zoo['animal name'].unique()
# this column will be removed later, since cannot be used for analysis
# 101 recorda and 18 variables
zoo.dtypes
# all variables are binary in nature, shown as integers. But Type should
# be factor, so converting it into factors
zoo['type'] = zoo['type'].astype('category')

zoo.type.unique()
zoo.type.value_counts()
# 1, 4, 2, 7, 6, 5, 3      there are 7 types

# splitting the data into train:test 70:30
random.seed(123)
train, test = train_test_split(zoo, test_size = 0.3)

##################### Step 2. KNN Classification 

###### Building model using k=3
knn = KNC(n_neighbors = 3)

x = train.iloc[:,1:17]
y = train.iloc[:,17]

knn.fit(x, y)
# train accuracy
train_acc = np.mean(knn.predict(train.iloc[:,1:17]) ==train.iloc[:,17]) # 98.57
# test accuracy
test_acc = np.mean(knn.predict(test.iloc[:,1:17]) == test.iloc[:,17]) # 87.10

####### Running the model for various values of k
accuracy = []

for i in range(1,50,1):
    knn = KNC(n_neighbors =i)
    knn.fit(train.iloc[:,1:17], train.iloc[:,17])
    train_acc = np.mean(knn.predict(train.iloc[:,1:17]) == train.iloc[:,17])
    test_acc = np.mean(knn.predict(test.iloc[:,1:17]) == test.iloc[:,17])
    accuracy.append([train_acc, test_acc])
    
%matplotlib qt    
plt.plot(np.arange(1,50,1), [i[0] for i in accuracy], "bo-")
plt.plot(np.arange(1,50,1), [i[1] for i in accuracy], "ro-")
plt.legend(['train_accuracy','test_accuracy'])

'''
for k=1, we have test and train accuracy as 100%, but if choose smaller k
value, noise will impact the result. small changes in training set, will cause
large changes in classification. 
hence we are using training set to run the model and validation set to
validate our results. 
(If take higher values of k, will lead to high model bias by predicting the
most frequent class.)
another approach is taking k = sqrt(n), here it will be around 10.
when k=8, train accuracy is 80% and test accuracy is 87%
when k=9 and k=10, both accuracies are almost same to 80%
'''


######## using train and validation set in the ratio 60:40
random.seed(123)
train, test = train_test_split(zoo, test_size = 0.4)
accuracy = []
for i in range(1,30,1):
    knn = KNC(n_neighbors =i)
    knn.fit(train.iloc[:,1:17], train.iloc[:,17])
    train_acc = np.mean(knn.predict(train.iloc[:,1:17]) == train.iloc[:,17])
    test_acc = np.mean(knn.predict(test.iloc[:,1:17]) == test.iloc[:,17])
    accuracy.append([train_acc, test_acc])
    
plt.plot(np.arange(1,30,1), [i[0] for i in accuracy], "bo-")
plt.plot(np.arange(1,30,1), [i[1] for i in accuracy], "ro-")
plt.legend(['train_accuracy','test_accuracy'])
''' when k=6 to 8, train acc is 85% and and test acc is 78% 
when k=9,10, both accuracies lie btw 78% to 82%'''


######## using full data to train the model
accuracy = []
for i in range(1,30,1):
    knn = KNC(n_neighbors =i)
    knn.fit(zoo.iloc[:,1:17], zoo.iloc[:,17])
    train_acc = np.mean(knn.predict(zoo.iloc[:,1:17]) == zoo.iloc[:,17])
    accuracy.append([train_acc])
    
plt.plot(np.arange(1,30,1), [i[0] for i in accuracy], "bo-")
plt.legend(['train_accuracy'])
''' till k=7, acc is very high, more than 90% 
from k=8 to 10, it lies btw 80 to 90% '''


# using train and validation set in the ratio 75:25
random.seed(123)
train, test = train_test_split(zoo, test_size = 0.25)
accuracy = []
for i in range(1,30,1):
    knn = KNC(n_neighbors =i)
    knn.fit(train.iloc[:,1:17], train.iloc[:,17])
    train_acc = np.mean(knn.predict(train.iloc[:,1:17]) == train.iloc[:,17])
    test_acc = np.mean(knn.predict(test.iloc[:,1:17]) == test.iloc[:,17])
    accuracy.append([train_acc, test_acc])
    
plt.plot(np.arange(1,30,1), [i[0] for i in accuracy], "bo-")
plt.plot(np.arange(1,30,1), [i[1] for i in accuracy], "ro-")
plt.legend(['train_accuracy','test_accuracy'])
''' when k=9 to 14 train acc is 79% and and test acc is 77% '''


# using train and validation set in the ratio 70:30 but changing the seed
random.seed(1234)
train, test = train_test_split(zoo, test_size = 0.30)
accuracy = []
for i in range(1,30,1):
    knn = KNC(n_neighbors =i)
    knn.fit(train.iloc[:,1:17], train.iloc[:,17])
    train_acc = np.mean(knn.predict(train.iloc[:,1:17]) == train.iloc[:,17])
    test_acc = np.mean(knn.predict(test.iloc[:,1:17]) == test.iloc[:,17])
    accuracy.append([train_acc, test_acc])
    
plt.plot(np.arange(1,30,1), [i[0] for i in accuracy], "bo-")
plt.plot(np.arange(1,30,1), [i[1] for i in accuracy], "ro-")
plt.legend(['train_accuracy','test_accuracy'])
''' when k=8 to 13 train acc is around 80% and and test acc is 77% '''


####### solving using Multinomial Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

## splitting data into train/ test
train, test = train_test_split(zoo, test_size = 0.3, random_state=123)

### fitting the model
log_reg = LogisticRegression(solver = 'lbfgs')
log_reg.fit(train.iloc[:,1:17], train.iloc[:,17])

# using the model to make predictions with the test data and train data
test_pred = log_reg.predict(test.iloc[:,1:17])
train_pred = log_reg.predict(train.iloc[:,1:17])

test.iloc[:,17]
test_pred

# how did our model perform?
count_misclassified_test = (test.iloc[:,17] != test_pred).sum()
test_accuracy = metrics.accuracy_score(test.iloc[:,17], test_pred)

count_misclassified_train = (train.iloc[:,17] != train_pred).sum()
train_accuracy = metrics.accuracy_score(train.iloc[:,17], train_pred)

print('Misclassified samples of validation set: {}'.format(count_misclassified_test))
print('Test Accuracy: {:.2f}'.format(test_accuracy))

print('Misclassified samples of training set: {}'.format(count_misclassified_train))
print('Train accuracy: {:.2f}'.format(train_accuracy))
# train accuracy is 100% and test accuracy is 87% (with 4/31 misclassified samples)


''' logistic regression can perform well when have large datasets.
KNN is easy to implement and is non-parametric (free of distribution). If more 
variables results may not be accurate.
Both KNN and logistic regression are influenced by outliers. '''

'''
CONCLUSIONS
We have zoo dataset. There are around 100 recorda and  17 variables. The 
animals are categorised into 7 classes. 
The independent variables are all binary in nature, hence feature scaling
is not done. 
We have to classify the animals into various classes. We have used KNN algorithm
to predict the correct classes.
To choose optimal k value, we have used training and validation sets and
compared their accuracies.
We have set different ratios of training and validation sets, we have 
set different random.seeds to improve the model performance.
We have also used Multinomial Logistic Regression to predict the classes.
'''




