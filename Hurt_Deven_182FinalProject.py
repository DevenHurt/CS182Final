#Deven Hurt 
#CS182 Final Project

import numpy
import matplotlib.pyplot as plt 
import pandas as pd 
from patsy import dmatrices
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def logistic_regression(data):
    y,X = dmatrices('Smoking ~ Gender + Age + Race + ServedInMilitary + CountryofBirth + EducationLevel + MaritalStatus + HouseholdIncome + FamilyIncome + ChildrenInHouse + QuantitiyofAlcohol + PerUnitTime + ShortnessOfBreath + Asthma + Exercise + SmokedBefore + AgeStarted', data, return_type = 'dataframe')
    y = numpy.ravel(y)
    model = LogisticRegression()
    model = model.fit(X,y)
    print "The logistic regression model accuracy with smoking questions included is " + str(model.score(X,y))

    y2,X2 = dmatrices('Smoking ~ Gender + Age + Race + ServedInMilitary + CountryofBirth + EducationLevel + MaritalStatus + HouseholdIncome + FamilyIncome + ChildrenInHouse + QuantitiyofAlcohol + PerUnitTime + ShortnessOfBreath + Asthma + Exercise', data, return_type = 'dataframe')
    y2 = numpy.ravel(y)
    model2 = LogisticRegression()
    model2 = model2.fit(X2,y2)
    print "The logistic regression model accuracy without smoking questions included is " + str(model2.score(X2,y2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    modeltest = LogisticRegression()
    modeltest.fit(X_train, y_train)
    predicted = modeltest.predict(X_test)
    print "The logistic regression model accuracy with a training set is " + str(metrics.accuracy_score(y_test, predicted))

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=0)
    modeltest2 = LogisticRegression()
    modeltest2.fit(X_train2, y_train2)
    predicted = modeltest2.predict(X_test2)
    print "The logistic regression model accuracy with a training set and no smoking questions is " + str(metrics.accuracy_score(y_test2, predicted))
    

    return model.score


def nearest_neighbor(data):
    y,X = dmatrices('Smoking ~ Gender + Age + Race + ServedInMilitary + CountryofBirth + EducationLevel + MaritalStatus + HouseholdIncome + FamilyIncome + ChildrenInHouse + QuantitiyofAlcohol + PerUnitTime + ShortnessOfBreath + Asthma + Exercise + SmokedBefore + AgeStarted', data, return_type = 'dataframe')
    y = numpy.ravel(y)

    y2,X2 = dmatrices('Smoking ~ Gender + Age + Race + ServedInMilitary + CountryofBirth + EducationLevel + MaritalStatus + HouseholdIncome + FamilyIncome + ChildrenInHouse + QuantitiyofAlcohol + PerUnitTime + ShortnessOfBreath + Asthma + Exercise', data, return_type = 'dataframe')
    y2 = numpy.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=0)

    knnNoTrainSmoke = KNeighborsClassifier()
    knnNoTrainSmoke.fit(X, y)
    knnNoTrainSmokePred = knnNoTrainSmoke.predict(X)

    knnNoTrainNoSmoke = KNeighborsClassifier()
    knnNoTrainNoSmoke.fit(X2, y2)
    KnnNoTrainNoSmokePred = knnNoTrainNoSmoke.predict(X2)


    knnTrainSmoke = KNeighborsClassifier()
    knnTrainSmoke.fit(X_train, y_train)
    knnTrainSmokePred = knnTrainSmoke.predict(X_test)

    knnTrainNoSmoke = KNeighborsClassifier()
    knnTrainNoSmoke.fit(X_train2, y_train2)
    knnTrainNoSmokePred = knnTrainNoSmoke.predict(X_test2)

    print("% mislabeled via K Nearest Neighbors with smoking questions and no training set = " + (str(float((y != knnNoTrainSmokePred).sum())/float(X.shape[0]))))
    print("% mislabeled via K Nearest Neighbors with smoking questions = " + (str(float((y_test != knnTrainSmokePred).sum())/float(X_test.shape[0]))))
    print("% mislabeled via K Nearest Neighbors without smoking questions= " + (str(float((y_test2 != knnTrainNoSmokePred).sum())/float(X_train2.shape[0]))))
    print("% mislabeled via K Nearest Neighbors without smoking questions and no training set = " + (str(float((y2 != KnnNoTrainNoSmokePred).sum())/float(X2.shape[0]))))

    return 


def naiveBayes(data):
    y,X = dmatrices('Smoking ~ Gender + Age + Race + ServedInMilitary + CountryofBirth + EducationLevel + MaritalStatus + HouseholdIncome + FamilyIncome + ChildrenInHouse + QuantitiyofAlcohol + PerUnitTime + ShortnessOfBreath + Asthma + Exercise + SmokedBefore + AgeStarted', data, return_type = 'dataframe')
    y = numpy.ravel(y)

    y2,X2 = dmatrices('Smoking ~ Gender + Age + Race + ServedInMilitary + CountryofBirth + EducationLevel + MaritalStatus + HouseholdIncome + FamilyIncome + ChildrenInHouse + QuantitiyofAlcohol + PerUnitTime + ShortnessOfBreath + Asthma + Exercise', data, return_type = 'dataframe')
    y2 = numpy.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=0)


    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_train)
    finalTest = gnb.predict(X_test)

    gnbNoTrain = GaussianNB
    y_predNoTrain = gnb.fit(X, y).predict(X)

    gnb2 = GaussianNB()
    y_pred2 = gnb2.fit(X_train2, y_train2).predict(X_train2)
    finalTest2 = gnb2.predict(X_test2)

    gnbNoTrain2 = GaussianNB
    y_predNoTrain2 = gnb.fit(X2, y2).predict(X2)

    #do with and without smoking
    #print("Number of mislabeled points in training data out of a total %d points : %d" % (X_train.shape[0],(y_train != y_pred).sum()))
    print("% mislabeled via Bayes with smoking questions and no training set = " + (str(float((y != y_predNoTrain).sum())/float(X.shape[0]))))
    print("% mislabeled via Bayes with smoking questions = " + (str(float((y_test != finalTest).sum())/float(X_test.shape[0]))))
    print("% mislabeled via Bayes without smoking questions= " + (str(float((y_test2 != finalTest2).sum())/float(X_test2.shape[0]))))
    print("% mislabeled via Bayes without smoking questions and no training set = " + (str(float((y2 != y_predNoTrain2).sum())/float(X2.shape[0]))))


    return 

data = pd.read_csv('BaselineData.csv', index_col=0)
smoker = data['Smoking']
data.head()

data.shape
print data.shape

#print smoker
smokerCount = 0 
for person in smoker[2:]:
    if person == 1:
        smokerCount += 1 

print "The total number of smokers is " + str(smokerCount)

logisticSuccess = logistic_regression(data)
neighborSuccess = nearest_neighbor(data)
bayesSuccess = naiveBayes(data)