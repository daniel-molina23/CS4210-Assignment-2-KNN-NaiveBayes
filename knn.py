#-------------------------------------------------------------------------
# AUTHOR: Daniel Molina
# FILENAME: knn.py
# SPECIFICATION: Program uses the KNN algorithm to check the error rate based on Leave one out cross validation. Using 1NN for checking/testing
# FOR: CS 4200- Assignment #2
# TIME SPENT: 30-45 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)


# for all iterations
incorrectPredictions = 0
#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]
    X = []
    for j in range(len(db)):
        X.append([int(db[j][0]),int(db[j][1])])
    X_test = [X.pop(i)] # ith row must go

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]
    Y = []
    for j in range(len(db)):
        if(db[j][2]=="-"):
            Y.append(0)
        else: # '+'
            Y.append(1)
    Y_test = [Y.pop(i)] # ith class must go

    ####Test Sample is X_test and Y_test#####


    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction.
    # print("X_Test = ", X_test)
    class_predicted = clf.predict(X_test)

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    if(class_predicted != Y_test):
        incorrectPredictions += 1
    
    # print("Class Predicted = ", class_predicted, ", and actual = ", Y_test)

#print the error rate
total = len(db)
print("Error rate for 1NN algorithm = %.3f" % (incorrectPredictions/total))






