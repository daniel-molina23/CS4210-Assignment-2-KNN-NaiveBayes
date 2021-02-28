#-------------------------------------------------------------------------
# AUTHOR: Daniel Molina
# FILENAME: decision_tree(1).py
# SPECIFICATION: this program runs all 3 datasets for training on a Decision Tree with max-depth of 3 and retrieving the lowest score from all 3 training/testing instances. Each dataset makes 10 iterations.
# FOR: CS 4200- Assignment #2
# TIME SPENT: 1 hour and 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append(row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    X_Dicts = [] # used later on
    for j in range(len(dbTraining[0])-1):
        # go by columns first, except the last column
        s = {}
        count = 1
        for i in range(len(dbTraining)):
            # go by rows now
            tag = dbTraining[i][j]
            if(j==0):
                if(tag not in s):
                    s[tag] = count
                    X.append([s[tag]])
                    count += 1
                else:
                    X.append([s[tag]])
            else:
                if(tag not in s):
                    s[tag] = count
                    X[i].append(s[tag])
                    count += 1
                else:
                    X[i].append(s[tag])
        # first for loop done
        X_Dicts.append(s.copy())

    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    lastCol = len(dbTraining[0])-1
    Y_Dicts = {}
    count = 0
    for i in range(len(dbTraining)):
        # go by rows now from only the last column
        tag = dbTraining[i][lastCol]
        if(tag not in Y_Dicts):
            Y_Dicts[tag] = count
            Y.append(Y_Dicts[tag])
            count += 1
        else:
            Y.append(Y_Dicts[tag])
    

    minAccuracy = float("inf") # initialize minAccuracy to infinity for 10 iterations
    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        # dbTest =
        dbTest = []
        with open('contact_lens_test.csv', 'r') as testfile:
            reader = csv.reader(testfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)
        
        X_test = []
        Y_test = []
        lastCol = len(dbTest[0]) - 1
        for j in range(len(dbTest[0])): # columns first
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            if(j < lastCol): # handle x_test set
                tempSet = X_Dicts[j] # different for every column
                for i in range(len(dbTest)):
                    tag = dbTest[i][j]
                    if(j==0):
                        X_test.append([tempSet[tag]])
                    else:
                        X_test[i].append(tempSet[tag])
            elif(j == lastCol):
                for i in range(len(dbTest)):
                    tag = dbTest[i][j]
                    Y_test.append(Y_Dicts[tag])
        # end for
        # both X_test and Y_test are now transformed


        #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
        #--> add your Python code here
        predictions = clf.predict(X_test)
        # calculating accuracy based on class examples: (TN+TP)/(TN+TP+FP+FN)
        total = 0
        correct = 0
        for i in range(len(predictions)):
            if(predictions[i]==Y_test[i]):
                correct += 1
            total += 1
        accuracy = correct / total


        #find the lowest accuracy of this model during the 10 runs (training and test set)
        minAccuracy = min(minAccuracy,accuracy)

    

    #print the lowest accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that:
         #final accuracy when training on contact_lens_training_1.csv: 0.2
         #final accuracy when training on contact_lens_training_2.csv: 0.3
         #final accuracy when training on contact_lens_training_3.csv: 0.4
    print("final accuracy when training on %s: %f " % (ds,round(minAccuracy,3)))

