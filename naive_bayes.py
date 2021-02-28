#-------------------------------------------------------------------------
# AUTHOR: Daniel Molina
# FILENAME: naive_bayes.py
# SPECIFICATION: This program computes the naive bayes prediction for the datasets weather_training.csv and weather_test.csv
# FOR: CS 4200- Assignment #2
# TIME SPENT: 1 hour for this coding 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data
db = []
with open("weather_training.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: # skip the header
            db.append(row[1:]) # append after the day column


#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# tranforming X
X = []
for j in range(len(db[0])-1): # exclude class
    s = {}
    count = 1
    for i in range(len(db)):
        tag = db[i][j]
        if tag not in s: # create mapping
            s[tag] = count
            count += 1
        # append mapping
        if(j == 0):
            X.append([s[tag]])
        else:
            X[i].append(s[tag])


#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# transforming Y
Y = []
lastCol = len(db[0])-1
for i in range(len(db)):
    temp = db[i][lastCol]
    if(temp == "Yes"):
        Y.append(1)
    else: # NO
        Y.append(0)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file (test data)
testData = []
with open("weather_test.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: # skip the header
            testData.append(row[1:]) # append after the day column

# transform the test data to be digestabile by the model
# tranforming X_test
X_test = []
mapX = []
for j in range(len(testData[0])-1): # exclude class
    s = {}
    count = 1
    for i in range(len(testData)):
        tag = testData[i][j]
        if tag not in s: # create mapping
            s[tag] = count
            count += 1
        # append mapping
        if(j == 0):
            X_test.append([s[tag]])
        else:
            X_test[i].append(s[tag])
    mapX.append(s.copy())

    


#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]
predicted = clf.predict_proba(X_test)

start = 15
for i, row in enumerate(predicted):
    day = "D" + str(start)
    start+=1
    if(row[0] >= 0.75 or row[1] >= 0.75):
        outcome = "No" if(row[0] >= 0.75) else "Yes"
        highest = str(round(row[0],2)) if(row[0] >= 0.75) else str(round(row[1],2))
        print(day.ljust(15) + testData[i][0].ljust(15) + testData[i][1].ljust(15) + testData[i][2].ljust(15) + testData[i][3].ljust(15) + outcome.ljust(15) + highest.ljust(15))