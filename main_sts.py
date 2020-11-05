import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read training data
sts_train = open('/Users/dz/Desktop/MSCourses/CMPE255DataMining/Project/datasets/sts-train.csv', 'r')

new_sts_train = []

for line in sts_train:
    num_strings = line.split('\t')
    numbers = (n for n in num_strings)
    new_sts_train.append(numbers)

# Create columns for each elements
new_sts_train = pd.DataFrame(new_sts_train, columns = ['Col1','Col2','Col3','Col4','Col5','Sentence_1','Sentence_2','Col8','Col9'])
# Drop redundancy columns
new_sts_train.drop(columns = ['Col1','Col2','Col3','Col4','Col5','Col8','Col9'], inplace = True)
# Display fully sentence in the column
pd.set_option('display.max_colwidth', None)
# Clean data
new_sts_train['Sentence_2'].replace('\n', ' ', regex = True, inplace = True)

print (new_sts_train.head())

# Check and count the sum of Null or NaN value in each column. If print 0, means No null or nan value in column
print (new_sts_train['Sentence_1'].isnull().sum())
print (new_sts_train['Sentence_2'].isnull().sum())

# Statistics data, which are strings, with count, unique, top, and freq
print (new_sts_train.describe())

# Convert string to integer in training dataset
le = LabelEncoder()
for i in new_sts_train:
    new_sts_train[i] = le.fit_transform(new_sts_train[i])

print (new_sts_train)

# Statistics data, which convert string to integer, with count, unique, top, and freq in training dataset
print (new_sts_train.describe())

# Read testing data
sts_test = open('/Users/dz/Desktop/MSCourses/CMPE255DataMining/Project/datasets/sts-test.csv', 'r')

new_sts_test = []

for line in sts_test:
    num_strings = line.split('\t')
    numbers = (n for n in num_strings)
    new_sts_test.append(numbers)


# Create columns for each elements
new_sts_test = pd.DataFrame(new_sts_test, columns = ['Col1','Col2','Col3','Col4','Col5','Sentence_1','Sentence_2','Col8','Col9'])
# Drop redundancy columns
new_sts_test.drop(columns = ['Col1','Col2','Col3','Col4','Col5','Col8','Col9'], inplace = True)
# Display fully sentence in the column
pd.set_option('display.max_colwidth', None)
# Clean data
new_sts_test['Sentence_2'].replace('\n', ' ', regex = True, inplace = True)
print (new_sts_test.head())

# Check and count the sum of Null or NaN value in each column. If print 0, means No null or nan value in column
print (new_sts_test['Sentence_1'].isnull().sum())
print (new_sts_test['Sentence_2'].isnull().sum())

# Statistics data, which are strings, with count, unique, top, and freq in test dataset
print (new_sts_test.describe())

# Convert string to integer in testing dataset
le = LabelEncoder()
for i in new_sts_test:
    new_sts_test[i] = le.fit_transform(new_sts_test[i])

print (new_sts_test)

# Statistics data, which convert string to integer, with count, unique, top, and freq in test dataset
print (new_sts_test.describe())

