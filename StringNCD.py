import zlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Remove truncation of long values (do not run with whole dataset!!)
#pd.set_option('display.max_colwidth', None)

#Read in dataset into booksDS
booksDS = pd.read_csv('books.csv')
#print(booksDS)

#Generates a plot of the distribution of genres
#sns.countplot(x='genre', data=booksDS)
#plt.show()

#See how many data entries for specific genre
#print((booksDS['genre'] == 'crime').sum())

#Prints a specific row from dataset
#print(booksDS.iloc[1])

#x = data, y = labels
X= booksDS['summary'] 
y= booksDS['genre'] 

#Train-test split. Random_state used for reproducability or results. Test_size of 0.2 means test size is 0.2 of population.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)


# printing out train and test sets 
print('X_train : ') 
print(X_train.head()) 
print('') 
print('X_test : ') 
print(X_test.head()) 
print('') 
print('y_train : ') 
print(y_train.head()) 
print('') 
print('y_test : ') 
print(y_test.head())
