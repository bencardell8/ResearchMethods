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


min_count = min(y.value_counts())

#trim dataframe down to balanced number of data
balanced_booksDS = pd.concat([booksDS[booksDS['genre'] == label].sample(min_count) for label in booksDS['genre'].unique()])


#Train-test split. Random_state used for reproducability of results_zlib. Test_size of 0.2 means test size is 0.2 of population.
X_train, X_test, y_train, y_test = train_test_split(booksDS[['summary']], booksDS['genre'], test_size=0.2, random_state=1, stratify=booksDS['genre'])


# printing out train and test sets 
# print('X_train : ') 
# print(X_train.head()) 
# print('') 
# print('X_test : ') 
# print(X_test.head()) 
# print('') 
# print('y_train : ') 
# print(y_train.head()) 
# print('') 
# print('y_test : ') 
# print(y_test.head())


#print(balanced_booksDS.head())
#print(y.value_counts())
#print(y_train.value_counts())
#print(y_train.iloc[1])
#print(balanced_booksDS['genre'].value_counts())
#print(booksDS['genre'].value_counts())

#Prints train and test data counts
#print(y_train.value_counts())
#print(y_test.value_counts())

#print(y_train.iloc[1])

#seq1 = X_train.iloc[1]
#print(seq1)

#X_train.loc[X_train['summary'] == 'a', 'Port'].values[0]


#load data into a DataFrame object:

data = {
  "Train_Genre": [],
  "Test_Genre": [],
  "NCD": []
}

NCDs = pd.DataFrame(data)
#print(NCDs)

for i in range(len(X_train)):
    #print(X_train.iloc[i,:].to_string(header=False, index=False))
    ncd_list = []

    seq1 = (X_train.iloc[i].to_string(header=False, index=False)).encode('utf-8')
    seq2 = (X_test.iloc[7].to_string(header=False, index=False)).encode('utf-8')

    seq1_compressed = len(zlib.compress(seq1))
    seq2_compressed = len(zlib.compress(seq2))
    seqs_compressed = len(zlib.compress(seq1 + seq2))
    ncd = (seqs_compressed - min(seq1_compressed,seq2_compressed)) / max(seq1_compressed,seq2_compressed)

    #print(y_train.iloc[i], f'NCD(seq1, seq2) = {ncd}')

    #values = {'X_train_Genre': y_train, 'X_test_Genre': y_test, 'NCD': ncd}
    #NCDs = NCDs.append(values, ignore_index = True)  

    NCDs.loc[len(NCDs.index)] = [y_train.iloc[i], y_test.iloc[7], ncd]  

    i+=1

#print(NCDs)
#print(NCDs.min(axis=0))
smallest_rows = NCDs.nsmallest(10, "NCD", keep="all")
print(NCDs.loc[NCDs['NCD'] == NCDs['NCD'].max()])
#print(smallest_rows)
#print("Test data genre: " + y_test.iloc[1])    

#print(X_train.iloc[1:3].to_string(header=False, index=True))