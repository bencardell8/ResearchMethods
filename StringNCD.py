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

