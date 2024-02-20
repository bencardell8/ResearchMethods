import zlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

booksDS = pd.read_csv('books.csv')
print(booksDS.head())