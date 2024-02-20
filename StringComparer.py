import zlib

#Data to be classified
#seq1 = "g7e87urhghh488guww974jb9bih86h3j8duishisuhvihvspiuhsvasdvsd".encode('latin-1')
#seq2 = "g7e87urhghh488guww974jb9bih86h3j8duishisuhvihvspiuhsvasdvsd".encode('latin-1')

#seq1 = "1010101010101010101010101010101010101".encode('latin-1')
#seq2 = "0101010101010101010101010101010101010".encode('latin-1')

seq1 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum".encode('latin-1')
seq2 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum".encode('latin-1')


#Compresses strings, retrieves lengths
seq1_compressed = len(zlib.compress(seq1))
seq2_compressed = len(zlib.compress(seq2))
print(f'Uncompressed size of sequence 1 = {seq1}')
print(f'Compressed size of sequence 1 = {seq1_compressed}')
print(f'Compressed size of sequence 2 = {seq2_compressed}')

#Compressed size of concatentated strings (presumably one after the other)
seqs_compressed = len(zlib.compress(seq1 + seq2))
print(f'Compressed size of sequence 1 + 2 (concatenated) = {seqs_compressed}')

ncd = (seqs_compressed - min(seq1_compressed,seq2_compressed)) / max(seq1_compressed,seq2_compressed)
print(f'NCD(seq1, seq2) = {ncd}') #Prints normalized compression distance