import zlib

#Data to be classified
seq1 = "ABCDEFGHIJK".encode('latin-1')
seq2 = "ABCDEFGHIJK".encode('latin-1')

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