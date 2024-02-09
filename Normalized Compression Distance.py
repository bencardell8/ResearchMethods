import zlib

#Data to be classified
seq1 = "ABCDEFGHIJK".encode('latin-1')
seq2 = "AABBCCDDEEFF".encode('latin-1')

z1 = len(zlib.compress(seq1))
z2 = len(zlib.compress(seq2))
print(f'Compressed size: Z(seq1) = {z1}, Z(seq2)={z2}')

z12 = len(zlib.compress(seq1 + seq2))
print(f'Compressed size: Z(seq1+seq2) = {z12}')

ncd = (z12 - min(z1,z2)) / max(z1,z2)
print(f'NCD(seq1, seq2) = {ncd}') #Prints normalized compression distance