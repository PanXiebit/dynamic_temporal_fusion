import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math


def split_long_text(l):
    return l.split('|')

annotations = []
dictionary = {}
with open("/home/dell/xp_workspace/sign-lang/dynamic_attn/Data/slr-phoenix14/train.corpus.csv") as d:
    data = d.readlines()
    for dd in data:
        l = dd.split("|")
        ll = l[-1].split('|')
        for w in ll[0].split(' '):
            w = w.replace('\n', '')
            try:
                dictionary[w] += 1
            except:
                dictionary[w] = 1
#print(dictionary)
count = []
# print(dictionary.items())

for key, value in dictionary.items():
    count.append(value)
print(max(count))
print(min(count))
dictionary = sorted(dictionary.items(), key=lambda item: item[1])
if os.path.exists("Data/output/word_frequency.txt"):
    os.remove("Data/output/word_frequency.txt")
with open("Data/output/word_frequency.txt", 'a+') as t:
    for key, value in dictionary:
        t.write(key)
        t.write('  ')
        t.write(str(value))
        t.write('\n')


plt.hist(count, bins=50)
plt.savefig('word_frequency.png')
# count_norm = [i / float(len(count)) for i in count]
# print(count)
# print(count_norm)
# count_norm_exp = [math.exp(-1 * i) for i in count_norm]
# print(count_norm_exp)