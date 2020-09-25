import os
from collections import defaultdict


total_word = defaultdict(int)
rare_word = defaultdict(int)
with open("output/word_frequency.txt", "r") as f:
    # rare word in training dataset
    for line in f:
        content = line.strip().split("  ")
        word = content[0]
        cnt = int(content[1])
        total_word[word] = cnt
        if cnt <= 10:
            rare_word[word] = cnt


test_rare_word = defaultdict(int)
test_rare_word_cnt = 0
with open("output/ref_ctc.txt", "r") as f:
    # rare word in training dataset and also in test data
    for line in f:
        words = line.strip().split()
        for word in words:
            if word in rare_word:
                test_rare_word[word] = rare_word[word]
                test_rare_word_cnt += 1
test_rare_word = sorted(test_rare_word.items(), key=lambda item: item[1], reverse=True)
print("training rare word in test dataset:", len(test_rare_word), test_rare_word)
print("the number of rare in test dataset: ", test_rare_word_cnt)


# how many rare are recognited in ctc_decode
hypo_ctc_words= defaultdict(int)
with open("output/hypo_ctc.txt", "r") as f:
    for line in f:
        words = line.strip().split()
        for word in words:
            hypo_ctc_words[word] += 1

not_recog_cnt = defaultdict(int)
for word, cnt in test_rare_word:
    if word not in hypo_ctc_words:
        # print(word, cnt)
        not_recog_cnt[word] += 1
    else:
        print("Have been recognized: ", word, cnt)
print("not recognized number: ", len(not_recog_cnt))




