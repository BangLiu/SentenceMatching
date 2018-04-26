# -*- coding: utf-8 -*-
import gensim
import pandas as pd
from order_wassertein_distance import *
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr


# load word vectors
model = gensim.models.Word2Vec.load_word2vec_format(
    '../../../data/raw/Google-w2v/GoogleNews-vectors-negative300.bin',
    binary=True)

word = "OOV"
if word in model.vocab:
    print "has and"
    print model[word]
else:
    print "don't have and"


# iterate over each line to get text pair
fin_name = "../../../data/processed/sick2014/SICK_train.txt"
df = pd.read_csv(fin_name, sep="|")
sentence_A_normalized = df["sentence_A_normalized"].values
sentence_B_normalized = df["sentence_B_normalized"].values
print len(sentence_A_normalized)
print len(sentence_B_normalized)


def replace_OOV(text, model):
    text = text.split()
    new_text = []
    for i in range(len(text)):
        if text[i] in model.vocab:
            new_text.append(text[i])
    new_text = " ".join(new_text)
    return new_text


# use OWD
lambda1s = [20, 0.2, 1]   # seems lambda1 = 2 is a good choice
lambda2s = [0.03] # lambda2 cannot be 0 or too small. seems lambda2 = 0.03 is good
sigmas = [100] # sigma 1000 will have error. I think fix sigma = 10 ~ 100 is good enough.
for idx_lam1 in range(len(lambda1s)):
    for idx_lam2 in range(len(lambda2s)):
        for idx_sigma in range(len(sigmas)):
            lambda1 = lambda1s[idx_lam1]
            lambda2 = lambda2s[idx_lam2]
            sigma = sigmas[idx_sigma]
            fout_name = "../../../data/output/sick2014/SICK_train.score.s_owd."\
                + str(lambda1) + "_" + str(lambda2) + "_" + str(sigma) + ".txt"
            fout = open(fout_name, "w")
            for i in range(len(sentence_A_normalized)):
                # print i
                snt1 = replace_OOV(sentence_A_normalized[i], model)
                snt2 = replace_OOV(sentence_B_normalized[i], model)
                score = owd_snt_pair(snt1, snt2, model, lambda1, lambda2, sigma, 200)
                fout.write(str(score) + "\n")
            fout.close()

            y_pred = np.loadtxt(fout_name)
            df = pd.read_csv(fin_name, sep="|")
            y = df["relatedness_score"].values.astype(float)
            print "lambda1 = " + str(lambda1)
            print "lambda2 = " + str(lambda2)
            print "sigma = " + str(sigma)
            print pearsonr(y, y_pred)
            # plt.plot(y, y_pred, 'o')
            # plt.show()


# # use WMD
# fout_name = "../../../data/output/sick2014/SICK_train.score.wmd.txt"
# fout = open(fout_name, "w")
# for i in range(len(sentence_A_normalized)):
#     # print i
#     snt1 = replace_OOV(sentence_A_normalized[i], model)
#     snt2 = replace_OOV(sentence_B_normalized[i], model)
#     score = model.wmdistance(snt1.split(), snt2.split())
#     fout.write(str(score) + "\n")
# fout.close()

# y_pred = np.loadtxt(fout_name)
# df = pd.read_csv(fin_name, sep="|")
# y = df["relatedness_score"].values.astype(float)
# print "use WMD"
# print pearsonr(y, y_pred)


# fit polynomial and calculate again
lambda1 = 2
lambda2 = 0.03
sigma = 32
fout_name = "../../../data/output/sick2014/SICK_train.score.s_owd."\
            + str(lambda1) + "_" + str(lambda2) + "_" + str(sigma) + ".txt"
y_pred = np.loadtxt(fout_name)
df = pd.read_csv(fin_name, sep="|")
y = df["relatedness_score"].values.astype(float)
y_pred_fit = np.poly1d(np.polyfit(y_pred, y, 8))
print "lambda1 = " + str(lambda1)
print pearsonr(y, y_pred_fit(y_pred))
print "mse: " + str(np.mean((y - y_pred_fit(y_pred)) ** 2))
# plt.plot(y, y_pred, 'o')
# plt.show()

# fout_name = "../../../data/output/sick2014/SICK_train.score.s_wmd.txt"
# y_pred = np.loadtxt(fout_name)
# df = pd.read_csv(fin_name, sep="|")
# y = df["relatedness_score"].values.astype(float)
# y_pred_fit = np.poly1d(np.polyfit(y_pred, y, 8))
# print "wmd"
# print pearsonr(y, y_pred_fit(y_pred))
# print "mse: " + str(((y - y_pred_fit(y_pred)) ** 2).mean(axis=0))

# fin_name = "../../../data/processed/sick2014/SICK_train.txt"
# fout_name = "../../../data/output/sick2014/SICK_train.score.s_owd." + str(2) + ".txt"
# y_pred = np.loadtxt(fout_name)
# df = pd.read_csv(fin_name, sep="|")
# y = df["relatedness_score"].values.astype(float)
# print "lambda1 = " + str(2)
# print pearsonr(y, y_pred)
# plt.plot(y, y_pred, 'o')
# plt.show()
