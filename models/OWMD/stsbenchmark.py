# -*- coding: utf-8 -*-
import gensim
import pandas as pd
from order_wassertein_distance import *
from scipy.stats.stats import pearsonr


def load_w2v(fin, type):
    model = {}
    if type == "Google":
        model = gensim.models.KeyedVectors.load_word2vec_format(fin, binary=True)
    elif type == "Glove":  # converted glove to word2vec format
        model = gensim.models.KeyedVectors.load_word2vec_format(fin, binary=True)
    elif type == "Tencent":
        with open(fin, "r") as fread:
            for line in fread.readlines():
                line_list = line.strip().split(" ")
                word = line_list[0]
                word_vec = np.fromstring(" ".join(line_list[1:]),
                                         dtype=float, sep=" ")
                model[word] = word_vec
    else:
        print "type must be Glove or Google or Tencent."
        sys.exit(1)
    print type
    return model


def replace_OOV(text, model):
    # print "text split is: ", text
    text = text.split()
    new_text = []
    for i in range(len(text)):
        if text[i] in model.vocab:
            new_text.append(text[i])
    new_text = " ".join(new_text)
    return new_text


def sts_owd(fin_name, col1, col2, col_score, fout_name,
            model, lambda1, lambda2, sigma, max_iter,
            metric):
    df = pd.read_csv(fin_name, sep="\t")
    text1 = df[col1].values
    text2 = df[col2].values
    print text1
    print text2

    fout = open(fout_name, "w")
    for i in range(len(text1)):
        snt1 = replace_OOV(text1[i], model)
        snt2 = replace_OOV(text2[i], model)
        score = owd_snt_pair(snt1, snt2, model,
                             lambda1, lambda2, sigma, max_iter,
                             metric)
        fout.write(str(score) + "\n")
    fout.close()


def sts_wmd(fin_name, col1, col2, col_score, fout_name, model):
    df = pd.read_csv(fin_name, sep="\t")
    text1 = df[col1].values
    text2 = df[col2].values

    fout = open(fout_name, "w")
    for i in range(len(text1)):
        snt1 = replace_OOV(text1[i], model)
        snt2 = replace_OOV(text2[i], model)
        score = model.wmdistance(snt1.split(), snt2.split())
        fout.write(str(score) + "\n")
    fout.close()


def eval(ftrue, col_score, fpred):
    y_pred = np.loadtxt(fpred)
    df = pd.read_csv(ftrue, sep="\t")
    y = df[col_score].values.astype(float)
    pearson_score = pearsonr(y, y_pred)
    print pearson_score
    return pearson_score


def fit_polynomial(fpred, ftrue, order):
    # fit polynomial and calculate again
    y_pred = np.loadtxt(fpred)
    df = pd.read_csv(ftrue, sep="\t")
    y = df["score"].values.astype(float)
    y_pred_fit = np.poly1d(np.polyfit(y_pred, y, order))
    print "poly fit order: ", order
    print pearsonr(y, y_pred_fit(y_pred))


if __name__ == "__main__":
    exp = "stsbenchmark"

    if exp == "stsbenchmark":
        # EXP1: stsbenchmark
        #fw2v = "../../../data/raw/Google-w2v/GoogleNews-vectors-negative300.bin"
        #w2v_type = "Google"
        fw2v = "../../../data/raw/Glove-EN-300d/glove.840B.300d.bin"
        w2v_type = "Glove"
        col1_nor = "sentence1_normalized"
        col2_nor = "sentence2_normalized"
        col1 = "sentence1"
        col2 = "sentence2"
        col_score = "score"
        lambda1 = 1
        lambda2 = 0.03
        sigma = 10
        max_iter = 20
        model = load_w2v(fw2v, w2v_type)

        # train
        ftextpair = "../../../data/processed/stsbenchmark/sts-train.csv"
        fout_name_owd = "../../../data/output/stsbenchmark/sts-train.score.s_owd."\
                        + str(lambda1) + "_" + str(lambda2) + "_" + str(sigma) +\
                        ".txt"
        fout_name_wmd = "../../../data/output/stsbenchmark/sts-train.score.wmd.txt"
        sts_owd(ftextpair, col1_nor, col2_nor, col_score, fout_name_owd, model,
                lambda1, lambda2, sigma, max_iter, "euclidean")
        eval(ftextpair, col_score, fout_name_owd)
        sts_owd(ftextpair, col1_nor, col2_nor, col_score, fout_name_owd, model,
                lambda1, lambda2, sigma, max_iter, "cosine")
        eval(ftextpair, col_score, fout_name_owd)
        sts_wmd(ftextpair, col1_nor, col2_nor, col_score, fout_name_wmd, model)
        eval(ftextpair, col_score, fout_name_wmd)
        sts_wmd(ftextpair, col1, col2, col_score, fout_name_wmd, model)
        eval(ftextpair, col_score, fout_name_wmd)
        # dev
        ftextpair = "../../../data/processed/stsbenchmark/sts-dev.csv"
        fout_name_owd = "../../../data/output/stsbenchmark/sts-dev.score.s_owd."\
                        + str(lambda1) + "_" + str(lambda2) + "_" + str(sigma) +\
                        ".txt"
        fout_name_wmd = "../../../data/output/stsbenchmark/sts-dev.score.wmd.txt"
        sts_owd(ftextpair, col1_nor, col2_nor, col_score, fout_name_owd, model,
                lambda1, lambda2, sigma, max_iter, "euclidean")
        eval(ftextpair, col_score, fout_name_owd)
        sts_owd(ftextpair, col1_nor, col2_nor, col_score, fout_name_owd, model,
                lambda1, lambda2, sigma, max_iter, "cosine")
        eval(ftextpair, col_score, fout_name_owd)
        sts_wmd(ftextpair, col1_nor, col2_nor, col_score, fout_name_wmd, model)
        eval(ftextpair, col_score, fout_name_wmd)
        sts_wmd(ftextpair, col1, col2, col_score, fout_name_wmd, model)
        eval(ftextpair, col_score, fout_name_wmd)
        # test
        ftextpair = "../../../data/processed/stsbenchmark/sts-test.csv"
        fout_name_owd = "../../../data/output/stsbenchmark/sts-test.score.s_owd."\
                        + str(lambda1) + "_" + str(lambda2) + "_" + str(sigma) +\
                        ".txt"
        fout_name_wmd = "../../../data/output/stsbenchmark/sts-test.score.wmd.txt"
        sts_owd(ftextpair, col1_nor, col2_nor, col_score, fout_name_owd, model,
                lambda1, lambda2, sigma, max_iter, "euclidean")
        eval(ftextpair, col_score, fout_name_owd)
        sts_owd(ftextpair, col1_nor, col2_nor, col_score, fout_name_owd, model,
                lambda1, lambda2, sigma, max_iter, "cosine")
        eval(ftextpair, col_score, fout_name_owd)
        sts_wmd(ftextpair, col1_nor, col2_nor, col_score, fout_name_wmd, model)
        eval(ftextpair, col_score, fout_name_wmd)
        sts_wmd(ftextpair, col1, col2, col_score, fout_name_wmd, model)
        eval(ftextpair, col_score, fout_name_wmd)

        # fit_polynomial(fout_name_owd, ftextpair, 3)
        # fit_polynomial(fout_name_owd, ftextpair, 4)
        # fit_polynomial(fout_name_owd, ftextpair, 5)
        # fit_polynomial(fout_name_owd, ftextpair, 6)

    if exp == "sick2014":
        # EXP2: sick2014
        fw2v = "../../../data/raw/Google-w2v/GoogleNews-vectors-negative300.bin"
        w2v_type = "Glove"
        col1_nor = "sentence_A_normalized"
        col2_nor = "sentence_B_normalized"
        col1 = "sentence_A"
        col2 = "sentence_B"
        col_score = "relatedness_score"
        lambda1 = 1
        lambda2 = 0.03
        sigma = 100
        max_iter = 200
        # load word vectors
        model = load_w2v(fw2v, w2v_type)

        ftextpair = "../../../data/processed/sick2014/SICK_test.txt"
        fout_name_owd = "../../../data/output/sick2014/SICK_test.score.s_owd."\
                        + str(lambda1) + "_" + str(lambda2) + "_" + str(sigma) +\
                        ".txt"
        fout_name_wmd = "../../../data/output/sick2014/SICK_test.score.wmd.txt"


        # OWD_euclidean
        sts_owd(ftextpair, col1_nor, col2_nor, col_score, fout_name_owd, model,
                lambda1, lambda2, sigma, max_iter, "euclidean")
        eval(ftextpair, col_score, fout_name_owd)

        # OWD_cosine
        sts_owd(ftextpair, col1_nor, col2_nor, col_score, fout_name_owd, model,
                lambda1, lambda2, sigma, max_iter, "cosine")
        eval(ftextpair, col_score, fout_name_owd)

        # WMD_AMRnor
        sts_wmd(ftextpair, col1_nor, col2_nor, col_score, fout_name_wmd, model)
        eval(ftextpair, col_score, fout_name_wmd)

        # WMD
        sts_wmd(ftextpair, col1, col2, col_score, fout_name_wmd, model)
        eval(ftextpair, col_score, fout_name_wmd)

# # fit polynomial and calculate again
# lambda1 = 1
# lambda2 = 0.03
# sigma = 100
# fout_name = "../../../data/output/stsbenchmark/sts-test.score.s_owd."\
#             + str(lambda1) + "_" + str(lambda2) + "_" + str(sigma) + ".txt"
# y_pred = np.loadtxt(fout_name)
# df = pd.read_csv(fin_name, sep="\t")
# y = df["score"].values.astype(float)
# y_pred_fit = np.poly1d(np.polyfit(y_pred, y, 8))
# print "lambda1 = " + str(lambda1)
# print pearsonr(y, y_pred_fit(y_pred))

# # use WMD
# fout_name = "../../../data/output/stsbenchmark/sts-test.score.wmd.txt"
# fout = open(fout_name, "w")
# for i in range(len(sentence_A)):
#     # print i
#     snt1 = replace_OOV(sentence_A[i], model)
#     snt2 = replace_OOV(sentence_B[i], model)
#     score = model.wmdistance(snt1.split(), snt2.split())
#     fout.write(str(score) + "\n")
# fout.close()

# y_pred = np.loadtxt(fout_name)
# df = pd.read_csv(fin_name, sep="\t")
# y = df["score"].values.astype(float)
# print "use WMD"
# print pearsonr(y, y_pred)
