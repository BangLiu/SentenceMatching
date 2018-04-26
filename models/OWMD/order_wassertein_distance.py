# -*- coding: utf-8 -*-
import sys
import getopt
import time
import numpy as np
import pickle as pkl
import gzip
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import spatial


def load_word2vec(w2v_file, vector_size):
    """ Load word vector file
    Get a dictionary of {word: word_vec}
    """
    print("Loading word2vec model: " + w2v_file)
    w2v = {}
    w2v["PADDING"] = np.zeros(vector_size)
    w2v["UNKNOWN"] = np.random.uniform(-0.25, 0.25, vector_size)
    with open(w2v_file, "r") as fread:
        for line in fread.readlines():
            line_list = line.strip().split(" ")
            if len(line_list) != 201:
                print(len(line_list))
                continue
            word = line_list[0]
            word_vec = np.fromstring(" ".join(line_list[1:]),
                                     dtype=float, sep=" ")
            w2v[word] = word_vec
    print("Word vectors: %d" % len(w2v))
    return w2v


def get_distance_mat(snt1, snt2, w2v, metric):
    snt1 = snt1.split()
    snt2 = snt2.split()
    D = np.zeros([len(snt1), len(snt2)])
    for i in range(len(snt1)):
        for j in range(len(snt2)):
            if metric == "euclidean":
                D[i][j] = np.linalg.norm(w2v[snt1[i]] - w2v[snt2[j]])
            elif metric == "cosine":
                D[i][j] = spatial.distance.cosine(w2v[snt1[i]], w2v[snt2[j]])
            else:
                print "please choose euclidean or cosine as metric."
                sys.exit()
    return D


def get_weight(snt):
    snt = snt.split()
    w = np.ones(len(snt)) / float(len(snt))
    return w


def owd_snt_pair(snt1, snt2, w2v, lambda1, lambda2, sigma, max_iter, metric):
    if snt1 == "" and snt2 == "":
        return 0
    if snt1 == "" and snt2.strip() != "":
        w2 = snt2.split()
        d = 0
        if metric == "euclidean":
            for i in range(len(w2)):
                d += np.linalg.norm(w2v[w2[i]])
            d = d / len(w2)
        elif metric == "cosine":
            d = 1.0
        return d
    if snt2 == "" and snt1.strip() != "":
        w1 = snt1.split()
        d = 0
        if metric == "euclidean":
            for i in range(len(w1)):
                d += np.linalg.norm(w2v[w1[i]])
            d = d / len(w1)
        elif metric == "cosine":
            d = 1.0
        return d
    a = get_weight(snt1)
    b = get_weight(snt2)
    D = get_distance_mat(snt1, snt2, w2v, metric)
    T = sinkhorn_knopp(a, b, D, lambda1, lambda2, sigma, numItermax=max_iter)
    distance = np.sum(T * D)
    return distance


def get_doc_distance_mat(snts1, snts2, w2v,
                         lambda1, lambda2, sigma, max_iter, metric):
    D = np.zeros([len(snts1), len(snts2)])
    for i in range(len(snts1)):
        for j in range(len(snts2)):
            D[i][j] = owd_snt_pair(snts1[i], snts2[j], w2v,
                                   lambda1, lambda2, sigma, max_iter, metric)
    return D


def get_snt_weight(snts):
    w = np.ones(len(snts)) / float(len(snts))
    return w


def owd_doc_pair(doc1, doc2, w2v,
                 lambda1_snt, lambda2_snt, sigma_snt, max_iter_snt,
                 lambda1_doc, lambda2_doc, sigma_doc, max_iter_doc,
                 metric):
    snts1 = sent_tokenize(doc1)
    snts2 = sent_tokenize(doc2)
    a = get_snt_weight(snts1)
    b = get_snt_weight(snts2)
    D = get_doc_distance_mat(snts1, snts2, w2v,
                             lambda1_snt, lambda2_snt, sigma_snt, max_iter_snt,
                             metric)
    T = sinkhorn_knopp(a, b, D,
                       lambda1_doc, lambda2_doc, sigma_doc, max_iter_doc)
    # print a, b, D, T
    distance = np.sum(T * D)
    return distance


def sinkhorn_knopp(a, b, D, lambda1, lambda2, sigma,
                   numItermax=200, stopThr=1e-9,
                   verbose=False, log=False, **kwargs):
    a = np.asarray(a, dtype=np.float64)  # alpha in paper
    b = np.asarray(b, dtype=np.float64)  # beta in paper
    D = np.asarray(D, dtype=np.float64)
    if len(a) == 0:
        a = np.ones((D.shape[0],), dtype=np.float64) / D.shape[0]
    if len(b) == 0:
        b = np.ones((D.shape[1],), dtype=np.float64) / D.shape[1]
    Nini = len(a)  # N in paper
    Nfin = len(b)  # M in paper
    if log:
        log = {'err': []}
    u = np.ones(Nini) / Nini  # k1 in paper
    v = np.ones(Nfin) / Nfin  # k2 in paper
    K = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            lij = np.abs(float(i) / Nini - float(j) / Nfin) / \
                np.sqrt(1 / float(Nini)**2 + 1 / float(Nfin)**2)
            p = np.exp(-np.square(lij) / (2 * np.square(sigma))) / \
                (sigma * np.sqrt(2 * np.pi))
            s = float(lambda1) / \
                (np.square(float(i) / Nini - float(j) / Nfin) + 1)
            d = D[i][j]
            K[i][j] = p * np.exp((s - d) / float(lambda2))
    iter = 0
    err = 1
    while (iter < numItermax):
        uprev = u
        vprev = v
        u = np.divide(a, np.dot(K, v))
        v = np.divide(b, np.dot(K.T, u))
        if (np.any(np.dot(K.T, u) == 0) or
                np.any(np.isnan(u)) or np.any(np.isnan(v)) or
                np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', iter)
            u = uprev
            v = vprev
            break
        if iter % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = np.dot(np.dot(np.diag(u), K), np.diag(v))
            err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if iter % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(iter, err))
        iter = iter + 1
    if log:
        log['u'] = u
        log['v'] = v
    T = np.dot(np.dot(np.diag(u), K), np.diag(v))
    if log:
        return T, log
    else:
        return T


def hierarchical_owd_snt_pair(snt1, snt2, w2v,
                              lambda1_snt, lambda2_snt, sigma_snt,
                              max_iter_snt):
    return 0
