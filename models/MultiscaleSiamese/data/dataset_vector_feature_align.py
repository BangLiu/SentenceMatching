# -*-coding:utf-8-*-
import sys
import pandas as pd
import numpy as np
import gensim
import itertools
import torch
from torch.utils import data
from order_wassertein_distance import *
sys.path.insert(0, '../utils/')
from utils import *
reload(sys)
sys.setdefaultencoding('utf-8')


class STSDataset(data.Dataset):
    """ Load STS datasets.
    """
    def __init__(self, data_path, opt):
        """ Initialize dataset.
        """
        self.w2v_size = opt.W2V_SIZE
        self.fw2v = opt.W2V_FILE
        self.w2v_type = opt.W2V_TYPE
        self.w2v = self.load_w2v(self.fw2v)
        self.max_width = opt.MAX_WIDTH
        self.max_depth = opt.MAX_DEPTH
        self.feature_size = 0
        for d in range(self.max_depth + 1):
            self.feature_size += np.power(self.max_width, d)
        self.data_path = data_path
        self.col_text1 = opt.COL_TEXT1
        self.col_text2 = opt.COL_TEXT2
        self.col_score = opt.COL_SCORE
        self.opt = opt
        self.X, self.y = self.load_data(self.data_path,
                                        self.col_text1, self.col_text2,
                                        self.col_score,
                                        self.max_width, self.max_depth,
                                        opt.metrics)

    def load_w2v(self, fw2v):
        """ Load binary word vector file.
        """
        w2v = gensim.models.KeyedVectors.load_word2vec_format(
            fw2v, binary=True)
        return w2v

    def replace_OOV(self, text, model):
        """ Remove OOV words in a text.
        """
        text = text.split()
        new_text = []
        for i in range(len(text)):
            if text[i] in model.vocab:
                new_text.append(text[i])
        new_text = " ".join(new_text)
        return new_text

    def get_semantic_units_by_depth(self, alignment_str, depth):
        """ Get sentence semantic components given semantic depth in AMR.
        """
        if depth < 0:
            print "Minimum depth is 0 for sentence level."
            sys.exit(1)

        align = alignment_str.split()
        align = [x.split("|") for x in align]
        keys = [x[0] for x in align]
        keys = [longestCommonPrefix(x.split("+")) for x in keys]
        vals = [x[1] for x in align]
        alignments = dict(itertools.izip(keys, vals))
        alignments = sort_dict_by_key_str(alignments)

        len_align_id = 2 * depth + 1
        key_items = {}
        for i, j in itertools.groupby(alignments.keys(),
                                      key=lambda x: x[0:len_align_id]):
            ids = list(j)
            vals = [alignments[x] for x in ids]
            sub_aligns = dict(itertools.izip(ids, vals))
            sub_aligns = sort_dict_by_key_str(sub_aligns)
            key_items[i] = sub_aligns

        result = {}
        for k in key_items.keys():
            new_key = self.transform_alignment_key_by_depth(k, depth)
            result[new_key] = " ".join(key_items[k].values())
        result = sort_dict_by_key_str(result)
        return result

    def transform_alignment_key_by_depth(self, key, depth):
        ks = key.split(".")
        ks = [int(k) for k in ks]
        if len(ks) > 1:
            for i in range(1, len(ks)):
                ks[i] = ks[i] + 1
        if len(ks) < depth + 1:
            for i in range(len(ks), depth + 1):
                ks.append(0)
        ks = [str(k) for k in ks]
        new_key = ".".join(ks)
        return new_key

    def filter_semantic_units_by_width(self, semantic_units, width):
        """ Filter semantic units by maximum width.
        Notice: the unit key is the transformed alignment key.
        """
        def out_of_range(k, width):
            if ("." not in k) and (k != "0"):
                print "Warning: potential alignment error!"
                print semantic_units
                print k
                return True
            index = [int(x) for x in k.split(".")]
            result = np.sum([x >= width for x in index])
            return result
        filtered_dict = {
            k: v for k, v in semantic_units.iteritems()
            if not out_of_range(k, width)}
        result = sort_dict_by_key_str(filtered_dict)
        return result

    def key_to_index(self, key, width, depth):
        """ Alignment key to feature index.
        """
        k = [int(x) for x in key.split(".")]
        result = 0
        for i in range(depth + 1):
            result += k[i] * np.power(width, depth - i)
        return result

    def concat_vec_feature(self, text1, text2):
        """ Concatenate the word vector of two texts.
        v1 denotes the sum of word vectors of words in text1.
        v2 denotes the sum of word vectors of words in text2.
        return [v1, v2] concatenation.
        """
        text1 = self.replace_OOV(text1, self.w2v).split()
        text2 = self.replace_OOV(text2, self.w2v).split()
        v1 = np.zeros(self.w2v_size)
        if len(text1) > 0:
            v1 = np.zeros([len(text1), self.w2v_size])
            for i in range(len(text1)):
                v1[i, :] = self.w2v[text1[i]]
            v1 = np.sum(v1, 0)
        v2 = np.zeros(self.w2v_size)
        if len(text2) > 0:
            v2 = np.zeros([len(text2), self.w2v_size])
            for i in range(len(text2)):
                v2[i, :] = self.w2v[text2[i]]
            v2 = np.sum(v2, 0)
        return np.concatenate([v1, v2])

    def distance(self, v1, v2, metric):
        """ Define different feature calculation.
        """
        val = 0
        if metric == "OWD_euclidean":
            val = owd_snt_pair(self.replace_OOV(v1, self.w2v),
                               self.replace_OOV(v2, self.w2v),
                               self.w2v,
                               self.opt.lambda1,
                               self.opt.lambda2,
                               self.opt.sigma,
                               self.opt.max_iter,
                               "euclidean")
        elif metric == "OWD_cosine":
            val = owd_snt_pair(self.replace_OOV(v1, self.w2v),
                               self.replace_OOV(v2, self.w2v),
                               self.w2v,
                               self.opt.lambda1,
                               self.opt.lambda2,
                               self.opt.sigma,
                               self.opt.max_iter,
                               "cosine")
        elif metric == "WMD":
            snt1 = self.replace_OOV(v1, self.w2v)
            snt2 = self.replace_OOV(v2, self.w2v)
            if snt1 == "" and snt2 == "":
                val = 0
            elif snt1 == "" and snt2 != "":
                w2 = snt2.split()
                d = 0
                for ii in range(len(w2)):
                    d += np.linalg.norm(self.w2v[w2[ii]])
                val = d / len(w2)
            elif snt2 == "" and snt1 != "":
                w1 = snt1.split()
                d = 0
                for ii in range(len(w1)):
                    d += np.linalg.norm(self.w2v[w1[ii]])
                val = d / len(w1)
            else:
                val = self.w2v.wmdistance(
                    snt1.split(), snt2.split())
        else:
            print "warning: metrics not implemented yet!"
            sys.exit()
        return val

    def text_pair_vector_feature(self, alignment_str1, alignment_str2,
                                 max_width, max_depth):
        """ Extract the text pair concatenate vector feature.
        """
        n_row = self.feature_size
        n_col = 2 * self.w2v_size
        features = np.zeros([n_row, n_col])

        for d in range(max_depth + 1):
            key_shift = 0
            for d_shift in range(d):
                key_shift += np.power(max_width, d_shift)
            units1 = self.get_semantic_units_by_depth(alignment_str1, d)
            units1 = self.filter_semantic_units_by_width(units1, max_width)
            units2 = self.get_semantic_units_by_depth(alignment_str2, d)
            units2 = self.filter_semantic_units_by_width(units2, max_width)

            for k1, v1 in units1.iteritems():
                v2 = ""
                if k1 in units2.keys():
                    v2 = units2.get(k1)
                key = self.key_to_index(k1, max_width, d) + key_shift
                val = self.concat_vec_feature(v1, v2)
                features[key, :] = val

            for k2, v2 in units2.iteritems():
                v1 = ""
                if k2 in units1.keys():
                    v1 = units1.get(k2)
                key = self.key_to_index(k2, max_width, d) + key_shift
                val = self.concat_vec_feature(v1, v2)
                features[key, :] = val
        return features

    def text_pair_define_feature(self, alignment_str1, alignment_str2,
                                 max_width, max_depth, metrics):
        """ Extract feature vector x from text pair.
        """
        features = np.zeros([self.feature_size, len(metrics)])
        for i in range(len(metrics)):
            metric = metrics[i]
            result = np.zeros(self.feature_size)
            for d in range(max_depth + 1):
                key_shift = 0
                for d_shift in range(d):
                    key_shift += np.power(max_width, d_shift)
                units1 = self.get_semantic_units_by_depth(alignment_str1, d)
                units1 = self.filter_semantic_units_by_width(units1, max_width)
                units2 = self.get_semantic_units_by_depth(alignment_str2, d)
                units2 = self.filter_semantic_units_by_width(units2, max_width)

                for k1, v1 in units1.iteritems():
                    v2 = ""
                    if k1 in units2.keys():
                        v2 = units2.get(k1)
                    key = self.key_to_index(k1, max_width, d) + key_shift
                    val = self.distance(v1, v2, metric)
                    result[key] = val

                for k2, v2 in units2.iteritems():
                    v1 = ""
                    if k2 in units1.keys():
                        v1 = units1.get(k2)
                    key = self.key_to_index(k2, max_width, d) + key_shift
                    val = self.distance(v1, v2, metric)
                    result[key] = val
            features[:, i] = result
        return features

    def load_data(self, data_path, col_text1, col_text2, col_score,
                  max_width, max_depth, metrics):
        """ Load feature matrix X and target array y from data file.
        """
        print "Start loading data ..."
        X_vector = []
        X_define = []
        df = pd.read_csv(data_path, sep="\t")
        text1 = df[col_text1].values
        text2 = df[col_text2].values
        for i in range(len(text1)):
            X_vector.append(self.text_pair_vector_feature(
                text1[i], text2[i], max_width, max_depth))
            X_define.append(self.text_pair_define_feature(
                text1[i], text2[i], max_width, max_depth, metrics))
        X_vector = torch.FloatTensor(X_vector)
        X_define = torch.FloatTensor(X_define)
        X = torch.cat([X_vector, X_define], 2)
        y = torch.FloatTensor(df[col_score].values.astype(
            np.float32)) / 5.0  # normalize target
        print "End loading data ..."
        print "Data set size: ", X.size(), y.size()
        return X, y

    def __getitem__(self, index):
        """ Get data and label given index.
        """
        data = self.X[index]
        label = self.y[index]
        return data, label

    def __len__(self):
        """ Get dataset size.
        """
        return len(self.y)
