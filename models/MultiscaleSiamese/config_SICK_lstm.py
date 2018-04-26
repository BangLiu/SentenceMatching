# -*-coding:utf-8-*-
import warnings


class DefaultConfig(object):
    # task type
    task = "Regression"

    # word vector
    W2V_TYPE = "Glove"
    W2V_SIZE = 0
    W2V_FILE = ""
    if W2V_TYPE == "Glove":
        W2V_FILE = "../../data/raw/Glove-EN-300d/glove.840B.300d.bin"
        W2V_SIZE = 300
    elif W2V_TYPE == "Google":
        W2V_FILE = "../../data/raw/Google-w2v/GoogleNews-vectors-negative300.bin"
        W2V_SIZE = 300

    # sentence hierarchical factorization
    MAX_DEPTH = 2
    MAX_WIDTH = 4

    metrics = ["OWD_euclidean", "OWD_cosine", "WMD"]

    lambda1 = 1
    lambda2 = 0.03
    sigma = 10
    max_iter = 20
    full_metric = "_".join(metrics) + "_" +\
        str(MAX_DEPTH) + "_" +\
        str(MAX_WIDTH) + "_" +\
        str(lambda1) + "_" +\
        str(lambda2) + "_" +\
        str(sigma) + "_" + str(max_iter)

    # model configuration
    env = 'main'
    model = 'SemMatchNet_malstm'
    load_model_path = None

    batch_size = 200
    max_epoch = 40
    lr = 0.004
    lr_decay = 0.9
    weight_decay = 1e-4

    use_gpu = False
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    # datasets
    COL_TEXT1 = "sentence_A_alignments"
    COL_TEXT2 = "sentence_B_alignments"
    COL_SCORE = "relatedness_score"

    train_data_path = '../../data/processed/sick2014/SICK_train.txt'
    train_features_path = train_data_path + "." + full_metric + "_" + W2V_TYPE + ".X.pth"
    train_targets_path = train_data_path + "." + full_metric + "_" + W2V_TYPE + ".y.pth"

    dev_data_path = '../../data/processed/sick2014/SICK_trial.txt'
    dev_features_path = dev_data_path + "." + full_metric + "_" + W2V_TYPE + ".X.pth"
    dev_targets_path = dev_data_path + "." + full_metric + "_" + W2V_TYPE + ".y.pth"

    test_data_path = '../../data/processed/sick2014/SICK_test.txt'
    test_features_path = test_data_path + "." + full_metric + "_" + W2V_TYPE + ".X.pth"
    test_targets_path = test_data_path + "." + full_metric + "_" + W2V_TYPE + ".y.pth"


def parse(self, kwargs):
    """ Update configure parameters according to kwargs.
    """
    for k, v in kwargs.iteritems():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
