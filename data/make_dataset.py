# -*- coding: utf-8 -*-
"""
Transform raw datasets into desired format.
"""
import csv
import pandas as pd
from subprocess import check_call
from get_amr_alignment import *


def replace_sep(fin, fout, sep_ini, sep_fin):
    """ Replace delimiter in a file.
    """
    fin = open(fin, "r")
    fout = open(fout, "w")
    for line in fin:
        fout.write(line.replace(sep_ini, sep_fin))
    fin.close()
    fout.close()


def export_columns(fin, fout, col_list,
                   sep_in, sep_out, keep_header=False):
    """ Export column(s) from a file.
    """
    df = pd.read_csv(fin, sep=sep_in)
    df_out = df[col_list]
    df_out.to_csv(fout, sep=sep_out, header=keep_header, index=False)


def import_column(fin, fcol, fout, col,
                  sep_in, sep_out, contain_header=False):
    """ Merge a column from a file.
    """
    fcol = open(fcol, "r")
    lines = fcol.read().splitlines()
    df = pd.read_csv(fin, sep=sep_in, quoting=csv.QUOTE_NONE)
    df[col] = lines
    df.to_csv(fout, sep=sep_out, header=True, index=False)


def normalize_sentences(fin, fout):
    """ Normalize sentences in a file.
    For each line in input file, get the normalized sentence
    in output file.
    """
    check_call(["./normalize_sentence.sh", "JAMR", fin, fout])


def remove_quotes(fin, fout):
    """ Remove quotes in lines.
    If a line has odd number quotes, remove all quotes in this line.
    """
    fin = open(fin)
    fout = open(fout, "w")
    for line in fin:
        fout.write(line.replace("\"", ""))
    fin.close()
    fout.close()


def preprocessing_MSPC():
    remove_quotes(
        "../../data/raw/MSPC/train.txt",
        "../../data/processed/MSPC/train.txt")
    export_columns(
        "../../data/processed/MSPC/train.txt",
        "../../data/processed/MSPC/train.sentence1.txt",
        ["sentence1"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSPC/train.sentence1.txt",
        "../../data/processed/MSPC/train.sentence1.nor.txt")
    import_column(
        "../../data/processed/MSPC/train.txt",
        "../../data/processed/MSPC/train.sentence1.nor.txt",
        "../../data/processed/MSPC/train.txt",
        "sentence1_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSPC/train.sentence1.nor.txt.aligned",
        "../../data/processed/MSPC/train.sentence1.nor.alignments")
    import_column(
        "../../data/processed/MSPC/train.txt",
        "../../data/processed/MSPC/train.sentence1.nor.alignments",
        "../../data/processed/MSPC/train.txt",
        "sentence1_alignments", "\t", "\t")

    export_columns(
        "../../data/processed/MSPC/train.txt",
        "../../data/processed/MSPC/train.sentence2.txt",
        ["sentence2"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSPC/train.sentence2.txt",
        "../../data/processed/MSPC/train.sentence2.nor.txt")
    import_column(
        "../../data/processed/MSPC/train.txt",
        "../../data/processed/MSPC/train.sentence2.nor.txt",
        "../../data/processed/MSPC/train.txt",
        "sentence2_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSPC/train.sentence2.nor.txt.aligned",
        "../../data/processed/MSPC/train.sentence2.nor.alignments")
    import_column(
        "../../data/processed/MSPC/train.txt",
        "../../data/processed/MSPC/train.sentence2.nor.alignments",
        "../../data/processed/MSPC/train.txt",
        "sentence2_alignments", "\t", "\t")

    remove_quotes(
        "../../data/raw/MSPC/test.txt",
        "../../data/processed/MSPC/test.txt")
    export_columns(
        "../../data/processed/MSPC/test.txt",
        "../../data/processed/MSPC/test.sentence1.txt",
        ["sentence1"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSPC/test.sentence1.txt",
        "../../data/processed/MSPC/test.sentence1.nor.txt")
    import_column(
        "../../data/processed/MSPC/test.txt",
        "../../data/processed/MSPC/test.sentence1.nor.txt",
        "../../data/processed/MSPC/test.txt",
        "sentence1_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSPC/test.sentence1.nor.txt.aligned",
        "../../data/processed/MSPC/test.sentence1.nor.alignments")
    import_column(
        "../../data/processed/MSPC/test.txt",
        "../../data/processed/MSPC/test.sentence1.nor.alignments",
        "../../data/processed/MSPC/test.txt",
        "sentence1_alignments", "\t", "\t")

    export_columns(
        "../../data/processed/MSPC/test.txt",
        "../../data/processed/MSPC/test.sentence2.txt",
        ["sentence2"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSPC/test.sentence2.txt",
        "../../data/processed/MSPC/test.sentence2.nor.txt")
    import_column(
        "../../data/processed/MSPC/test.txt",
        "../../data/processed/MSPC/test.sentence2.nor.txt",
        "../../data/processed/MSPC/test.txt",
        "sentence2_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSPC/test.sentence2.nor.txt.aligned",
        "../../data/processed/MSPC/test.sentence2.nor.alignments")
    import_column(
        "../../data/processed/MSPC/test.txt",
        "../../data/processed/MSPC/test.sentence2.nor.alignments",
        "../../data/processed/MSPC/test.txt",
        "sentence2_alignments", "\t", "\t")


def preprocessing_sick2014():
    export_columns(
        "../../data/raw/sick2014/SICK_train.txt",
        "../../data/processed/sick2014/SICK_train.sentence_A.txt",
        ["sentence_A"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/sick2014/SICK_train.sentence_A.txt",
        "../../data/processed/sick2014/SICK_train.sentence_A.nor.txt")
    import_column(
        "../../data/raw/sick2014/SICK_train.txt",
        "../../data/processed/sick2014/SICK_train.sentence_A.nor.txt",
        "../../data/processed/sick2014/SICK_train.txt",
        "sentence_A_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/sick2014/SICK_train.sentence_A.nor.txt.aligned",
        "../../data/processed/sick2014/SICK_train.sentence_A.nor.alignments")
    import_column(
        "../../data/processed/sick2014/SICK_train.txt",
        "../../data/processed/sick2014/SICK_train.sentence_A.nor.alignments",
        "../../data/processed/sick2014/SICK_train.txt",
        "sentence_A_alignments", "\t", "\t")

    export_columns(
        "../../data/raw/sick2014/SICK_train.txt",
        "../../data/processed/sick2014/SICK_train.sentence_B.txt",
        ["sentence_B"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/sick2014/SICK_train.sentence_B.txt",
        "../../data/processed/sick2014/SICK_train.sentence_B.nor.txt")
    import_column(
        "../../data/processed/sick2014/SICK_train.txt",
        "../../data/processed/sick2014/SICK_train.sentence_B.nor.txt",
        "../../data/processed/sick2014/SICK_train.txt",
        "sentence_B_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/sick2014/SICK_train.sentence_B.nor.txt.aligned",
        "../../data/processed/sick2014/SICK_train.sentence_B.nor.alignments")
    import_column(
        "../../data/processed/sick2014/SICK_train.txt",
        "../../data/processed/sick2014/SICK_train.sentence_B.nor.alignments",
        "../../data/processed/sick2014/SICK_train.txt",
        "sentence_B_alignments", "\t", "\t")

    export_columns(
        "../../data/raw/sick2014/SICK_test.txt",
        "../../data/processed/sick2014/SICK_test.sentence_A.txt",
        ["sentence_A"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/sick2014/SICK_test.sentence_A.txt",
        "../../data/processed/sick2014/SICK_test.sentence_A.nor.txt")
    import_column(
        "../../data/raw/sick2014/SICK_test.txt",
        "../../data/processed/sick2014/SICK_test.sentence_A.nor.txt",
        "../../data/processed/sick2014/SICK_test.txt",
        "sentence_A_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/sick2014/SICK_test.sentence_A.nor.txt.aligned",
        "../../data/processed/sick2014/SICK_test.sentence_A.nor.alignments")
    import_column(
        "../../data/processed/sick2014/SICK_test.txt",
        "../../data/processed/sick2014/SICK_test.sentence_A.nor.alignments",
        "../../data/processed/sick2014/SICK_test.txt",
        "sentence_A_alignments", "\t", "\t")

    export_columns(
        "../../data/raw/sick2014/SICK_test.txt",
        "../../data/processed/sick2014/SICK_test.sentence_B.txt",
        ["sentence_B"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/sick2014/SICK_test.sentence_B.txt",
        "../../data/processed/sick2014/SICK_test.sentence_B.nor.txt")
    import_column(
        "../../data/processed/sick2014/SICK_test.txt",
        "../../data/processed/sick2014/SICK_test.sentence_B.nor.txt",
        "../../data/processed/sick2014/SICK_test.txt",
        "sentence_B_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/sick2014/SICK_test.sentence_B.nor.txt.aligned",
        "../../data/processed/sick2014/SICK_test.sentence_B.nor.alignments")
    import_column(
        "../../data/processed/sick2014/SICK_test.txt",
        "../../data/processed/sick2014/SICK_test.sentence_B.nor.alignments",
        "../../data/processed/sick2014/SICK_test.txt",
        "sentence_B_alignments", "\t", "\t")

    export_columns(
        "../../data/raw/sick2014/SICK_trial.txt",
        "../../data/processed/sick2014/SICK_trial.sentence_A.txt",
        ["sentence_A"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/sick2014/SICK_trial.sentence_A.txt",
        "../../data/processed/sick2014/SICK_trial.sentence_A.nor.txt")
    import_column(
        "../../data/raw/sick2014/SICK_trial.txt",
        "../../data/processed/sick2014/SICK_trial.sentence_A.nor.txt",
        "../../data/processed/sick2014/SICK_trial.txt",
        "sentence_A_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/sick2014/SICK_trial.sentence_A.nor.txt.aligned",
        "../../data/processed/sick2014/SICK_trial.sentence_A.nor.alignments")
    import_column(
        "../../data/processed/sick2014/SICK_trial.txt",
        "../../data/processed/sick2014/SICK_trial.sentence_A.nor.alignments",
        "../../data/processed/sick2014/SICK_trial.txt",
        "sentence_A_alignments", "\t", "\t")

    export_columns(
        "../../data/raw/sick2014/SICK_trial.txt",
        "../../data/processed/sick2014/SICK_trial.sentence_B.txt",
        ["sentence_B"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/sick2014/SICK_trial.sentence_B.txt",
        "../../data/processed/sick2014/SICK_trial.sentence_B.nor.txt")
    import_column(
        "../../data/processed/sick2014/SICK_trial.txt",
        "../../data/processed/sick2014/SICK_trial.sentence_B.nor.txt",
        "../../data/processed/sick2014/SICK_trial.txt",
        "sentence_B_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/sick2014/SICK_trial.sentence_B.nor.txt.aligned",
        "../../data/processed/sick2014/SICK_trial.sentence_B.nor.alignments")
    import_column(
        "../../data/processed/sick2014/SICK_trial.txt",
        "../../data/processed/sick2014/SICK_trial.sentence_B.nor.alignments",
        "../../data/processed/sick2014/SICK_trial.txt",
        "sentence_B_alignments", "\t", "\t")


def preprocessing_STSbenchmark():
    # train
    remove_quotes(
        "../../data/raw/stsbenchmark/sts-train.csv",
        "../../data/processed/stsbenchmark/sts-train.csv")
    export_columns(
        "../../data/processed/stsbenchmark/sts-train.csv",
        "../../data/processed/stsbenchmark/sts-train.sentence1.txt",
        ["sentence1"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/stsbenchmark/sts-train.sentence1.txt",
        "../../data/processed/stsbenchmark/sts-train.sentence1.nor.txt")
    import_column(
        "../../data/processed/stsbenchmark/sts-train.csv",
        "../../data/processed/stsbenchmark/sts-train.sentence1.nor.txt",
        "../../data/processed/stsbenchmark/sts-train.csv",
        "sentence1_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/stsbenchmark/sts-train.sentence1.nor.txt.aligned",
        "../../data/processed/stsbenchmark/sts-train.sentence1.nor.alignments")
    import_column(
        "../../data/processed/stsbenchmark/sts-train.csv",
        "../../data/processed/stsbenchmark/sts-train.sentence1.nor.alignments",
        "../../data/processed/stsbenchmark/sts-train.csv",
        "sentence1_alignments", "\t", "\t")

    export_columns(
        "../../data/processed/stsbenchmark/sts-train.csv",
        "../../data/processed/stsbenchmark/sts-train.sentence2.txt",
        ["sentence2"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/stsbenchmark/sts-train.sentence2.txt",
        "../../data/processed/stsbenchmark/sts-train.sentence2.nor.txt")
    import_column(
        "../../data/processed/stsbenchmark/sts-train.csv",
        "../../data/processed/stsbenchmark/sts-train.sentence2.nor.txt",
        "../../data/processed/stsbenchmark/sts-train.csv",
        "sentence2_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/stsbenchmark/sts-train.sentence2.nor.txt.aligned",
        "../../data/processed/stsbenchmark/sts-train.sentence2.nor.alignments")
    import_column(
        "../../data/processed/stsbenchmark/sts-train.csv",
        "../../data/processed/stsbenchmark/sts-train.sentence2.nor.alignments",
        "../../data/processed/stsbenchmark/sts-train.csv",
        "sentence2_alignments", "\t", "\t")

    # test
    remove_quotes(
        "../../data/raw/stsbenchmark/sts-test.csv",
        "../../data/processed/stsbenchmark/sts-test.csv")
    export_columns(
        "../../data/processed/stsbenchmark/sts-test.csv",
        "../../data/processed/stsbenchmark/sts-test.sentence1.txt",
        ["sentence1"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/stsbenchmark/sts-test.sentence1.txt",
        "../../data/processed/stsbenchmark/sts-test.sentence1.nor.txt")
    import_column(
        "../../data/processed/stsbenchmark/sts-test.csv",
        "../../data/processed/stsbenchmark/sts-test.sentence1.nor.txt",
        "../../data/processed/stsbenchmark/sts-test.csv",
        "sentence1_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/stsbenchmark/sts-test.sentence1.nor.txt.aligned",
        "../../data/processed/stsbenchmark/sts-test.sentence1.nor.alignments")
    import_column(
        "../../data/processed/stsbenchmark/sts-test.csv",
        "../../data/processed/stsbenchmark/sts-test.sentence1.nor.alignments",
        "../../data/processed/stsbenchmark/sts-test.csv",
        "sentence1_alignments", "\t", "\t")

    export_columns(
        "../../data/processed/stsbenchmark/sts-test.csv",
        "../../data/processed/stsbenchmark/sts-test.sentence2.txt",
        ["sentence2"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/stsbenchmark/sts-test.sentence2.txt",
        "../../data/processed/stsbenchmark/sts-test.sentence2.nor.txt")
    import_column(
        "../../data/processed/stsbenchmark/sts-test.csv",
        "../../data/processed/stsbenchmark/sts-test.sentence2.nor.txt",
        "../../data/processed/stsbenchmark/sts-test.csv",
        "sentence2_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/stsbenchmark/sts-test.sentence2.nor.txt.aligned",
        "../../data/processed/stsbenchmark/sts-test.sentence2.nor.alignments")
    import_column(
        "../../data/processed/stsbenchmark/sts-test.csv",
        "../../data/processed/stsbenchmark/sts-test.sentence2.nor.alignments",
        "../../data/processed/stsbenchmark/sts-test.csv",
        "sentence2_alignments", "\t", "\t")

    # dev
    remove_quotes(
        "../../data/raw/stsbenchmark/sts-dev.csv",
        "../../data/processed/stsbenchmark/sts-dev.csv")
    export_columns(
        "../../data/processed/stsbenchmark/sts-dev.csv",
        "../../data/processed/stsbenchmark/sts-dev.sentence1.txt",
        ["sentence1"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/stsbenchmark/sts-dev.sentence1.txt",
        "../../data/processed/stsbenchmark/sts-dev.sentence1.nor.txt")
    import_column(
        "../../data/processed/stsbenchmark/sts-dev.csv",
        "../../data/processed/stsbenchmark/sts-dev.sentence1.nor.txt",
        "../../data/processed/stsbenchmark/sts-dev.csv",
        "sentence1_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/stsbenchmark/sts-dev.sentence1.nor.txt.aligned",
        "../../data/processed/stsbenchmark/sts-dev.sentence1.nor.alignments")
    import_column(
        "../../data/processed/stsbenchmark/sts-dev.csv",
        "../../data/processed/stsbenchmark/sts-dev.sentence1.nor.alignments",
        "../../data/processed/stsbenchmark/sts-dev.csv",
        "sentence1_alignments", "\t", "\t")

    export_columns(
        "../../data/processed/stsbenchmark/sts-dev.csv",
        "../../data/processed/stsbenchmark/sts-dev.sentence2.txt",
        ["sentence2"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/stsbenchmark/sts-dev.sentence2.txt",
        "../../data/processed/stsbenchmark/sts-dev.sentence2.nor.txt")
    import_column(
        "../../data/processed/stsbenchmark/sts-dev.csv",
        "../../data/processed/stsbenchmark/sts-dev.sentence2.nor.txt",
        "../../data/processed/stsbenchmark/sts-dev.csv",
        "sentence2_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/stsbenchmark/sts-dev.sentence2.nor.txt.aligned",
        "../../data/processed/stsbenchmark/sts-dev.sentence2.nor.alignments")
    import_column(
        "../../data/processed/stsbenchmark/sts-dev.csv",
        "../../data/processed/stsbenchmark/sts-dev.sentence2.nor.alignments",
        "../../data/processed/stsbenchmark/sts-dev.csv",
        "sentence2_alignments", "\t", "\t")


def preprocessing_MSRvid():
    # train
    remove_quotes(
        "../../data/raw/MSRvid/train.txt",
        "../../data/processed/MSRvid/train.txt")
    export_columns(
        "../../data/processed/MSRvid/train.txt",
        "../../data/processed/MSRvid/train.sentence1.txt",
        ["sentence1"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSRvid/train.sentence1.txt",
        "../../data/processed/MSRvid/train.sentence1.nor.txt")
    import_column(
        "../../data/processed/MSRvid/train.txt",
        "../../data/processed/MSRvid/train.sentence1.nor.txt",
        "../../data/processed/MSRvid/train.txt",
        "sentence1_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSRvid/train.sentence1.nor.txt.aligned",
        "../../data/processed/MSRvid/train.sentence1.nor.alignments")
    import_column(
        "../../data/processed/MSRvid/train.txt",
        "../../data/processed/MSRvid/train.sentence1.nor.alignments",
        "../../data/processed/MSRvid/train.txt",
        "sentence1_alignments", "\t", "\t")

    export_columns(
        "../../data/processed/MSRvid/train.txt",
        "../../data/processed/MSRvid/train.sentence2.txt",
        ["sentence2"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSRvid/train.sentence2.txt",
        "../../data/processed/MSRvid/train.sentence2.nor.txt")
    import_column(
        "../../data/processed/MSRvid/train.txt",
        "../../data/processed/MSRvid/train.sentence2.nor.txt",
        "../../data/processed/MSRvid/train.txt",
        "sentence2_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSRvid/train.sentence2.nor.txt.aligned",
        "../../data/processed/MSRvid/train.sentence2.nor.alignments")
    import_column(
        "../../data/processed/MSRvid/train.txt",
        "../../data/processed/MSRvid/train.sentence2.nor.alignments",
        "../../data/processed/MSRvid/train.txt",
        "sentence2_alignments", "\t", "\t")

    # test
    remove_quotes(
        "../../data/raw/MSRvid/test.txt",
        "../../data/processed/MSRvid/test.txt")
    export_columns(
        "../../data/processed/MSRvid/test.txt",
        "../../data/processed/MSRvid/test.sentence1.txt",
        ["sentence1"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSRvid/test.sentence1.txt",
        "../../data/processed/MSRvid/test.sentence1.nor.txt")
    import_column(
        "../../data/processed/MSRvid/test.txt",
        "../../data/processed/MSRvid/test.sentence1.nor.txt",
        "../../data/processed/MSRvid/test.txt",
        "sentence1_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSRvid/test.sentence1.nor.txt.aligned",
        "../../data/processed/MSRvid/test.sentence1.nor.alignments")
    import_column(
        "../../data/processed/MSRvid/test.txt",
        "../../data/processed/MSRvid/test.sentence1.nor.alignments",
        "../../data/processed/MSRvid/test.txt",
        "sentence1_alignments", "\t", "\t")

    export_columns(
        "../../data/processed/MSRvid/test.txt",
        "../../data/processed/MSRvid/test.sentence2.txt",
        ["sentence2"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSRvid/test.sentence2.txt",
        "../../data/processed/MSRvid/test.sentence2.nor.txt")
    import_column(
        "../../data/processed/MSRvid/test.txt",
        "../../data/processed/MSRvid/test.sentence2.nor.txt",
        "../../data/processed/MSRvid/test.txt",
        "sentence2_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSRvid/test.sentence2.nor.txt.aligned",
        "../../data/processed/MSRvid/test.sentence2.nor.alignments")
    import_column(
        "../../data/processed/MSRvid/test.txt",
        "../../data/processed/MSRvid/test.sentence2.nor.alignments",
        "../../data/processed/MSRvid/test.txt",
        "sentence2_alignments", "\t", "\t")


def preprocessing_MSRpar():
    # train
    remove_quotes(
        "../../data/raw/MSRpar/train.txt",
        "../../data/processed/MSRpar/train.txt")
    export_columns(
        "../../data/processed/MSRpar/train.txt",
        "../../data/processed/MSRpar/train.sentence1.txt",
        ["sentence1"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSRpar/train.sentence1.txt",
        "../../data/processed/MSRpar/train.sentence1.nor.txt")
    import_column(
        "../../data/processed/MSRpar/train.txt",
        "../../data/processed/MSRpar/train.sentence1.nor.txt",
        "../../data/processed/MSRpar/train.txt",
        "sentence1_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSRpar/train.sentence1.nor.txt.aligned",
        "../../data/processed/MSRpar/train.sentence1.nor.alignments")
    import_column(
        "../../data/processed/MSRpar/train.txt",
        "../../data/processed/MSRpar/train.sentence1.nor.alignments",
        "../../data/processed/MSRpar/train.txt",
        "sentence1_alignments", "\t", "\t")

    export_columns(
        "../../data/processed/MSRpar/train.txt",
        "../../data/processed/MSRpar/train.sentence2.txt",
        ["sentence2"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSRpar/train.sentence2.txt",
        "../../data/processed/MSRpar/train.sentence2.nor.txt")
    import_column(
        "../../data/processed/MSRpar/train.txt",
        "../../data/processed/MSRpar/train.sentence2.nor.txt",
        "../../data/processed/MSRpar/train.txt",
        "sentence2_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSRpar/train.sentence2.nor.txt.aligned",
        "../../data/processed/MSRpar/train.sentence2.nor.alignments")
    import_column(
        "../../data/processed/MSRpar/train.txt",
        "../../data/processed/MSRpar/train.sentence2.nor.alignments",
        "../../data/processed/MSRpar/train.txt",
        "sentence2_alignments", "\t", "\t")

    # test
    remove_quotes(
        "../../data/raw/MSRpar/test.txt",
        "../../data/processed/MSRpar/test.txt")
    export_columns(
        "../../data/processed/MSRpar/test.txt",
        "../../data/processed/MSRpar/test.sentence1.txt",
        ["sentence1"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSRpar/test.sentence1.txt",
        "../../data/processed/MSRpar/test.sentence1.nor.txt")
    import_column(
        "../../data/processed/MSRpar/test.txt",
        "../../data/processed/MSRpar/test.sentence1.nor.txt",
        "../../data/processed/MSRpar/test.txt",
        "sentence1_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSRpar/test.sentence1.nor.txt.aligned",
        "../../data/processed/MSRpar/test.sentence1.nor.alignments")
    import_column(
        "../../data/processed/MSRpar/test.txt",
        "../../data/processed/MSRpar/test.sentence1.nor.alignments",
        "../../data/processed/MSRpar/test.txt",
        "sentence1_alignments", "\t", "\t")

    export_columns(
        "../../data/processed/MSRpar/test.txt",
        "../../data/processed/MSRpar/test.sentence2.txt",
        ["sentence2"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/MSRpar/test.sentence2.txt",
        "../../data/processed/MSRpar/test.sentence2.nor.txt")
    import_column(
        "../../data/processed/MSRpar/test.txt",
        "../../data/processed/MSRpar/test.sentence2.nor.txt",
        "../../data/processed/MSRpar/test.txt",
        "sentence2_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/MSRpar/test.sentence2.nor.txt.aligned",
        "../../data/processed/MSRpar/test.sentence2.nor.alignments")
    import_column(
        "../../data/processed/MSRpar/test.txt",
        "../../data/processed/MSRpar/test.sentence2.nor.alignments",
        "../../data/processed/MSRpar/test.txt",
        "sentence2_alignments", "\t", "\t")


def preprocessing_Quora():
    remove_quotes(
        "../../data/raw/Quora/quora.txt",
        "../../data/processed/Quora/quora.txt")
    export_columns(
        "../../data/processed/Quora/quora.txt",
        "../../data/processed/Quora/quora.question1.txt",
        ["question1"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/Quora/quora.question1.txt",
        "../../data/processed/Quora/quora.question1.nor.txt")
    import_column(
        "../../data/processed/Quora/quora.txt",
        "../../data/processed/Quora/quora.question1.nor.txt",
        "../../data/processed/Quora/quora.txt",
        "question1_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/Quora/quora.question1.nor.txt.aligned",
        "../../data/processed/Quora/quora.question1.nor.alignments")
    import_column(
        "../../data/processed/Quora/quora.txt",
        "../../data/processed/Quora/quora.question1.nor.alignments",
        "../../data/processed/Quora/quora.txt",
        "question1_alignments", "\t", "\t")

    export_columns(
        "../../data/processed/Quora/quora.txt",
        "../../data/processed/Quora/quora.question2.txt",
        ["question2"], "\t", "\t", keep_header=False)
    normalize_sentences(
        "../../data/processed/Quora/quora.question2.txt",
        "../../data/processed/Quora/quora.question2.nor.txt")
    import_column(
        "../../data/processed/Quora/quora.txt",
        "../../data/processed/Quora/quora.question2.nor.txt",
        "../../data/processed/Quora/quora.txt",
        "question2_normalized", "\t", "\t")
    get_amr_alignment(
        "../../data/processed/Quora/quora.question2.nor.txt.aligned",
        "../../data/processed/Quora/quora.question2.nor.alignments")
    import_column(
        "../../data/processed/Quora/quora.txt",
        "../../data/processed/Quora/quora.question2.nor.alignments",
        "../../data/processed/Quora/quora.txt",
        "question2_alignments", "\t", "\t")


if __name__ == "__main__":
    # preprocessing_STSbenchmark()
    # preprocessing_sick2014()
    # preprocessing_MSPC()
    # preprocessing_Quora()
    preprocessing_MSRvid()
    preprocessing_MSRpar()
