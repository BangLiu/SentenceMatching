# -*- coding: utf-8 -*-
import sys
import getopt
import time
import itertools
from collections import OrderedDict


def longestCommonPrefix(strs):
    if len(strs) == 0:
        return ""
    str = strs[0]
    Min = len(str)
    for i in range(1, len(strs)):
        j = 0
        p = strs[i]
        while j < Min and j < len(p) and p[j] == str[j]:
            j += 1
        Min = Min if Min < j else j
    return str[:Min]


# def normalize_sentence(fin, fout):
#     fin = open(fin)
#     fout = open(fout, "w")
#     tok = []
#     alignments = {}
#     for line in fin.readlines():
#         line = line.strip()
#         if "# ::tok " in line:
#             tok = line[8:].split()
#         if "# ::alignments " in line:
#             align = line.split(" ::")[1][11:].split()
#             align = [x.split("|") for x in align]
#             keys = [x[1] for x in align]
#             keys = [longestCommonPrefix(x.split("+")) for x in keys]
#             vals = [x[0] for x in align]
#             alignments = dict(itertools.izip(keys, vals))
#             alignments = OrderedDict(
#                 sorted(alignments.items(), key=lambda t: t[0]))
#             ordered_sentence = [
#                 tok[int(x.split("-")[0]): int(x.split("-")[1])]
#                 for x in alignments.values()]
#             ordered_sentence = [
#                 item for sublist in ordered_sentence for item in sublist]
#             ordered_sentence = " ".join(ordered_sentence)
#             fout.write(ordered_sentence + "\n")
#     fin.close()
#     fout.close()


def normalize_sentence(fin, fout):
    fin = open(fin)
    fout = open(fout, "w")
    tok = []
    alignments = {}
    for line in fin.readlines():
        line = line.strip()
        if "# ::tok " in line:
            tok = line[8:].split()
        if "# ::alignments " in line:
            align = line.split(" ::")[1][11:].split()
            align = [x.split("|") for x in align]
            keys_origin = [x[1] for x in align]
            keys = [longestCommonPrefix(x.split("+")) for x in keys_origin]
            for i in range(len(keys)):
                if keys[i][-1] == ".":
                    keys[i] = " ".join([k[:len(keys[i]) + 1] for k in keys_origin[i].split("+")])
            vals = [x[0] for x in align]
            alignments = dict(itertools.izip(keys, vals))
            for key in alignments.keys():
                val = alignments.get(key)
                val = tok[int(val.split("-")[0]): int(val.split("-")[1])]
                alignments[key] = val
            align_keys = alignments.keys()
            for key in align_keys:
                if " " in key:
                    ks = key.split()
                    for k in ks:
                        alignments[k] = val
                    del alignments[key]
            ordered_sentence = ""
            alignments = OrderedDict(
                sorted(alignments.items(), key=lambda t: t[0]))
            for key in alignments.keys():
                val = alignments.get(key)
                ordered_sentence += " ".join(val) + " "
            fout.write(ordered_sentence.strip() + "\n")
    fin.close()
    fout.close()


def usage():
    """
    Show how to use this script.
    """
    print "python normalize_sentence.py -i [input] -o [output]"


def main(argv):
    for arg in argv:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:")
        fin = ""
        fout = ""
        for op, value in opts:
            if op == "-i":
                fin = value
            elif op == "-o":
                fout = value
            elif op == "-h":
                usage()
                sys.exit()
    start_time = time.clock()
    normalize_sentence(fin, fout)
    end_time = time.clock()
    print "Time of normalize sentence: %f s" % (end_time - start_time)


if __name__ == "__main__":
    main(sys.argv)
