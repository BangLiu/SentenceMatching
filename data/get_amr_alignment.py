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


# def get_amr_alignment(fin, fout):
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
#             for key in alignments.keys():
#                 val = alignments.get(key)
#                 val = tok[int(val.split("-")[0]): int(val.split("-")[1])]
#                 alignments[key] = val
#             align_str = ""
#             for key in alignments.keys():
#                 val = alignments.get(key)
#                 align_str += key + "|" + "-".join(val) + " "
#             fout.write(align_str + "\n")
#     fin.close()
#     fout.close()


def get_amr_alignment(fin, fout):
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
            align_str = ""
            alignments = OrderedDict(
                sorted(alignments.items(), key=lambda t: t[0]))
            for key in alignments.keys():
                val = alignments.get(key)
                align_str += key + "|" + "-".join(val) + " "
            fout.write(align_str.strip() + "\n")
    fin.close()
    fout.close()


def usage():
    """
    Show how to use this script.
    """
    print "python get_amr_alignment.py -i [input] -o [output]"


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
    get_amr_alignment(fin, fout)
    end_time = time.clock()
    print "Time of get_amr_alignment: %f s" % (end_time - start_time)


if __name__ == "__main__":
    main(sys.argv)
