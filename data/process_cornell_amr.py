# -*- coding: utf-8 -*-
import sys
import getopt
import time


def process_cornell_amr_for_align(fin, fout):
    fin = open(fin)
    fout = open(fout, "w")
    preline = ""
    for line in fin.readlines():
        if preline == "":
            fout.write("# ::snt " + line)
        else:
            fout.write(line)
        preline = line.strip()
    fin.close()
    fout.close()


def usage():
    """
    Show how to use this script.
    """
    print "python process_cornell_amr.py -i [input] -o [output]"


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
    process_cornell_amr_for_align(fin, fout)
    end_time = time.clock()
    print "Time of process cornell amr data: %f s" % (end_time - start_time)


if __name__ == "__main__":
    main(sys.argv)
