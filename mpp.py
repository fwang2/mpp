#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import argparse
import sys

args = None


def check_py_version():
    if sys.version_info <= (3, 5):
        print("You are running Python version below 3.5")
        print("Please consider upgrade or use a virtual env")
        sys.exit(1)


def parse():
    global args
    parser = argparse.ArgumentParser(description=" MPP Demo ")
    parser.add_argument("--train", metavar="FILE", help="training data file")
    parser.add_argument("--test", metavar="FILE", help="test data file")

    args = parser.parse_args()


def mpp(train, test, clabel, pw, cases):
    """
    train - training dataset
    test - test dataset
    clabel - class labels
    pw - prior probability
    """
    pass


if __name__ == "__main__":

    parse()
    print(args)
