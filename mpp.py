import numpy as np
import argparse
import sys

args = None

# def check_py_version():
#     if sys.version_info <= (3, 5):
#         print("You are running Python version below 3.5")
#         print("Please consider upgrade or use a virtual env")
#         sys.exit(1)


def parse():
    """ parse command line """
    global args
    parser = argparse.ArgumentParser(description=" MPP Demo ")
    parser.add_argument("train", metavar="Training", help="training data file")
    parser.add_argument("test", metavar="Testing", help="test data file")
    parser.add_argument("case", type=int, choices=range(1,4), help="choose from 1, 2, 3")
    args = parser.parse_args()

def pre_calc(df):
    pass

def mpp(df):
    """
    df['train_data'] - training data
    df['train_label'] - training label
    df['test_data'] - testing data
    df['test_label'] - testing label
    df['case'] - chodef mpp(df):sen case
    """
    pass

def load_data(trainf='datasets/synth.tr', testf='datasets/synth.te'):

    # process training data
    data = np.loadtxt(trainf)
    train_data = data[:,:-1]                    # return all feature columns except last
    train_label = data[:, -1].astype(int)       # put all labels in separate array
    
    # number of unique class
    classes = np.unique(train_label)

    # number of rows and columns
    nr, nc = np.shape(train_data)

    # processing testing data
    data = np.loadtxt(testf)
    test_data = data[:,:-1]
    test_label = data[:, -1].astype(int)

    # sanity check (# of columns must be equal)
    assert np.size(test_data, 1) == np.size(train_data, 1)


    return (train_data, train_label, test_data, test_label, nr, nc, classes)


def main():

    # now parse command line
    parse()


    # print("Loaded: {} records of training data, {} feature columns, {} classes".format(nr, nc, classes))
    # print("Loaded: {} records of test data, sanity check okay".format(np.size(test_data)))
    # print("Chosen case: ", args.case)


    # mpp(df)

if __name__ == "__main__":
    main()
