import numpy as np
import argparse
import sys


# module level globals
means, covs = {}, {}
covavg, varavg = None, None

# def check_py_version():
#     if sys.version_info <= (3, 5):
#         print("You are running Python version below 3.5")
#         print("Please consider upgrade or use a virtual env")
#         sys.exit(1)


def parse_cmdline():
    """ parse command line """
    parser = argparse.ArgumentParser(description=" MPP Demo ")
    parser.add_argument("train", metavar="Training", help="training data file")
    parser.add_argument("test", metavar="Testing", help="test data file")
    parser.add_argument("case", type=int, choices=range(1,4), help="choose from 1, 2, 3")
    return parser.parse_args()
 
def mpp1(Tr, sample, classn, case, pw):
    """ return a label """
    assert varavg != None, \
        "mpp is not initialized yet, call load_data(...), then mpp_init(...) first."

def mpp_init(Tr, Te, classes):
    """ initialize """
    global covs, means, covavg, varavg

    covsum = None

    for c in classes:
        arr = Tr[Tr[:, -1] == c]
        covs[c] = np.cov(arr[:,:-1]) # do cov on array sans last column
        means[c] = np.mean(arr[:,:-1], axis=0) # mean along rows
        if covsum is None:
            covsum = covs[c]
        else:
            covsum += covs[c]

    # used by case II
    covavg = covsum / len(classes)

    # used by case I
    varavg = np.sum(covavg) / len(classes)

    return covs, means, covavg, varavg
    
def load_data(trainf='datasets/synth.tr', testf='datasets/synth.te'):
    """ initialize with training and testing data, return a tuple of
    (training, train_label, test, test_label, etc.)
    """
    global covs, means, covavg, varavg 

    # process training data
    train_data = np.genfromtxt(trainf)
    # return all feature columns except last
    # train_data = train_data[:,:-1]    

    # number of unique class
    classes = np.unique(train_data[:,-1])

    # number of rows and columns
    nr, nc = np.shape(train_data)
    nf = nc - 1

    # processing testing data
    test_data = np.genfromtxt(testf)

    # sanity check (# of columns must be equal)
    assert np.size(test_data, 1) == np.size(train_data, 1)

    # init prio probability, equal distribution
    classn = len(classes)
    pw = np.full(classn, 1/classn)

    return (train_data, test_data, classes, nr, nf, pw)


def main():

    # now parse command line
    args = parse_cmdline()
    train_data, test_data, classes, nr, nf, pw = load_data(args.train, args.test)
    mpp_init(train_data, test_data, classes)
    mpp1(train_data, None, None, None, pw)

    # print("Loaded: {} records of training data, {} feature columns, {} classes".format(nr, nc, classes))
    # print("Loaded: {} records of test data, sanity check okay".format(np.size(test_data)))
    # print("Chosen case: ", args.case)


    # mpp(df)

if __name__ == "__main__":
    main()
