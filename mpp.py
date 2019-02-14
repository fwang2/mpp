import numpy as np
import argparse
import sys



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

class MPP:

    def __init__(self, trainf, testf):
        # process training data
        self.train_data = np.genfromtxt(trainf)
        # return all feature columns except last
        # train_data = train_data[:,:-1]    

        # number of unique class
        self.classes = np.unique(self.train_data[:,-1])

        # number of rows and columns
        self.train_nr, self.nc = np.shape(self.train_data)
        self.nf = self.nc - 1

        # processing testing data
        self.test_data = np.genfromtxt(testf)
        self.test_nr, _ = np.shape(self.test_data)
        
        # sanity check (# of columns must be equal)
        assert np.size(self.test_data, 1) == np.size(self.train_data, 1)

        # init prio probability, equal distribution
        self.classn = len(self.classes)
        self.pw = np.full(self.classn, 1/self.classn)

        self.covs, self.means, self.covavg, self.varavg = \
            self.mpp_init(self.train_data, self.test_data, self.classes)

    def mpp_init(self, Tr, Te, classes):
        """ initialize: return (covs, means, covavg, varavg) """
        covs, means = {}, {}
        covavg, varavg, covsum = None, None, None

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

    def set_case(self, case):
        self.case = case

    def train1(self, sample):
        """ return a label """
        print("calling train")

    def train(self, case = 1):
        """ train all data """


def main():

    # now parse command line
    args = parse_cmdline()
    mpp = MPP(args.train, args.test)
    mpp.train(args.case)

    # print("Loaded: {} records of training data, {} feature columns, {} classes".format(nr, nc, classes))
    # print("Loaded: {} records of test data, sanity check okay".format(np.size(test_data)))
    # print("Chosen case: ", args.case)


    # mpp(df)

if __name__ == "__main__":
    main()
