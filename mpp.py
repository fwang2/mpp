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
            self.train(self.train_data, self.classes)

    def train(self, Tr, classes):
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

    def evaluate(self, case = 1):
        """ eval all data """
        disc = np.zeros(len(self.classes))
        for i in range(self.test_nr):
            if case == 1:
                for c in self.classes:
                    c = int(c)  # class is float, convert to int for indexing
                    ti = self.test_data[i,:-1]  # get ith row
                    edist = np.linalg.norm(self.means[c] - ti)
                    disc[c] = -edist*edist/(2*self.varavg) + np.log(self.pw[c])  
                    print("edist: {}, disc{}".format(edist, disc[c]))
            elif case == 2:
                pass
            elif case == 3:
                pass
            else:
                raise Exception("Can't handle case value", case)

def main():

    # now parse command line
    # args = parse_cmdline()
    # mpp = MPP(args.train, args.test)
    # mpp.train(args.case)

    m = MPP("datasets/synth.tr", "datasets/synth.te")
    m.eval(case = 1)

if __name__ == "__main__":
    main()
