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
    parser.add_argument(
        "case", type=int, choices=range(1, 4), help="choose from 1, 2, 3")
    return parser.parse_args()


class mpp:
    def __init__(self, case=1):
        # init prio probability, equal distribution
        # self.classn = len(self.classes)
        # self.pw = np.full(self.classn, 1/self.classn)

        # self.covs, self.means, self.covavg, self.varavg = \
        #     self.train(self.train_data, self.classes)
        self.case_ = case

    def fit(self, Tr, y):
        self.covs_, self.means_ = {}, {}
        self.covsum_ = None

        self.classes_ = np.unique(y)
        self.classn_ = len(self.classes_)
        self.pw_ = np.full(self.classn_, 1 / self.classn_)

        for c in self.classes_:
            arr = Tr[y == c]
            self.covs_[c] = np.cov(arr)
            self.means_[c] = np.mean(arr, axis=0)  # mean along rows
            if self.covsum_ is None:
                self.covsum_ = self.covs_[c]
            else:
                self.covsum_ += self.covs_[c]

        # used by case II
        self.covavg_ = self.covsum_ / len(self.classes_)

        # used by case I
        self.varavg_ = np.sum(self.covavg_) / len(self.classes_)

    def predict(self, T):
        """ eval all data """
        y = []
        disc = np.zeros(self.classn_)
        nr, _ = T.shape
        for i in range(nr):
            for c in self.classes_:
                edist = np.linalg.norm(self.means_[c] - T[i])
                disc[c] = -(edist * edist) / (2 * self.varavg_) + np.log(
                    self.pw_[c])
            y.append(disc.argmax())

        return y

    def score(self, X, y, sample_weight=None):
        """ Return the mean accuracy on the give test data and labels

        This function follows sci-kit idiom:

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """

        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight = sample_weight)

def load_data(f="datasets/synth.tr"):
    """ Assume data format:
    feature1 feature 2 ... label 
    """

    # process training data
    data = np.genfromtxt(f)
    # return all feature columns except last
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    return X, y


def main():

    from sklearn.metrics import accuracy_score

    Xtrain, ytrain = load_data("datasets/synth.tr")
    Xtest, ytest = load_data("datasets/synth.te")
    model = mpp()
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)
    print("model accuracy: {}".format(accuracy_score(ytest, y_model)))

if __name__ == "__main__":
    main()
