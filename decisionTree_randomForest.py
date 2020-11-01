import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter
from sklearn import tree
import graphviz
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score

class DataPoint(object): # DataPoint class helps to group data and methods
    def __init__(self, attr):
        self.clump = attr['clump_thickness']
        self.cellsize = attr['uniformity_cellsize']
        self.cellshape = attr['uniformity_cellshape']
        self.adhesion = attr['marginal_adhesion']
        self.epithelial = attr['single_epithelial_cellsize']
        self.nuclei = attr['bare_nuclei']
        self.chromatin = attr['bland_chromatin']
        self.nucleoli = attr['normal_nucleoli']
        self.mitoses = attr['mitoses']
        self.outcome = attr['benign_malignant']

    def attributes_vector(self):
        return np.array([self.clump, self.cellsize, self.cellshape, self.adhesion, self.epithelial, self.nuclei, self.chromatin, self.nucleoli, self.mitoses, self.outcome])

    def __str__(self):
        return "Clump_thickness: {}, Uniformity of Cell Size: {}, Uniformity of Cell Shape: {}, Marginal Adhesion: {}, Single Epithelial Cell Size: {}, Bare Nuclei: {}, Bland Chromatin: {}, Normal Nucleoli: {}, Mitoses: {}, Benign or Malignant: {}".format(self.clump, self.cellsize, self.cellshape, self.adhesion, self.epithelial, self.nuclei, self.chromatin, self.nucleoli, self.mitoses, self.outcome)
        
def parse_dataset(filename):
    data_file = open(filename, 'r')  # Open File "to read"
    dataset = []  # List to hold Datapoint objects

    for line in data_file:
        clump,cellsize,cellshape,adhesion,epithelial,nuclei,chromatin,nucleoli,mitoses,outcome = line.strip().split(',')  # strip() removes '\n', and split(',') splits the line at tabs
        dataset.append(DataPoint({'clump_thickness':int(clump), 'uniformity_cellsize':int(cellsize), 'uniformity_cellshape':int(cellshape), 'marginal_adhesion':int(adhesion), 'single_epithelial_cellsize':int(epithelial), 'bare_nuclei':int(nuclei), 'bland_chromatin':int(chromatin), 'normal_nucleoli':int(nucleoli), 'mitoses':int(mitoses), 'benign_malignant':int(outcome)}))  # Create DataPoint object for the given data

    print("Number of benign cases - {0} , Number of malignant cases - {1}".format(len([i for i in dataset if i.outcome == 2]), len([i for i in dataset if i.outcome == 4])))
    return dataset

train_set = parse_dataset('hw4_train.csv')

def plot_histo(dataset):
    feat1 = [case.clump for case in dataset]
    feat2 = [case.cellsize for case in dataset]
    feat3 = [case.cellshape for case in dataset]
    feat4 = [case.adhesion for case in dataset]
    feat5 = [case.epithelial for case in dataset]
    feat6 = [case.nuclei for case in dataset]
    feat7 = [case.chromatin for case in dataset]
    feat8 = [case.nucleoli for case in dataset]
    feat9 = [case.mitoses for case in dataset]
    feat10 = [case.outcome for case in dataset]

    plt.figure(figsize = (10, 6))

    h1 = plt.subplot(251)
    h1.hist(feat1, bins = 9)
    plt.title("Clump Thickness")

    h2 = plt.subplot(252)
    h2.hist(feat2, bins = 9)
    plt.title("Uniformity of Cell Size")

    h3 = plt.subplot(253)
    h3.hist(feat3, bins = 9)
    plt.title("Uniformity of Cell Shape")

    h4 = plt.subplot(254)
    h4.hist(feat4, bins = 9)
    plt.title("Marginal Adhesion")

    h5 = plt.subplot(255)
    h5.hist(feat5, bins = 9)
    plt.title("Single Epithelial Cell Size")

    h6 = plt.subplot(256)
    h6.hist(feat6, bins = 9)
    plt.title("Bare Nuclei")

    h7 = plt.subplot(257)
    h7.hist(feat7, bins = 9)
    plt.title("Bland Chromatin")

    h8 = plt.subplot(258)
    h8.hist(feat8, bins = 9)
    plt.title("Normal Nucleoli")

    h9 = plt.subplot(259)
    h9.hist(feat9, bins = 9)
    plt.title("Mitoses")

    h10 = plt.subplot(2,5,10)
    h10.hist(feat10, bins = 2)
    plt.title("Tumor Classification")

    plt.suptitle('Histograms for Tumor Features')
    plt.show()

# plot_histo(train_set)


def cond_entropy(dataset):
    feats = list(dataset[0].__dict__) # put all attributes in list
    for i in range(9): # each feature
        caseTotal = len(dataset)
        entropy = 0 # entropy on each discrete value
        for j in range (1,11): # each possible value
            featAttr = 0
            benign = len([case for case in dataset if case.attributes_vector()[i] == j and case.attributes_vector()[9] == 2])
            malignant = len([case for case in dataset if case.attributes_vector()[i] == j and case.attributes_vector()[9] == 4])
            valTotal = benign + malignant
            if valTotal != 0: 
                fraction1 = benign/valTotal # benign case
                fraction2 = malignant/valTotal # malignant case
                if fraction1 != 0:
                    featAttr += fraction1 * math.log(fraction1)
                if fraction2 != 0:
                    featAttr += fraction2 * math.log(fraction2)
            entropy += valTotal/caseTotal * (-featAttr)
        print("Entropy of feature ", feats[i], " is ", entropy)

cond_entropy(train_set)

# For question 3, import training data in matrix format to facilitate construction of X and Y matrices
training_set = np.array(list(csv.reader(open("hw4_train.csv", "r"), delimiter=","))).astype("int")

testing_set = np.array(list(csv.reader(open("hw4_test.csv", "r"), delimiter=","))).astype("int")

def decision_tree(dataset, testset):
    y = dataset[:, -1] # last column of matrix
    X = dataset[:, :-1] # all columns upto last column
    
    hpAcc = [] # store average accuracy for each hyperparameter value (tree depth) tested
    # try out different values for tree depth
    for i in range(3,8):
        DTC = DecisionTreeClassifier(max_depth=i) # define decision tree classifier

        # perform 5-fold cross-validation
        acc = cross_val_score(estimator = DTC, X=X, y=y, cv=5)
        hpAcc.append((i,acc.mean()))
        
    
    best_hp = max(hpAcc, key=itemgetter(1))[0]
    
    DTC_opt = DecisionTreeClassifier(max_depth=best_hp)
    DTC_opt.fit(X, y)

    # visualizing the decision tree with optimal hyperparameter
    # features = ["clump", "cellsize", "cellshape", "adhesion", "epithelial", "nuclei", "chromatin", "nucleoli", "mitoses"]
    # txttree = export_text(DTC_opt, feature_names=features)
    # print(txttree)

    X_test = testset[:,:-1]
    y_test = testset[:,-1]
    y_predict = DTC_opt.predict(X_test)
    testAcc =  accuracy_score(y_test, y_predict)
    print("Accuracy on test set is ", testAcc)

decision_tree(training_set, testing_set)