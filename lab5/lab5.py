# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: lab5.py                                           #
#    DATE: 11 OCT 2019                                       #
# ********************************************************** #

# MIT 6.034 Lab 5: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math


def log2(x): return math.log(x, 2)


INF = float('inf')


# ##########################################################################
# ######################## IDENTIFICATION TREES ############################
# ##########################################################################


# Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """
    Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification.
    """
    if (id_tree.is_leaf()):
        classification = id_tree.get_node_classification()
        return(classification)
    else:
        child = id_tree.apply_classifier(point)
        return(id_tree_classify_point(point, child))


# Part 1B: Splitting data with a classifier #################################


def split_on_classifier(data, classifier):
    """
    Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each
    feature values to a list of points that have that value.
    """
    data_split = {}
    for point in data:
        feature = classifier.classify(point)
        if (feature not in data_split):
            data_split[feature] = [point]
        else:
            points = data_split[feature]
            points.append(point)
            data_split[feature] = points
    return(data_split)


# Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """
    Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and
    returns the disorder of the branch.
    """
    # Tally target classifier matches for each data point in branch:
    branch = {}
    for point in data:
        feature = target_classifier.classify(point)
        if (feature not in branch):
            branch[feature] = 1
        else:
            n_bc = branch[feature]
            n_bc += 1
            branch[feature] = n_bc

    # Compute branch disorder:
    n_b = len(data)
    D = 0
    for n_bc in branch.values():
        D -= ((n_bc/n_b) * log2(n_bc/n_b))

    D = round(D, 4)
    return(D)


def average_test_disorder(data, test_classifier, target_classifier):
    """
    Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and
    returns the disorder of the feature-test stump.
    """
    # Split data into branch groups:
    data_split = split_on_classifier(data, test_classifier)

    # Determine weighted average branch disorder for test:
    n_test = len(data)
    Q = 0
    for branch_data in data_split.values():
        n_set = len(branch_data)
        D = branch_disorder(branch_data, target_classifier)
        Q += D*(n_set/n_test)

    Q = round(Q, 4)
    return(Q)


""""
# To use your functions to solve part A2 of the "Identification of Trees"
# problem from 2014 Q2, uncomment the lines below and run lab5.py:
for classifier in tree_classifiers:
    print(classifier.name,
          average_test_disorder(tree_data,
                                classifier,
                                feature_test("tree_type")))
"""
# Part 1D: Constructing an ID tree ##########################################


def find_best_classifier(data, possible_classifiers, target_classifier):
    """
    Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.
    Breaks ties by preferring classifiers that appear earlier in the list.
    If best classifier has only one branch, raises NoGoodClassifiersError.
    """
    """
    testQ = {}
    for classifier in possible_classifiers:
        testQ[classifier.name] = (classifier,
                                  average_test_disorder(data,
                                                        classifier,
                                                        target_classifier))
"""
    testQ = []
    for classifier in possible_classifiers:
        testQ.append(average_test_disorder(data,
                                           classifier,
                                           target_classifier))
    zipped = list(zip(possible_classifiers, testQ))
    sorted_classifiers = [c[0] for c in sorted(zipped, key=lambda x: x[1])]

    best_classifier = sorted_classifiers[0]
    data_split = split_on_classifier(data, best_classifier)
    if (len(data_split) == 1):
        msg = 'ERROR: Best classifier has only 1 branch.'
        raise NoGoodClassifiersError(msg)
    return(best_classifier)


# To find the best classifier from 2014 Q2, Part A, uncomment:
print(find_best_classifier(tree_data,
                           tree_classifiers,
                           feature_test("tree_type")))


def construct_greedy_id_tree(data,
                             possible_classifiers,
                             target_classifier,
                             id_tree_node=None):
    """
    Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either:
    (a) perfect classification has been achieved,
    or
    (b) there are no good classifiers left.
    """
    # If no tree defined, initiate root node:
    if (id_tree_node is None):
        # print('\n\nCreating New Tree...')
        id_tree_node = IdentificationTreeNode(target_classifier)

    # print('possible classifiers:', [c.name for c in possible_classifiers])
    # print('target classifier:', id_tree_node.target_classifier.name)
    # print('branch parent:', id_tree_node.get_parent_branch_name())

    # Check for data homogeneity (disorder = 0):
    D = branch_disorder(data, target_classifier)
    # print('Current Node Disorder:', D)
    if (D == 0):
        # print('Node is homogeneous! Assign classification.')
        classification = target_classifier.classify(data[0])
        id_tree_node.set_node_classification(classification)
        return(id_tree_node)
    else:
        # print('Node is not homogeneous. Assign best classifier.')
        try:
            best_classifier = find_best_classifier(data,
                                                   possible_classifiers,
                                                   target_classifier)
            Q = average_test_disorder(data,
                                      best_classifier,
                                      target_classifier)
            # print('BEST CLASSIFIER:', best_classifier)
            # print('Q(BEST CLASSIFIER):', Q)

            children = split_on_classifier(data, best_classifier)
            children_features = children.keys()

            id_tree_node.set_classifier_and_expand(best_classifier,
                                                   children_features)

            children_nodes = id_tree_node.get_branches()
            possible_classifiers.remove(best_classifier)  # remove

            for feature in children_features:
                child_data = children[feature]
                child_node = children_nodes[feature]
                construct_greedy_id_tree(child_data,
                                         possible_classifiers,
                                         target_classifier,
                                         child_node)
        except NoGoodClassifiersError:
            # print('NoGoodClassifiersError: Go back up a level')
            return(id_tree_node)

    # id_tree_node.print_with_data(data)
    return(id_tree_node)


"""
# To construct an ID tree for 2014 Q2, Part A:
print(construct_greedy_id_tree(tree_data,
                               tree_classifiers,
                               feature_test("tree_type")))

# To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
tree_tree = construct_greedy_id_tree(tree_data,
                                     tree_classifiers,
                                     feature_test("tree_type"))
print(id_tree_classify_point(tree_test_point, tree_tree))

# To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
print(construct_greedy_id_tree(angel_data,
                               angel_classifiers,
                               feature_test("Classification")))
print(construct_greedy_id_tree(numeric_data,
                               numeric_classifiers,
                               feature_test("class")))
"""
# Part 1E: Multiple choice ##################################################
ANSWER_1 = 'bark_texture'
ANSWER_2 = 'leaf_shape'
ANSWER_3 = 'orange_foliage'

ANSWER_4 = [2, 3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = 'No'
ANSWER_9 = 'No'


# OPTIONAL: Construct an ID tree with medical data ##########################

# Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = False

if DO_OPTIONAL_SECTION:
    from parse import *
    medical_id_tree = construct_greedy_id_tree(
        heart_training_data, heart_classifiers, heart_target_classifier_discrete)


############################################################################
########################## k-NEAREST NEIGHBORS #############################
############################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = None
BOUNDARY_ANS_2 = None

BOUNDARY_ANS_3 = None
BOUNDARY_ANS_4 = None

BOUNDARY_ANS_5 = None
BOUNDARY_ANS_6 = None
BOUNDARY_ANS_7 = None
BOUNDARY_ANS_8 = None
BOUNDARY_ANS_9 = None

BOUNDARY_ANS_10 = None
BOUNDARY_ANS_11 = None
BOUNDARY_ANS_12 = None
BOUNDARY_ANS_13 = None
BOUNDARY_ANS_14 = None


# Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """
    Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length.
    """
    raise NotImplementedError


def norm(v):
    "Computes length of vector v, represented as a tuple or list of coords."
    raise NotImplementedError


def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance."
    raise NotImplementedError


def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance."
    raise NotImplementedError


def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance."
    raise NotImplementedError


def cosine_distance(point1, point2):
    """
    Given two Points, computes and returns the cosine distance.
    Cosine distance is defined as 1-cos(angle_between(point1, point2)).
    """
    raise NotImplementedError


# Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """
    Given a test point, list of points(the data), an int 0 < k <= len(data),
    and a distance metric(a function), returns a list containing the
    k points from the data that are closest to the test point, according to
    the distance metric.  Breaks ties lexicographically by coordinates.
    """
    raise NotImplementedError


def knn_classify_point(point, data, k, distance_metric):
    """
    Given a test point, list of points(the data), an int 0 < k <= len(data),
    and a distance metric(a function), returns the classification of the
    test point based on its k nearest neighbors, as determined by the
    distance metric. Assumes there are no ties.
    """
    raise NotImplementedError


# To run your classify function on the k-nearest neighbors problem from 2014 Q2
# part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


# Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """
    Given a list of points(the data), an int 0 < k <= len(data), and a
    distance metric(a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float.
    """
    raise NotImplementedError


def find_best_k_and_metric(data):
    """
    Given a list of points(the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among
    the four distance metrics defined above.
    Returns a tuple(k, distance_metric),
    where k is an int and distance_metric is a function.
    """
    raise NotImplementedError


# To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))


# Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = None
kNN_ANSWER_2 = None
kNN_ANSWER_3 = None

kNN_ANSWER_4 = None
kNN_ANSWER_5 = None
kNN_ANSWER_6 = None
kNN_ANSWER_7 = None


# SURVEY ####################################################################

NAME = 'Blake Cole'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
