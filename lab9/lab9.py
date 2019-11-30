# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: lab9.py                                           #
#    DATE: 25 NOV 2019                                       #
# ********************************************************** #

# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


# Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """
    Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights.
    """
    N = len(training_points)
    weights = {point: make_fraction(1, N) for point in training_points}

    return(weights)


def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """
    Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates.
    """
    weak_classifiers = classifier_to_misclassified.keys()
    error_rate = {}

    for classifier in weak_classifiers:
        misclassified_points = classifier_to_misclassified[classifier]
        weight_sum = 0
        for point in misclassified_points:
            weight_sum += point_to_weight[point]
        error_rate[classifier] = weight_sum

    return(error_rate)


def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """
    Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier
    has error rate 1/2.  best* means 'smallest error rate' if use_smallest_
    error is True, otherwise 'error rate furthest from 1/2'.
    """
    c2er = classifier_to_error_rate

    if (use_smallest_error):
        def keyfn(x): return((x[1], x[0]))
        c2er_sort = sorted(c2er.items(), key=keyfn)
    else:
        def keyfn(x): return ((-abs(x[1] - make_fraction(1, 2)), x[0]))
        c2er_sort = sorted(c2er.items(), key=keyfn)

    best_classifier = c2er_sort[0]

    if (best_classifier[1] == make_fraction(1, 2)):
        raise NoGoodClassifiersError('BEST CLASSIFER IS NO GOOD. TRY AGAIN.')
    else:
        return(best_classifier[0])


def calculate_voting_power(error_rate):
    """
    Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier.
    """
    if (error_rate == 0):
        alpha = INF
    elif(error_rate == 1):
        alpha = -INF
    else:
        alpha = 0.5 * ln((1-error_rate)/error_rate)

    return(alpha)


def get_overall_misclassifications(H, training_points,
                                   classifier_to_misclassified):
    """
    Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples.
    """
    misclassified_points = set()

    for point in training_points:
        alpha_sum = 0
        if (isinstance(H, dict)):
            H = H.items()
        for weak_classifier_tuple in H:
            weak_classifier = weak_classifier_tuple[0]
            alpha = weak_classifier_tuple[1]
            if (point in classifier_to_misclassified[weak_classifier]):
                alpha_sum -= alpha
            else:
                alpha_sum += alpha
        if (alpha_sum <= 0):
            misclassified_points.add(point)

    return(misclassified_points)


def is_good_enough(H, training_points, classifier_to_misclassified,
                   mistake_tolerance=0):
    """
    Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples.
    """
    mistakes = get_overall_misclassifications(H, training_points,
                                              classifier_to_misclassified)
    return(len(mistakes) <= mistake_tolerance)


def update_weights(point_to_weight, misclassified_points, error_rate):
    """
    Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight.
    """
    for point in point_to_weight.keys():
        old_weight = point_to_weight[point]
        if (point in misclassified_points):
            update = make_fraction(1, 2) * make_fraction(1, error_rate)
        else:
            update = make_fraction(1, 2) * make_fraction(1, 1-error_rate)

        point_to_weight[point] = old_weight*update

    return(point_to_weight)


# Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """
    Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples.
    """
    H = list()
    GOOD_ENOUGH = False

    # 1) Initialize weights equally for all training points (dict):
    point_weights = initialize_weights(training_points)

    while ((len(H) < max_rounds) and (not GOOD_ENOUGH)):

        # 2) Calculate error rates for each weak classifier (dict):
        classifier_error = calculate_error_rates(point_weights,
                                                 classifier_to_misclassified)
        try:
            # 3) Add "best" weak classifier to aggregate classifier, H (str):
            best_weak_classifier = pick_best_classifier(classifier_error,
                                                        use_smallest_error)

            # 4) For "best" weak classifier, calculate alpha (float):
            error_rate = classifier_error[best_weak_classifier]
            alpha = calculate_voting_power(error_rate)

            # 5) Append weak classifier and voting power (alpha) to H:
            H.append((best_weak_classifier, alpha))

        except NoGoodClassifiersError:
            return(H)

        # 6) Is the aggregate classifer good enough?
        GOOD_ENOUGH = is_good_enough(H, training_points,
                                     classifier_to_misclassified,
                                     mistake_tolerance)

        # 7) If not, update weights and add another weak classifier:
        bad_pts = classifier_to_misclassified[best_weak_classifier]
        point_weights = update_weights(point_weights, bad_pts, error_rate)

    return(H)


# SURVEY ####################################################################
NAME = 'Blake Cole'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 7
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
