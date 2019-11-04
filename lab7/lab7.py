# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: lab7.py                                           #
#    DATE: 31 OCT 2019                                       #
# ********************************************************** #

# MIT 6.034 Lab 7: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce

INF = float('inf')

# Part 1: Vector Math #######################################################


def dot_product(u, v):
    """
    Computes the dot product of two vectors u and v, each represented
    as a tuple or list of coordinates. Assume the two vectors are the
    same length.
    """
    if (len(u) != len(v)):
        raise ValueError('ERROR: u and v  must be equal-length vectors)')
    return(sum([(ui * vi) for ui, vi in zip(u, v)]))


def norm(v):
    """
    Computes the norm (length) of a vector v, represented
    as a tuple or list of coords.
    """
    return(dot_product(v, v)**0.5)


# Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """ Computes the expression (w dot x + b) for the given Point x. """
    w = svm.w
    b = svm.b
    x = point.coords
    return(dot_product(w, x) + b)


def classify(svm, point):
    """
    Uses given SVM to classify Point. Assume the point's true classification
    is unknown. Returns +1 or -1, or 0 if point is on boundary.
    """
    point_value = positiveness(svm, point)

    if (point_value > 0):
        return(+1)
    elif (point_value < 0):
        return(-1)
    else:
        return(0)


def margin_width(svm):
    """ Calculate margin width based on the current boundary. """
    width = 2 / norm(svm.w)
    return(width)


def check_gutter_constraint(svm):
    """
    Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == class, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned.
    """
    bad_points = set()

    for point in svm.training_points:
        point_value = positiveness(svm, point)

        if ((point_value > -1) and (point_value < 1)):
            bad_points.add(point)

        if (point in svm.support_vectors):
            if ((point.classification == +1 and point_value != +1)
                    or (point.classification == -1 and point_value != -1)):
                bad_points.add(point)

    return(bad_points)


# Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """
    Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned.
    """
    bad_points = set()

    for point in svm.training_points:
        alpha = point.alpha
        if (point in svm.support_vectors):
            if (alpha <= 0):
                bad_points.add(point)
        else:
            if (alpha != 0):
                bad_points.add(point)

    return(bad_points)


def check_alpha_equations(svm):
    """
    Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned.
    """
    ya_sum = 0
    yax_sum = [0] * len(svm.w)

    for point in svm.training_points:
        yalpha = point.classification * point.alpha
        ya_sum += yalpha
        yax_sum = vector_add(yax_sum, scalar_mult(yalpha, point.coords))

    if (ya_sum == 0 and svm.w == yax_sum):
        return(True)
    else:
        return(False)


# Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """
    Returns the set of training points that are classified incorrectly
    using the current decision boundary.
    """
    misclassified = set()

    for point in svm.training_points:
        if (point.classification != classify(svm, point)):
            misclassified.add(point)

    return(misclassified)


# Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """
    Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM.
    """
    # Reset support vector list, initialize sum(y*alpha*x):
    svm.support_vectors.clear()
    yax_sum = [0, 0]

    # Update w, support_vectors:
    for point in svm.training_points:
        yalpha = point.classification * point.alpha
        yax_sum = vector_add(yax_sum, scalar_mult(yalpha, point.coords))
        if (point.alpha > 0):
            svm.support_vectors.append(point)
    svm.w = yax_sum

    # Update b:
    b_min = INF
    b_max = -INF

    for sv in svm.support_vectors:
        y = sv.classification
        b = y - dot_product(svm.w, sv.coords)

        if ((y == -1) and (b < b_min)):
            b_min = b
        elif ((y == +1) and (b > b_max)):
            b_max = b

    svm.b = (b_min + b_max)/2

    return(svm)


# Part 6: Multiple Choice ###################################################
ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ['A', 'D']
ANSWER_6 = ['A', 'B', 'D']
ANSWER_7 = ['A', 'B', 'D']
ANSWER_8 = []
ANSWER_9 = ['A', 'B', 'D']
ANSWER_10 = ['A', 'B', 'D']

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1, 3, 6, 8]
ANSWER_18 = [1, 2, 4, 5, 6, 7, 8]
ANSWER_19 = [1, 2, 4, 5, 6, 7, 8]

ANSWER_20 = 6

# SURVEY ####################################################################

NAME = 'Blake Cole'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = 'SVMs are cool!  I feel like there is so much moreto explore, and that we only just scratched the surface.  This thought is exciting, inticing, and slightly intimidating.'
WHAT_I_FOUND_BORING = 'Some of the required programs were a bit mundane.  But, I was also glad to finish the lab reasonably quickly for once.'
SUGGESTIONS = 'N/A'
