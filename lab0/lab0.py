# MIT 6.034 Lab 0: Getting Started
# Written by jb16, jmn, dxh, and past 6.034 staff

import math
from point_api import Point

# Multiple Choice ###########################################################

# These are multiple choice questions. You answer by replacing
# the symbol 'None' with a letter (or True or False), corresponding
# to your answer.

# True or False: Our code supports both Python 2 and Python 3
# Fill in your answer in the next line of code (True or False):
ANSWER_1 = False

# What version(s) of Python do we *recommend* for this course?
#   A. Python v2.3
#   B. Python V2.5 through v2.7
#   C. Python v3.2 or v3.3
#   D. Python v3.4 or higher
# Fill in your answer in the next line of code ("A", "B", "C", or "D"):
ANSWER_2 = "D"


#############################################################################
# Note: Each function we require you to fill in has brief documentatio      #
# describing what the function should input and output. For more detaile    #
# instructions, check out the lab 0 wiki page!                              #
#############################################################################


# Warmup ####################################################################

def is_even(x):
    """If x is even, returns True; otherwise returns False"""
    if (x % 2 == 0):
        return(True)
    else:
        return(False)


def decrement(x):
    """Given a number x, returns x - 1 unless that would be less than
    zero, in which case returns 0."""
    if (x < 1):
        x = 0
    else:
        x -= 1
    return(x)


def cube(x):
    """Given a number x, returns its cube (x^3)"""
    return(x**3)


# Iteration #################################################################

def is_prime(x):
    "Given a number x, returns True if it is prime; otherwise returns False"
    x = int(x)  # round down to nearest integer
    if ((x <= 1) or (is_even(x) and (x != 2))):
        return(False)
    else:
        # test odd factors only:
        start = 3
        step = 2
        stop = int(math.pow(x, 0.5))+step
        for i in range(start, stop, step):
            if (x % i == 0):
                return(False)
            else:
                pass
    return(True)


def primes_up_to(x):
    """Given a number x,
    returns an in-order list of all primes up to and including x"""
    x = int(x)  # round down to nearest integer
    if (x < 2):
        primes = []
    else:
        primes = [2]
        for num in range(3, x+1, 2):
            if (is_prime(num)):
                primes.append(num)
            else:
                pass
    return(primes)


# Recursion #################################################################

def fibonacci(n):
    """Given a positive int n,
    uses recursion to return the nth Fibonacci number."""
    if ((n < 0) or (not isinstance(n, int))):
        raise ValueError('fibonacci: ensure input >= 0')
    elif (n == 0):
        return(0)
    elif (n == 1):
        return(1)
    else:
        return(fibonacci(n-1) + fibonacci(n-2))


def expression_depth(expr):
    """Given an expression expressed as Python lists,
    uses recursion to return the depth of the expression,
    where depth is defined by the maximum number of nested operations."""
    if (isinstance(expr, list)):
        return(1 + max(expression_depth(i) for i in expr))
    else:
        return(0)


# Built-in data types #######################################################


def remove_from_string(string, letters):
    """Given an original string and a string of letters, returns a new string
    which is the same as the old one except all occurrences of those letters
    have been removed from it."""
    for i in letters:
        string = string.replace(i, '')
    return(string)


def compute_string_properties(string):
    """Given a string of lowercase letters, returns a tuple containing the
    following three elements:
        0. The length of the string
        1. A list of all the characters in the string
           (including duplicates, if any),
           sorted in REVERSE alphabetical order
        2. The number of distinct characters in the string (hint: use a set)
    """
    slen = len(string)
    chars = list(string)
    chars = sorted(chars, key=str.lower, reverse=True)
    unique_chars = len(set(chars))
    return(slen, chars, unique_chars)


def tally_letters(string):
    """Given a string of lowercase letters, returns a dictionary mapping each
    letter to the number of times it occurs in the string."""
    tally = dict()
    for i in string:
        if (i in tally):
            tally[i] += 1
        else:
            tally[i] = 1
    return(tally)


# Functions that return functions ###########################################

def create_multiplier_function(m):
    """Given a multiplier m,
    returns a function that multiplies its input by m."""
    def my_multiplier_fn(value):
        return(value*m)
    return(my_multiplier_fn)


def create_length_comparer_function(check_equal):
    """Returns a function that takes as input two lists.
    If check_equal == True,
    this function will check if the lists are of equal lengths.
    If check_equal == False,
    this function will check if the lists are of different lengths."""
    if (check_equal):
        def list_length_equal(list1, list2):
            if (not (isinstance(list1, list) and isinstance(list2, list))):
                raise ValueError('list_length_equal: \
                ...ERROR. Both inputs must be type:list')
            else:
                return(len(list1) == len(list2))
        return(list_length_equal)

    elif (not check_equal):
        def list_length_unequal(list1, list2):
            if (not(isinstance(list1, list)and isinstance(list2, list))):
                raise ValueError('list_length_equal: \
                ...ERROR. Both inputs must be type:list')
            else:
                return(len(list1) != len(list2))
        return(list_length_unequal)
    else:
        raise ValueError('create_length_comparer_function: \
        ...ERROR. Input must be type:bool')

# Objects and APIs: Copying and modifying objects ###########################


def sum_of_coordinates(point):
    """Given a 2D point (represented as a Point object), returns the sum
    of its X- and Y-coordinates."""
    return(point.getX() + point.getY())


def get_neighbors(point):
    """Given a 2D point (represented as a Point object),
    returns a list of the four points that neighbor it in the four
    coordinate directions. Uses the "copy" method to avoid modifying
    the original point."""
    if (not isinstance(point, Point)):
        raise ValueError('get_neighbors: ERROR. Input must be type:Point')
    else:
        # Get original X and Y positions
        x_pos = point.getX()
        y_pos = point.getY()

        # Create 4 copies of original point
        pointN = point.copy()
        pointE = point.copy()
        pointS = point.copy()
        pointW = point.copy()

        # Create 4 neighbor points
        pointN.setY(y_pos+1)
        pointE.setX(x_pos+1)
        pointS.setY(y_pos-1)
        pointW.setX(x_pos-1)

        return([pointN, pointE, pointS, pointW])


# Using the "key" argument ##################################################

def sort_points_by_Y(list_of_points):
    """Given a list of 2D points (represented as Point objects),
    uses "sorted" with the "key" argument to create and return a list of the
    SAME (not copied) points sorted in decreasing order based on their
    Y coordinates, without modifying the original list."""
    return(sorted(list_of_points,
                  key=lambda Point: Point.getY(),
                  reverse=True))


def furthest_right_point(list_of_points):
    """Given a list of 2D points (represented as Point objects),
    uses "max" with the "key" argument to return the point that is furthest
    to the right (that is, the point with the largest X coordinate)."""
    return(max(list_of_points, key=lambda Point: Point.getX()))


# SURVEY ####################################################################

# How much programming experience do you have, in any language?
#     A. No experience (never programmed before this lab)
#     B. Beginner (just started learning to program,e.g. took one programming class)
#     C. Intermediate (have written programs for a couple classes/projects)
#     D. Proficient (have been programming for multiple years, or wrote programs for many classes/projects)
#     E. Expert (could teach a class on programming, either in a specific language or in general)

PROGRAMMING_EXPERIENCE = "D"


# How much experience do you have with Python?
#     A. No experience (never used Python before this lab)
#     B. Beginner (just started learning, e.g. took 6.0001)
#     C. Intermediate (have used Python in a couple classes/projects)
#     D. Proficient (have used Python for multiple years or in many classes/projects)
#     E. Expert (could teach a class on Python)

PYTHON_EXPERIENCE = "A"


# Finally, the following questions will appear at the end of every lab.
# The first three are required in order to receive full credit for your lab.

NAME = 'Blake Cole'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 14
SUGGESTIONS = 'Perhaps a specialized, specific Python tutorial, or list of best practices, or guidelines for selecting an IDE (or customizing emacs).  I felt like it looks a really long time just to sort of get set up, and get the lay of the land.'
