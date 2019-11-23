# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: lab8.py                                           #
#    DATE: 21 NOV 2019                                       #
# ********************************************************** #

# MIT 6.034 Lab 8: Bayesian Inference
# Written by 6.034 staff

from nets import *


# Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    ancestors = net.get_parents(var)

    for a in list(ancestors):
        ancestors.update(get_ancestors(net, a))

    return(ancestors)


def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    descendants = net.get_children(var)

    for d in list(descendants):
        descendants.update(get_descendants(net, d))

    return(descendants)


def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    variables = set(net.get_variables())
    descendants = get_descendants(net, var)
    nondescendants = variables.difference(descendants.union(var))
    return(nondescendants)


# Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents. Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    descendants = get_descendants(net, var)
    nondescendants = get_nondescendants(net, var)
    parents = net.get_parents(var)
    given_vars = set(givens.keys())

    if (parents.issubset(given_vars) and descendants.isdisjoint(given_vars)):
        reduced_givens = givens.copy()
        rem_set = nondescendants.difference(parents).intersection(given_vars)
        [reduced_givens.pop(key) for key in rem_set]
        return(reduced_givens)
    else:
        return(givens)


def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    # net.CPT_print()
    # print('hypothesis =', hypothesis)
    # print('givens =', givens)

    variables = set(net.get_variables())

    # 1) Check for more than 1 hypothesis:
    hypothesis_var = hypothesis.keys()
    if (len(hypothesis_var) == 1):
        [hypothesis_var] = hypothesis_var
    else:
        raise LookupError('ERROR: multiple hypotheses provided.')
    # print('hypothesis_vars =', hypothesis_var)

    # 2) Ensure hypothesis variable exists
    if (hypothesis_var not in variables):
        raise LookupError('ERROR: hypothesis variable does not exist.')

    # 3) If givens are provided, simplify givens:
    if (givens is not None):
        givens = simplify_givens(net, hypothesis_var, givens)

        # 4) Ensure parents are contained in givens:
        parents = net.get_parents(hypothesis_var)
        if (not parents.issubset(set(givens.keys()))):
            raise LookupError('ERROR: all parent variables not in givens')

        # 5) Ensure no descendants contained in givens:
        descendants = get_descendants(net, hypothesis_var)
        if (not descendants.isdisjoint(set(givens.keys()))):
            raise LookupError('ERROR: descendants detected in givens.')

    P = net.get_probability(hypothesis, givens, infer_missing=True)

    return(P)


def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"

    sorted_hypothesis_vars = net.topological_sort(hypothesis.keys())
    P = 1

    while (sorted_hypothesis_vars):
        sub_hypothesis = sorted_hypothesis_vars.pop()
        sub_hypothesis_dict = {sub_hypothesis: hypothesis[sub_hypothesis]}
        sub_givens = sorted_hypothesis_vars
        sub_givens_dict = {given: hypothesis[given] for given in sub_givens}
        P *= probability_lookup(net, sub_hypothesis_dict, sub_givens_dict)

    return(P)


def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"

    variables = set(net.get_variables())
    joint_probabilities = net.combinations(variables, hypothesis)
    P = 0

    while (joint_probabilities):
        P += probability_joint(net, joint_probabilities.pop())

    return(P)


def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    # net.CPT_print()
    # print('hypothesis =', hypothesis)
    # print('givens =', givens)

    if (not givens):
        P = probability_marginal(net, hypothesis)

    else:
        hypothesis_vars = set(hypothesis.keys())
        givens_vars = set(givens.keys())
        overlap = hypothesis_vars.intersection(givens_vars)
        # print('overlap =', overlap)
        if (overlap):
            for var in overlap:
                # print('hypothesis[', var, '] =', hypothesis[var])
                # print('givens[', var, '] =', givens[var])
                if (hypothesis[var] != givens[var]):
                    P = 0
                    break
                else:
                    P = 1
        else:
            try:
                P = probability_lookup(net, hypothesis, givens)
            except LookupError:
                numP = probability_marginal(net, dict(hypothesis, **givens))
                denP = probability_marginal(net, givens)
                P = numP/denP

    return(P)


def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"

    P = probability_conditional(net, hypothesis, givens)
    return(P)


# Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    variables = net.get_variables()
    N = 0

    for var in variables:
        domain = net.get_domain(var)
        n_domain = len(domain)-1
        parents = net.get_parents(var)
        parents_domain_lengths = [len(net.get_domain(p)) for p in parents]
        N += n_domain*product(parents_domain_lengths)

    return(N)


# Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    domain1 = net.get_domain(var1)
    domain2 = net.get_domain(var2)
    hypothesis1 = {var1: domain1[-1]}
    hypothesis2 = {var2: domain2[-1]}

    if (not givens):
        P1 = probability(net, hypothesis1)
        P2 = probability(net, hypothesis1, hypothesis2)
    else:
        P1 = probability(net, hypothesis1, givens)
        P2 = probability(net, hypothesis1, dict(givens, **hypothesis2))
    return (approx_equal(P1, P2))


def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    if (var1 == var2):
        return(False)

    # --------------------------------------------------------------------- #
    # D-SEPERATION
    # --------------------------------------------------------------------- #

    # 1) Create ancestral graph for var1, var2, givens, and all ancestors
    subnet_vars = {var1, var2}
    subnet_vars.update(get_ancestors(net, var1))
    subnet_vars.update(get_ancestors(net, var2))
    if (givens):
        for var in givens.keys():
            subnet_vars.add(var)
            subnet_vars.update(get_ancestors(net, var))
    subnet = net.subnet(subnet_vars)

    # 2) For each variable in subnet, connect parent variables
    for var in subnet_vars:
        parents = net.get_parents(var)
        if (len(parents) > 1):
            for p1 in parents:
                for p2 in parents:
                    if (p1 != p2):
                        subnet.link(p1, p2)

    # 3) Make subnet edges bidirectional
    subnet.make_bidirectional()

    # 4) Delete all givens and their edges
    if (givens):
        [subnet.remove_variable(key) for key in givens.keys()]

    # 5) If variables are connected, they are not independent
    if (subnet.is_neighbor(var1, var2)):
        return(False)
    elif(subnet.find_path(var1, var2) is not None):
        return(False)
    else:
        return(True)


# SURVEY ####################################################################
NAME = 'Blake Cole'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 12
WHAT_I_FOUND_INTERESTING = 'Honestly, it was not my favorite.'
WHAT_I_FOUND_BORING = 'Some aspects of the lab felt a bit procedural -- like plug-and-chug programming exercises -- and didnt really augment my understanding of Bayes Nets.'
SUGGESTIONS = 'I felt like it would have been useful if some of the requested functions (e.g. get_descendants, get_ancestors...) were included in the Bayes Net API, thus freeing up more time for experimentation and application of Bayes Nets to real problems.  Unfortunately, it felt like most of this lab was focused on developing tools to facilitate the manipulation and inspection of Bayes Nets, rather than their use.'
