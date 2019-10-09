# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: lab4.py                                           #
#    DATE: 6 OCT 2019                                        #
# ********************************************************** #

# MIT 6.034 Lab 4: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


# Part 1: Warmup ############################################################

def has_empty_domains(csp):
    """
    Returns True if problem has one or more empty domains, otherwise False
    """
    variables = csp.get_all_variables()
    for v in variables:
        domain = csp.get_domain(v)
        if (not domain):
            return(True)
    return(False)


def check_all_constraints(csp):
    """
    Returns False if the problem's assigned values violate some constraint,
    otherwise True
    """
    variables = csp.get_all_variables()
    constraints = csp.get_all_constraints()
    # print('\n\nconstraints =', constraints)
    for v1 in variables:
        for v2 in variables:
            v1_assignment = csp.get_assignment(v1)
            v2_assignment = csp.get_assignment(v2)
            # print('(v1,v2) = (' + str(v1) + ',' + str(v2) + ') = (' +
            #       str(v1_assignment) + ',' +
            #       str(v2_assignment) + ')')
            if ((v1_assignment is not None) and (v2_assignment is not None)):
                for c in constraints:
                    if (v1 == c.var1 and v2 == c.var2):
                        if (not c.check(v1_assignment, v2_assignment)):
                            # print(c, 'VIOLATED!')
                            return(False)
    return(True)


# Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem):
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (number of problems popped off agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    presort = False
    # ---------------------
    #      PRE-SORT
    # ---------------------
    # get neighbors:
    variables = problem.get_all_variables()
    neighbors = {}
    for v in variables:
        neighbors[v] = problem.get_neighbors(v)

    # sort variables by domain size, then by number of neighbor constraints:
    domains = problem.domains
    unassigned_vars_ordered = [n for n in sorted(domains,
                                                 key=lambda k:
                                                 (len(domains[k]),
                                                  len(neighbors[k])),
                                                 reverse=True)]
    if (presort):
        problem.set_unassigned_vars_order(unassigned_vars_ordered)

    # ---------------------
    #   INITIALIZE QUEUE
    # ---------------------
    # print('\n\nPROBLEM:\n', problem)
    # print(' * neighbors:', neighbors)

    queue = [problem.copy()]
    extensions = 0
    solution = None

    # ---------------------
    #   COMPUTE SOLUTION
    # ---------------------
    while (queue):
        subprob = queue.pop()
        extensions += 1

        if ((check_all_constraints(subprob)) and
                (not has_empty_domains(subprob))):
            if (not subprob.unassigned_vars):
                solution = subprob.assignments
                print(' * SOLUTION = ', (solution, extensions))
                return((solution, extensions))
            else:
                newvar = subprob.pop_next_unassigned_var()
                for val in reversed(subprob.get_domain(newvar)):
                    queue.append(subprob.copy().set_assignment(newvar, val))

    # if queue empty, and there are still no solutions, return None
    print(' * SOLUTION = ', (solution, extensions))
    return((None, extensions))


# QUESTION 1: How many extensions does it take to solve the Pokemon
#             problem with DFS?
#       Hint: Use get_pokemon_problem() to get a copy of the Pokemon problem
#             each time you want to solve it with a different search method.
full_solution = solve_constraint_dfs(get_pokemon_problem())
ANSWER_1 = full_solution[1]


# Part 3: Forward Checking ##################################################

def eliminate_from_neighbors(csp, var):
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns alphabetically sorted list of the neighboring
    variables whose domains were reduced, w/ each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    # get neighbors:
    neighbors = csp.get_neighbors(var)
    # print(' * neighbors:', neighbors)

    # get domain for input variable
    domain1 = csp.get_domain(var)

    # initialize set of variables which have had their domains altered:
    eliminated_set = set()

    # check each value in neighbor with all values in var
    for n in neighbors:
        # ----------- FOR EACH NEIGHBOR ---------------
        # print('CURRENT NEIGHBOR:', n)
        death_row = []
        constraints = csp.constraints_between(n, var)
        assigned = csp.get_assignment(n)
        domain2 = csp.get_domain(n)
        for val2 in domain2:
            # ------- FOR EACH VALUE IN NEIGHBOR ------
            remove = True
            for val1 in domain1:
                # --- FOR EACH VALUE IN VAR -----------
                all_ok = True
                for c in constraints:
                    if(not c.check(val2, val1)):
                        all_ok = False

                # print('(val2,val1) = (', val2, ',', val1, ') :', c)
                if (all_ok):
                    remove = False
                    # print('val2 =', val2, 'safe! move on, please.')
                    break
                # -------------------------------------
            if (remove and val2 is not assigned):
                death_row.append(val2)
                # print(val2, 'slated for removal.')
                # print('death_row =', death_row)
            # -----------------------------------------
        for d in death_row:
            csp.eliminate(n, d)
            eliminated_set.add(n)
        # print('eliminated_set =', eliminated_set)

        if (has_empty_domains(csp)):
            #print('DOMAIN COMPLETELY REDUCED! eliiminate_from_neighbors=None')
            return(None)
        # ---------------------------------------------

    else:
        solution = sorted(eliminated_set, key=str.lower)
        return(solution)


# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors


def solve_constraint_forward_checking(problem):
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    # ---------------------
    #   INITIALIZE QUEUE
    # ---------------------
    # print('\n\nPROBLEM:\n', problem)

    queue = [problem.copy()]
    extensions = 0
    solution = None

    # ---------------------
    #   COMPUTE SOLUTION
    # ---------------------
    while (queue):
        subprob = queue.pop()
        extensions += 1

        if ((check_all_constraints(subprob)) and
                (not has_empty_domains(subprob))):
            if (not subprob.unassigned_vars):
                solution = subprob.assignments
                print(' * SOLUTION = ', (solution, extensions))
                return((solution, extensions))
            else:
                newvar = subprob.pop_next_unassigned_var()
                for val in reversed(subprob.get_domain(newvar)):
                    newprob = subprob.copy().set_assignment(newvar, val)
                    eliminate_from_neighbors(newprob, newvar)
                    queue.append(newprob)

    # if queue empty, and there are still no solutions, return None
    print(' * SOLUTION = ', (solution, extensions))
    return((None, extensions))


# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#             with DFS and forward checking?
full_solution = solve_constraint_forward_checking(get_pokemon_problem())
ANSWER_2 = full_solution[1]


# Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None):
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables
    in their default order.
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.
    Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    if (queue is None):
        queue = csp.get_all_variables()
    else:  # queue passed from search function after assigning variable
        if (isinstance(queue, list)):
            pass
        elif(isinstance(queue, str)):
            queue = [queue]

    # print('\n\nDOMAIN REDUCTION: ORIGINAL QUEUE =', queue)

    dequeued = []
    while (queue):
        var = queue.pop(0)
        dequeued.append(var)
        # print('dequeued =', dequeued)
        forward_check = eliminate_from_neighbors(csp, var)
        if (forward_check is not None):
            for n in forward_check:
                if (n not in queue):
                    # print('forward_check =', forward_check)
                    queue.append(n)

        if (has_empty_domains(csp)):
            return(None)

    return(dequeued)


# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#             with DFS (no forward checking) if you do domain reduction
#             before solving it?
pokemon_problem = get_pokemon_problem()
domain_reduction(pokemon_problem)
full_solution = solve_constraint_dfs(pokemon_problem)
ANSWER_3 = full_solution[1]


def solve_constraint_propagate_reduced_domains(problem):
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """

    # ---------------------
    #   INITIALIZE QUEUE
    # ---------------------
    # print('\n\nPROBLEM:\n', problem)

    queue = [problem.copy()]
    extensions = 0
    solution = None

    # ---------------------
    #   COMPUTE SOLUTION
    # ---------------------
    while (queue):
        subprob = queue.pop()
        extensions += 1

        if ((check_all_constraints(subprob)) and
                (not has_empty_domains(subprob))):
            if (not subprob.unassigned_vars):
                solution = subprob.assignments
                print(' * SOLUTION = ', (solution, extensions))
                return((solution, extensions))
            else:
                newvar = subprob.pop_next_unassigned_var()
                for val in reversed(subprob.get_domain(newvar)):
                    newprob = subprob.copy().set_assignment(newvar, val)
                    domain_reduction(newprob, newvar)
                    queue.append(newprob)

    # if queue empty, and there are still no solutions, return None
    print(' * SOLUTION = ', (solution, extensions))
    return((None, extensions))


# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#             with forward checking and propagation through reduced domains?
pokemon_problem = get_pokemon_problem()
full_solution = solve_constraint_propagate_reduced_domains(pokemon_problem)
ANSWER_4 = full_solution[1]


# Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None):
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    if (queue is None):
        queue = csp.get_all_variables()
    else:  # queue passed from search function after assigning variable
        if (isinstance(queue, list)):
            pass
        elif(isinstance(queue, str)):
            queue = [queue]

    # print('\n\nDOMAIN REDUCTION: ORIGINAL QUEUE =', queue)

    dequeued = []
    while (queue):
        var = queue.pop(0)
        dequeued.append(var)
        # print('dequeued =', dequeued)
        forward_check = eliminate_from_neighbors(csp, var)
        if (forward_check is not None):
            for n in forward_check:
                if(enqueue_condition_fn(csp, n) and (n not in queue)):
                    queue.append(n)

        if (has_empty_domains(csp)):
            return(None)

    return(dequeued)


def condition_domain_reduction(csp, var):
    """
    Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False
    """
    return(True)


def condition_singleton(csp, var):
    """
    Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False
    """
    if (len(csp.get_domain(var)) == 1):
        return(True)
    else:
        return(False)


def condition_forward_checking(csp, var):
    """
    Returns True if var should be enqueued under the forward-checking
    condition, otherwise False
    """
    return(False)


# Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None):
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    # ---------------------
    #   INITIALIZE QUEUE
    # ---------------------
    # print('\n\nPROBLEM:\n', problem)

    queue = [problem.copy()]
    extensions = 0
    solution = None

    # ---------------------
    #   COMPUTE SOLUTION
    # ---------------------
    while (queue):
        subprob = queue.pop()
        extensions += 1

        if ((check_all_constraints(subprob)) and
                (not has_empty_domains(subprob))):
            if (not subprob.unassigned_vars):
                solution = subprob.assignments
                print(' * SOLUTION = ', (solution, extensions))
                return((solution, extensions))
            else:
                newvar = subprob.pop_next_unassigned_var()
                for val in reversed(subprob.get_domain(newvar)):
                    newprob = subprob.copy().set_assignment(newvar, val)
                    if (enqueue_condition):
                        propagate(enqueue_condition, newprob, newvar)
                    queue.append(newprob)  # do ONLY this for DFS (no prop)

    # if queue empty, and there are still no solutions, return None
    print(' * SOLUTION = ', (solution, extensions))
    return((None, extensions))


# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#             w/ forward checking and propagation through singleton domains?
#            (Don't use domain reduction before solving it.)
pokemon_problem = get_pokemon_problem()
full_solution = solve_constraint_generic(pokemon_problem, condition_singleton)
ANSWER_5 = full_solution[1]


# Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n):
    """
    Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints.
    """
    if (abs(m-n) == 1):
        return(True)
    else:
        return(False)


def constraint_not_adjacent(m, n):
    """
    Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints.
    """
    if(abs(m-n) != 1):
        return(True)
    else:
        return(False)


def all_different(variables):
    """
    Returns a list of constraints, with one difference constraint between
    each pair of variables.
    """
    constraints = []
    while (variables):
        v1 = variables.pop()
        for v2 in variables:
            constraints.append(Constraint(v1, v2, constraint_different))
    return(constraints)


# SURVEY ####################################################################

NAME = 'Blake Cole'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 15
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
