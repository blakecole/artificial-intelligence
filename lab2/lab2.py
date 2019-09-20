# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: lab2.py                                           #
#    DATE: 18 SEP 2019                                       #
# ********************************************************** #

# MIT 6.034 Lab 2: Search
# Written by 6.034 staff

from search import (
    do_nothing_fn,
    make_generic_search
)

import read_graphs

all_graphs = read_graphs.get_graphs()
GRAPH_0 = all_graphs['GRAPH_0']
GRAPH_1 = all_graphs['GRAPH_1']
GRAPH_2 = all_graphs['GRAPH_2']
GRAPH_3 = all_graphs['GRAPH_3']
GRAPH_FOR_HEURISTICS = all_graphs['GRAPH_FOR_HEURISTICS']


# PART 1: Helper Functions ##################################################

def path_length(graph, path):
    """
    Returns the total length (sum of edge weights) of a path defined by a
    list of nodes coercing an edge-linked traversal through a graph.
    (That is, the list of nodes defines a path through the graph.)
    A path with fewer than 2 nodes should have length of 0.
    You can assume that all edges along the path have a valid numeric weight.
    """
    if (len(path) < 1):
        print('ERROR: path must have at least one node.')
        return(0)
    elif (len(path) == 1):
        return(0)
    else:
        plength = 0

        for n in range(len(path)-1):
            e = graph.get_edge(path[n], path[n+1])
            if (e is None):
                raise ValueError('ERROR: nodes ' + str(path[n]) + ' and ' +
                                 str(path[n+1]) + ' are not connected.')
            plength += e.length
        return(plength)


def has_loops(path):
    """
    Returns True if this path has a loop in it, i.e. if it
    visits a node more than once. Returns False otherwise.
    """
    nodes = dict()
    for n in range(len(path)):

        # If nodes is already in dictionary of nodes, has_loops = True
        if path[n] in nodes:
            return(True)

        # If node not present, add to dictionary of nodes
        else:
            nodes[path[n]] = 0

    # If no node repetitions detected, has_loops = False
    return(False)


def extensions(graph, path):
    """
    Returns a list of paths. Each path in the list should be a one-node
    extension of the input path, where an extension is defined as a path
    formed by adding a neighbor node (of the final node in the path) to
    the path.
    Returned paths should not have loops, i.e. should not visit the same
    node twice. The returned paths should be sorted in lexicographic order.
    """
    # Ensure path has at least one node
    if (len(path) < 1):
        raise ValueError('ERROR: path must have at least one node.')

    # If path has only one element, convert to list for correct indexing
    if(isinstance(path, str)):
        path = [path]

    # Generate list of candidate neighbor nodes
    options = graph.get_neighbors(path[-1])
    if (options and (len(path) > 1)):
        options.remove(path[-2])  # No backtracking allowed.
    options.sort(key=str.lower)   # Sort lexicographically.

    # Generate list of viable paths
    path_list = []
    for n in range(len(options)):
        path_option = list(path)
        path_option.append(options[n])

        # Add non-looping prospective paths to list
        if (not has_loops(path_option)):
            path_list.append(path_option)

    return(path_list)


def sort_by_heuristic(graph, goalNode, nodes):
    """
    Given a list of nodes, sorts them best-to-worst based on the heuristic
    from each node to the goal node. Here, and in general for this lab, we
    consider a smaller heuristic value to be "better" because it represents a
    shorter potential path to the goal. Break ties lexicographically by
    node name.
    """

    # Ensure heuristic data exists for goalNode
    hgoals = graph.heuristic_dict.keys()
    # print('HEURISTIC REFERENCE NODE: ' + goalNode)
    if (goalNode not in hgoals):
        raise ValueError('ERROR: no heuristic data for goalNode found.')

    # Given heuristic data for goalNode exists,
    # Ensure heuristic data exists for ALL requested nodes
    hnodes = graph.heuristic_dict[goalNode].keys()
    if (not set(nodes).issubset(set(hnodes))):
        raise ValueError('ERROR: heuristic data not found for all nodes.')

    heuristic_list = [(n, graph.heuristic_dict[goalNode][n]) for n in nodes]
    # print('REQUESTED NODES:')
    # print(nodes)
    # print(heuristic_list)
    nodes_sorted = [n[0] for n in sorted(heuristic_list,
                                         key=lambda x: (x[1], x[0]))]
    # print('NODES SORTED BY HEURISTIC:')
    # print(nodes_sorted)
    # print('\n')
    return(nodes_sorted)


# The following line allows generic_search (PART 3) to access the extensions,
# and has_loops functions that you just defined in (PART 1).

generic_search = make_generic_search(extensions, has_loops)  # DO NOT CHANGE


# PART 2: Basic Search ######################################################

def basic_dfs(graph, startNode, goalNode):
    """
    Performs a depth-first search on a graph from a specified start
    node to a specified goal node, returning a path-to-goal if it
    exists, otherwise returning None.
    Uses backtracking, but does not use an extended set.
    """
    print('\nstartNode = ' + startNode)
    print('goalNode  = ' + goalNode)

    # Check to ensure startNode and goalNode present on graph
    graph_nodes = graph.nodes
    if (not set([startNode, goalNode]).issubset(set(graph_nodes))):
        print('ERROR: "startNode" OR "goalNode" not found on graph.')
        return(None)

    # INITIALIZE QUEUE:
    queue = [startNode]
    path = queue.pop()

    # COMPUTE PATH: ---------------------------------------------------------
    while (path[-1] != goalNode):
        options = extensions(graph, path)

        # If dead end, len(options)==0, pop next (automatic backtracking)
        if (not options):
            print('--------- DEAD END DETECTED: BACKTRACKING ----------')

        for p in reversed(range(len(options))):
            queue.append(options[p])

        path = queue.pop()
        print('PATH = ' + str(path))
    # -----------------------------------------------------------------------
    return(path)


def basic_bfs(graph, startNode, goalNode):
    """
    Performs a breadth-first search on a graph from a specified start
    node to a specified goal node, returning a path-to-goal if it
    exists, otherwise returning None.
    """
    print('\nstartNode = ' + startNode)
    print('goalNode  = ' + goalNode)

    # Check to ensure startNode and goalNode present on graph
    graph_nodes = graph.nodes
    if (not set([startNode, goalNode]).issubset(set(graph_nodes))):
        print('ERROR: "startNode" OR "goalNode" not found on graph.')
        return(None)

    # INITIALIZE QUEUE:
    queue = [startNode]
    path = queue.pop(0)

    # COMPUTE PATH: ---------------------------------------------------------
    while (path[-1] != goalNode):
        options = extensions(graph, path)

        # If dead end, len(options)==0, pop next (automatic backtracking)
        if (not options):
            print('--------- DEAD END DETECTED: BACKTRACKING ----------')

        for p in range(len(options)):
            queue.append(options[p])

        path = queue.pop(0)
        print('PATH = ' + str(path))
    # -----------------------------------------------------------------------
    return(path)


# PART 3: Generic Search ####################################################

# Generic search requires four arguments (see wiki for more details):
# 1) sort_new_paths_fn: sorts new paths that are added to the agenda
# 2) add_paths_to_front_of_agenda: True if new paths should be added
#                                  to the front of the agenda
# 3) sort_agenda_fn: sort the agenda after adding all new paths
# 4) use_extended_set: True if the algorithm should utilize an extended set

def heuristic_sorting(graph, goalNode, paths):
    heads = [p[-1] for p in paths]
    data = [graph.get_heuristic_value(h, goalNode) for h in heads]
    zipped = list(zip(paths, data))
    sorted_paths = [p[0] for p in sorted(zipped, key=lambda x: x[1])]
    return(sorted_paths)


def path_length_sorting(graph, goalNode, paths):
    length = []
    for p in paths:
        path_length = 0
        for n in range(len(p)-1):
            edge = graph.get_edge(p[n], p[n+1])
            path_length += edge.length
        length.append(path_length)

    zipped = list(zip(paths, length))
    sorted_paths = [p[0] for p in sorted(zipped, key=lambda x: x[1])]
    return(sorted_paths)


def combined_sorting(graph, goalNode, paths):
    # Get heuristic data
    heads = [p[-1] for p in paths]
    data = [graph.get_heuristic_value(h, goalNode) for h in heads]

    # Get path length data
    length = []
    for p in paths:
        path_length = 0
        for n in range(len(p)-1):
            edge = graph.get_edge(p[n], p[n+1])
            path_length += edge.length
        length.append(path_length)

    # Sum(cumulative path length, lower-bound estimate on remaining cost)
    cost = [a + b for a, b in zip(data, length)]
    zipped = list(zip(paths, cost))
    sorted_paths = [p[0] for p in sorted(zipped, key=lambda x: x[1])]
    return(sorted_paths)


generic_dfs = [do_nothing_fn, True, do_nothing_fn, False]

generic_bfs = [do_nothing_fn, False, do_nothing_fn, False]

generic_hill_climbing = [heuristic_sorting, True, do_nothing_fn, False]

generic_best_first = [do_nothing_fn, True, heuristic_sorting, False]

generic_branch_and_bound = [do_nothing_fn, False, path_length_sorting, False]

generic_branch_and_bound_with_heuristic = [do_nothing_fn, False,
                                           combined_sorting, False]

generic_branch_and_bound_with_extended_set = [do_nothing_fn, False,
                                              path_length_sorting, True]

generic_a_star = [do_nothing_fn, False, combined_sorting, True]


# Here is an example of how to call generic_search (uncomment to run):
# my_dfs_fn = generic_search(*generic_dfs)
# my_dfs_path = my_dfs_fn(GRAPH_2, 'S', 'G')
# print(my_dfs_path)

# Or, combining the first two steps:
# my_dfs_path = generic_search(*generic_dfs)(GRAPH_2, 'S', 'G')
# print(my_dfs_path)

# my_bfs_path = generic_search(*generic_bfs)(GRAPH_2, 'S', 'G')
# print(my_bfs_path)

# OPTIONAL: Generic Beam Search

# If you want to run local tests for generic_beam,
# change TEST_GENERIC_BEAM to True:
TEST_GENERIC_BEAM = False

# The sort_agenda_fn for beam search takes fourth argument, beam_width:
# def my_beam_sorting_fn(graph, goalNode, paths, beam_width):
#     # YOUR CODE HERE
#     return sorted_beam_agenda

generic_beam = [do_nothing_fn, False, do_nothing_fn, None]

# Uncomment this to test your generic_beam search:
# print(generic_search(*generic_beam)(GRAPH_2, 'S', 'G', beam_width=2))


# PART 4: Heuristics ########################################################

def is_admissible(graph, goalNode):
    """
    Returns True if this graph's heuristic is admissible; else False.
    A heuristic is admissible if it is either:
    (1) Always exactly correct
    or
    (2) Always overly optimistic
    It never over-estimates the cost to the goal.
    """
    # Check to ensure goalNode present on graph
    graph_nodes = graph.nodes
    if (not set(goalNode).issubset(set(graph_nodes))):
        print('ERROR: "goalNode" not found on graph.')
        return(None)

    for node in graph_nodes:
        heuristic_distance = graph.get_heuristic_value(node, goalNode)
        shortest_path = generic_search(
            *generic_branch_and_bound_with_extended_set)(graph, node, goalNode)
        shortest_path_length = 0
        for i in range(len(shortest_path)-1):
            edge = graph.get_edge(shortest_path[i], shortest_path[i+1])
            shortest_path_length += edge.length
        if (heuristic_distance > shortest_path_length):
            print('ERROR: H(node=' + node + ', goal=' + goalNode +
                  ') = ' + str(heuristic_distance))
            print('       D(node=' + node + ', goal=' + goalNode +
                  ') = ' + str(shortest_path_length))
            print('       heuristic distance to goal must not exceed')
            print('       shortest possible path length from node to goal.')
            return(False)

    return(True)


def is_consistent(graph, goalNode):
    """
    Returns True if this graph's heuristic is consistent; else False.
    A consistent heuristic satisfies the following property for all
    nodes v in the graph:
        Suppose v is a node in the graph, and N is a neighbor of v,
        then, heuristic(v) <= heuristic(N) + edge_weight(v, N)
    In other words, moving from one node to a neighboring node never unfairly
    decreases the heuristic.
    This is equivalent to the heuristic satisfying the triangle inequality.
    """
    # Check to ensure goalNode present on graph
    graph_nodes = graph.nodes
    if (not set(goalNode).issubset(set(graph_nodes))):
        print('ERROR: "goalNode" not found on graph.')
        return(None)

    # For each edge, compare edge length and heuristic distances to goal
    graph_edges = graph.edges
    for edge in graph_edges:
        neighbors = (edge.startNode, edge.endNode)
        heuristic = [graph.get_heuristic_value(
            neighbor, goalNode) for neighbor in neighbors]

        if (abs(heuristic[1] - heuristic[0]) > edge.length):
            print('ERROR: graph heuristics not consistent.')
            print('       |H(node=' + neighbors[0] + ') - H(node='
                  + neighbors[1] + ')| > edge.length')
            return(False)

    return(True)

    # OPTIONAL: Picking Heuristics
    # If you want to run local tests on your heuristics,
    # change TEST_HEURISTICS to True.
    # Note that you MUST have completed generic a_star in order to do this:
TEST_HEURISTICS = False


# heuristic_1: admissible and consistent

[h1_S, h1_A, h1_B, h1_C, h1_G] = [None, None, None, None, None]

heuristic_1 = {'G': {}}
heuristic_1['G']['S'] = h1_S
heuristic_1['G']['A'] = h1_A
heuristic_1['G']['B'] = h1_B
heuristic_1['G']['C'] = h1_C
heuristic_1['G']['G'] = h1_G


# heuristic_2: admissible but NOT consistent

[h2_S, h2_A, h2_B, h2_C, h2_G] = [None, None, None, None, None]

heuristic_2 = {'G': {}}
heuristic_2['G']['S'] = h2_S
heuristic_2['G']['A'] = h2_A
heuristic_2['G']['B'] = h2_B
heuristic_2['G']['C'] = h2_C
heuristic_2['G']['G'] = h2_G


# heuristic_3: admissible but A* returns non-optimal path to G

[h3_S, h3_A, h3_B, h3_C, h3_G] = [None, None, None, None, None]

heuristic_3 = {'G': {}}
heuristic_3['G']['S'] = h3_S
heuristic_3['G']['A'] = h3_A
heuristic_3['G']['B'] = h3_B
heuristic_3['G']['C'] = h3_C
heuristic_3['G']['G'] = h3_G


# heuristic_4: admissible but not consistent, yet A* finds optimal path

[h4_S, h4_A, h4_B, h4_C, h4_G] = [None, None, None, None, None]

heuristic_4 = {'G': {}}
heuristic_4['G']['S'] = h4_S
heuristic_4['G']['A'] = h4_A
heuristic_4['G']['B'] = h4_B
heuristic_4['G']['C'] = h4_C
heuristic_4['G']['G'] = h4_G


# PART 5: Multiple Choice ###################################################

ANSWER_1 = '2'

ANSWER_2 = '4'

ANSWER_3 = '1'

ANSWER_4 = '3'


# SURVEY ####################################################################

NAME = 'Blake Cole'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 15
WHAT_I_FOUND_INTERESTING = 'Loved it.  Very enjoyable.'
WHAT_I_FOUND_BORING = 'Time consuming.'
SUGGESTIONS = 'Would be cool to visulaize the results a bit more.'


###########################################################
### Ignore everything below this line; for testing only ###
###########################################################

# The following lines are used in the online tester. DO NOT CHANGE!

generic_dfs_sort_new_paths_fn = generic_dfs[0]
generic_bfs_sort_new_paths_fn = generic_bfs[0]
generic_hill_climbing_sort_new_paths_fn = generic_hill_climbing[0]
generic_best_first_sort_new_paths_fn = generic_best_first[0]
generic_branch_and_bound_sort_new_paths_fn = generic_branch_and_bound[0]
generic_branch_and_bound_with_heuristic_sort_new_paths_fn = generic_branch_and_bound_with_heuristic[
    0]
generic_branch_and_bound_with_extended_set_sort_new_paths_fn = generic_branch_and_bound_with_extended_set[
    0]
generic_a_star_sort_new_paths_fn = generic_a_star[0]

generic_dfs_sort_agenda_fn = generic_dfs[2]
generic_bfs_sort_agenda_fn = generic_bfs[2]
generic_hill_climbing_sort_agenda_fn = generic_hill_climbing[2]
generic_best_first_sort_agenda_fn = generic_best_first[2]
generic_branch_and_bound_sort_agenda_fn = generic_branch_and_bound[2]
generic_branch_and_bound_with_heuristic_sort_agenda_fn = generic_branch_and_bound_with_heuristic[
    2]
generic_branch_and_bound_with_extended_set_sort_agenda_fn = generic_branch_and_bound_with_extended_set[
    2]
generic_a_star_sort_agenda_fn = generic_a_star[2]

# Creates the beam search using generic beam args, for optional beam tests
beam = generic_search(*generic_beam) if TEST_GENERIC_BEAM else None

# Creates the A* algorithm for use in testing the optional heuristics
if TEST_HEURISTICS:
    a_star = generic_search(*generic_a_star)
