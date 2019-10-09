# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: lab3.py                                           #
#    DATE: 25 SEP 2019                                       #
# ********************************************************** #

# MIT 6.034 Lab 3: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

# Part 1: Utility Functions #################################################


def is_game_over_connectfour(board):
    """
    Returns True if game is over, otherwise False.
    """
    # TEST 1: All columns full?
    full = True
    for c in range(board.num_cols):
        if (not board.is_column_full(c)):
            full = False
            break

    # TEST 2: Four consecutive pieces?
    four = False
    for chain in board.get_all_chains():
        if (len(chain) >= 4):
            four = True
            break

    # IF: Either test succeeds, game is over
    if (full or four):
        return(True)
    else:
        return(False)


def next_boards_connectfour(board):
    """
    Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made.
    """
    moves = []

    if (is_game_over_connectfour(board)):
        return(moves)

    for c in range(board.num_cols):
        if (not board.is_column_full(c)):
            moves.append(board.add_piece(c))

    return(moves)


def endgame_score_connectfour(board, is_current_player_maximizer):
    """
    Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie.
    1) Check to see if the game is over.  This indicates that either:
       a) Board is full (all columns full);
       OR
       b) Someone won (at least one 4-piece chain)
    2) Check to see if the board is full.
       a) If not full, somewon won. Assign points to previous player.
       b) If full, tie game. Assign no points, return(0)
    """
    if (not is_game_over_connectfour(board)):
        raise ValueError('ERROR: game not over, cannot assign score.')

    full = True
    for c in range(board.num_cols):
        if (not board.is_column_full(c)):
            full = False
            break

    score = 1000
    if (not full):
        if (is_current_player_maximizer):  # previous player: minimizer
            return(-score)
        else:                              # previous player: maximizer
            return(score)
    else:                                  # tie game
        return(0)


def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """
    Given an endgame board, returns endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner.
    """
    if (not is_game_over_connectfour(board)):
        raise ValueError('ERROR: game not over, cannot assign score.')

    full = True
    for c in range(board.num_cols):
        if (not board.is_column_full(c)):
            full = False
            break

    # REWARD FUNCTION:
    # 1) Establish function x-limits based on board size
    rows = board.num_rows
    cols = board.num_cols
    size = rows*cols

    # 2) Establish function y-limits based on base score and maximium bonus
    base = 1000
    bonus = 1000

    # 3) Determine how many plays were made by the winner, assign reward
    plays = board.count_pieces(current_player=False)
    linear_reward = int(bonus*(1 - (plays-4)/(size-4)))

    # decay_factor = (1/bonus)**(1/(size-4))
    # power_reward = int(bonus*(decay_factor**(plays-4)))

    score = base + linear_reward
    if (not full):
        if (is_current_player_maximizer):  # previous player: minimizer
            return(-score)
        else:                              # previous player: maximizer
            return(score)
    else:                                  # tie game
        return(0)


def heuristic_connectfour(board, is_current_player_maximizer):
    """
    Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer.
    """
    if (is_game_over_connectfour(board)):
        raise ValueError('ERROR: game over, cannot assign in-game score.')

    # REWARD FUNCTION:
    # 1) Establish function x-limits based on board size
    rows = board.num_rows
    cols = board.num_cols
    size = rows*cols

    # 2) Establish function y-limits based on maximium bonus
    bonus = 1000
    max_1_chains = size/2          # max pieces per player
    max_2_chains = max_1_chains/2  # max chains of 2 per player
    max_3_chains = max_1_chains/3  # max chains of 3 per player

    # 3) WEIGHTS
    c1 = int((bonus/16)/max_1_chains)  # IF(all singles): 62.5
    c2 = int((bonus/4)/max_2_chains)  # IF(all chains of 2): 250
    c3 = int((bonus/1)/max_3_chains)  # IF(all chains of 3): 1000

    # 4) Determine number of chains for each player
    # Current Player
    x1 = 0
    x2 = 0
    x3 = 0
    for chain in board.get_all_chains(current_player=True):
        if (len(chain) == 1):
            x1 += 1
        elif (len(chain) == 2):
            x2 += 1
        elif (len(chain) == 3):
            x3 += 1
        else:
            raise ValueError('ERROR: Unexpected chain length')

    score1 = c1*x1 + c2*x2 + c3*x3  # linear polynomial reward

    # Previous Player
    x1 = 0
    x2 = 0
    x3 = 0
    for chain in board.get_all_chains(current_player=False):
        if (len(chain) == 1):
            x1 += 1
        elif (len(chain) == 2):
            x2 += 1
        elif (len(chain) == 3):
            x3 += 1
        else:
            raise ValueError('ERROR: Unexpected chain length')

    score2 = c1*x1 + c2*x2 + c3*x3  # linear polynomial reward

    score = score2 - score1
    if (abs(score) >= 1000):
        raise ValueError('WARNING! score > 1000 for unfinished game')

    if (is_current_player_maximizer):
        return(-score)
    else:
        return(score)


# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard,
# before the game has started:
state_starting_connectfour = AbstractGameState(
    snapshot=ConnectFourBoard(),
    is_game_over_fn=is_game_over_connectfour,
    generate_next_states_fn=next_boards_connectfour,
    endgame_score_fn=endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard
# "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(
    snapshot=NEARLY_OVER,
    is_game_over_fn=is_game_over_connectfour,
    generate_next_states_fn=next_boards_connectfour,
    endgame_score_fn=endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard
# "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(
    snapshot=BOARD_UHOH,
    is_game_over_fn=is_game_over_connectfour,
    generate_next_states_fn=next_boards_connectfour,
    endgame_score_fn=endgame_score_connectfour_faster)


# Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API,
#       not ConnectFourBoard.

def dfs_maximizing(state):
    """
    Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
        0. the best path (a list of AbstractGameState objects),
        1. the score of the leaf node (a number), and
        2. the number of static evaluations performed (a number)
    """
    # WHAT ARE WE DEALING WITH?
    print(state)

    best_path = ([state], 0, 0)
    options = state.generate_next_states()
    # If dead end, len(options)==0, pop next (automatic backtracking)
    if (not options):
        print('--------- NO MOVES AVAILABLE: ARRIVED AT LEAF ----------')

    for p in reversed(range(len(options))):
        test = options[p].generate_next_states()
        for t in reversed(range(len(test))):
            print(test[t].generate_next_states())
            print(test[t])
            print(test[t].is_game_over())
            print(test[t].get_snapshot())
            print(test[t].describe_previous_move())
            print(test[t].get_endgame_score(is_current_player_maximizer=True))
        # queue.append(options[p])


# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:
# pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize=True):
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    raise NotImplementedError


# Uncomment line below to try minimax_endgame_search on an AbstractGameState
# representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True):
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    raise NotImplementedError


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True):
    """"Performs minimax with alpha-beta pruning. Same return type
    as dfs_maximizing."""
    raise NotImplementedError


# Uncomment line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True):
    """
    Runs minimax with alpha-beta pruning. At each level, updates anytime
    value with the tuple returned from minimax_search_alphabeta.
    Returns anytime_value.
    """
    raise NotImplementedError


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


# Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


# SURVEY ###################################################

NAME = 'Blake Cole'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = 'I enjoyed learning about adverserial search, and was grateful for the opportunity to implement the methodologies we covered in class.  I didnt have time to finish, which was disappointing, but so it goes.  Maybe I can finish up a day or two late.'
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = 'Again, I feel that the API documentation could be more thorough.  For example, in most online python documentation, example usage cases are given, which go a long way in helping people understand how each function works.  I feel like I spend an inordinate amount of time just learning how to use the APIs, when that time would be better used playing with different approaches to the actual problem at hand.'
