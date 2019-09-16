# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: lab1.py                                           #
#    DATE: 14 SEP 2019                                       #
# ********************************************************** #

# MIT 6.034 Lab 1: Rule-Based Systems
# Written by 6.034 staff

from production import (
    PASS, FAIL,
    match, populate, simplify,
    variables
)
from production import (
    IF, AND, OR, NOT,
    THEN, DELETE,
    forward_chain, pretty_goal_tree
)
from data import (
    poker_data,
    abc_data,
    minecraft_data,
    simpsons_data,
    black_data,
    sibling_test_data,
    grandparent_test_data,
    anonymous_family_test_data,
    zookeeper_rules,
    zoo_data
)
import pprint

pp = pprint.PrettyPrinter(indent=1)
pprint = pp.pprint

# Part 1: Multiple Choice ############################################

ANSWER_1 = '2'

ANSWER_2 = '4'

ANSWER_3 = '2'

ANSWER_4 = '0'

ANSWER_5 = '3'

ANSWER_6 = '1'

ANSWER_7 = '0'

# Part 2: Transitive Rule #############################################

transitive_rule = IF(AND('(?x) beats (?y)',
                         '(?y) beats (?z)'),
                     THEN('(?x) beats (?z)'))
# PRINT RESULTS:
# pprint(forward_chain([transitive_rule], abc_data))
# pprint(forward_chain([transitive_rule], poker_data))
# pprint(forward_chain([transitive_rule], minecraft_data))


# Part 3: Family Relations ############################################

# Define your rules here.
# friend_rule = IF(AND("person (?x)", "person (?y)"),
#                 THEN("friend (?x) (?y)", "friend (?y) (?x)"))

self_rule = IF('person (?x)',
               THEN('self (?x) (?x)'))

child_rule = IF('parent (?x) (?y)',
                THEN('child (?y) (?x)'))

sibling_rule = IF(AND('child (?y) (?x)',
                      'child (?z) (?x)',
                      NOT('self (?y) (?z)')),
                  THEN('sibling (?y) (?z)'))

cousin_rule = IF(AND('sibling (?w) (?x)',
                     'child (?y) (?w)',
                     'child (?z) (?x)'),
                 THEN('cousin (?y) (?z)'))

aunt_uncle_rule = IF(AND('sibling (?x) (?y)',
                         'parent (?y) (?z)'),
                     THEN('aunt-uncle (?x) (?z)'))

nibling_rule = IF('aunt-uncle (?x) (?y)',
                  THEN('nibling (?y) (?x)'))

grandparent_rule = IF(AND('parent (?x) (?y)',
                          'parent (?y) (?z)'),
                      THEN('grandparent (?x) (?z)'))

grandchild_rule = IF('grandparent (?x) (?y)',
                     THEN('grandchild (?y) (?x)'))

great_grandparent_rule = IF(AND('parent (?x) (?y)',
                                'grandparent (?y) (?z)'),
                            THEN('great-grandparent (?x) (?z)'))

great_grandchild_rule = IF('great-grandparent (?x) (?y)',
                           THEN('great-grandchild (?y) (?x)'))

# Add your rules to this list:
family_rules = [self_rule, child_rule, sibling_rule,
                cousin_rule, aunt_uncle_rule, nibling_rule,
                grandparent_rule, grandchild_rule,
                great_grandparent_rule, great_grandchild_rule]

# Uncomment this to test your data on the Simpsons family:
# pprint(forward_chain(family_rules, simpsons_data, verbose=False))

# These smaller datasets might be helpful for debugging:
# pprint(forward_chain(family_rules, sibling_test_data, verbose=True))
# pprint(forward_chain(family_rules, simpsons_data, verbose=True))
# pprint(forward_chain(family_rules, grandparent_test_data, verbose=True))

# The following should generate 14 cousin relationships, representing 7 pairs
# of people who are cousins:
black_family_cousins = [
    relation for relation in
    forward_chain(family_rules, black_data, verbose=False)
    if "cousin" in relation]

# To see if you found them all, uncomment this line:
# pprint(black_family_cousins)


# Part 4: Backward Chaining ###########################################

# Import additional methods for backchaining

def backchain_to_goal_tree(rules, hypothesis):
    """
    Takes a hypothesis (string) and a list of rules (list
    of IF objects), returning an AND/OR tree representing the
    backchain of possible statements we may need to test
    to determine if this hypothesis is reachable or not.

    This method should return an AND/OR tree, that is, an
    AND or OR object, whose constituents are the subgoals that
    need to be tested. The leaves of this tree should be strings
    (possibly with unbound variables), *not* AND or OR objects.
    Make sure to use simplify(...) to flatten trees where appropriate.
    """
    # raise NotImplementedError

    # Initialize Goal Tree with OR-Node to hypothesis
    goal_tree = OR(hypothesis)
    if (not rules):
        return(hypothesis)

    # Parse input string, split name from description
    split = hypothesis.split(' ', 1)
    pair = {'x': split[0], 'y': split[1]}

    # Initialize boolean to determine if any matches
    hit = False

    # ENTER RULE LIST LOOP
    for z in range(len(rules)):
        print('\nRULE Z' + str(z+1) + ':')
        print('consequent = ' + rules[z].consequent())
        print('antecedent = ' + str(rules[z].antecedent()))
        test = populate(rules[z].consequent(), pair)

        if (hypothesis == test):
            hit = True
            print('(!)(!)(!) hypothesis MATCHED consequent (!)(!)(!)')
            ants = populate(rules[z].antecedent(), pair)
            #print('ants = ' + str(ants) + ' len(ants) = ' + str(len(ants)))
            new = AND()
            for i in range(len(ants)):
                new.append(backchain_to_goal_tree(rules, ants[i]))

            goal_tree.append(new)

    if (not hit):
        # print('end of chain reached')
        return(hypothesis)
    else:
        return(simplify(goal_tree))


# Uncomment this to test out your backward chainer:
pretty_goal_tree(backchain_to_goal_tree(zookeeper_rules, 'opus is a penguin'))


#### Survey #########################################
NAME = 'Blake Cole'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 15
WHAT_I_FOUND_INTERESTING = 'It was cool.'
WHAT_I_FOUND_BORING = 'Not so much boring as incredibly frustrating: I felt like there was inadequate documentation for the required APIs.  Learning how these objects and functions worked was the only real challenge in this lab.  I felt like I knew what needed to be done, but struggled for a long time to find the correct way to implement the solution, given the mildly opaque nature of the APIs.'
SUGGESTIONS = 'More documentation on APIs please!'


###########################################################
### Ignore everything below this line; for testing only ###
###########################################################

# The following lines are used in the tester. DO NOT CHANGE!
print("(Doing forward chaining. This may take a minute.)")
transitive_rule_poker = forward_chain([transitive_rule], poker_data)
transitive_rule_abc = forward_chain([transitive_rule], abc_data)
transitive_rule_minecraft = forward_chain([transitive_rule], minecraft_data)
family_rules_simpsons = forward_chain(family_rules, simpsons_data)
family_rules_black = forward_chain(family_rules, black_data)
family_rules_sibling = forward_chain(family_rules, sibling_test_data)
family_rules_grandparent = forward_chain(family_rules, grandparent_test_data)
family_rules_anonymous_family = forward_chain(
    family_rules, anonymous_family_test_data)
