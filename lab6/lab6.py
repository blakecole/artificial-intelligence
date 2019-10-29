# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: lab6.py                                           #
#    DATE: 20 OCT 2019                                       #
# ********************************************************** #

# MIT 6.034 Lab 6: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


# Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2, 1]

nn_cross = [2, 2, 1]

nn_stripe = [3, 1]

nn_hexagon = [6, 1]

nn_grid = [4, 2, 1]


# Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if (x >= threshold):
        return(1)
    else:
        return(0)


def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return(pow(1 + pow(e, -steepness*(x-midpoint)), -1))


def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return(max(0, x))


# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return(-0.5*pow(desired_output-actual_output, 2))


# Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given:
      (1) A node (as an input or as a neuron),
      (2) A dictionary mapping input names to their values
      (3) A dictionary mapping neuron names to their outputs
    Returns:
      (1) Output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError(
            """Node '{}' not found in either the input values or neuron
            outputs dictionary.""".format(node))

    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node

    raise TypeError(
        "Node argument is {}; should be either a string or a number."
        .format(node))


def forward_prop(net, input_values, threshold_fn=stairstep):
    """
    Given:
      (1) A neural net
      (2) A dictionary of input values
    Does:
      Forward propagation with the given threshold function to compute binary
      output. This function should not modify the input net.
    Returns tuple containing:
      (1) The final output of the neural net
      (2) A dictionary mapping neurons to their immediate outputs
    """
    # Eleucidate net, inputs, and threshold function:
    # print('\n\nNET:\n', net)
    # print(' * threshold_fn:', threshold_fn.__name__)
    # print(' * input values:', input_values)

    # Get list of neurons, iterate through each, propagate values:
    neurons = net.topological_sort()
    neuron_outputs = {}
    for n in neurons:
        # Calculate weighted sum of inputs for each neuron:
        wsum = 0
        inputs = net.get_incoming_neighbors(n)
        for i in inputs:
            val = node_value(i, input_values, neuron_outputs)
            weight = net.get_wire(i, n).get_weight()
            wsum += val*weight
        neuron_outputs[n] = threshold_fn(wsum)

    out = node_value(net.get_output_neuron(), input_values, neuron_outputs)
    # print(' FORWARD PROP:', (out, neuron_outputs))
    return(out, neuron_outputs)


# Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """
    Given:
      (1) An unknown function of three variables
      (2) A list of three values representing current function inputs
    Does:
      Increments each variable by +/- step_size or 0, with the goal of
      maximizing the function output. After trying all possible variable
      assignments...
    Returns a tuple containing:
      (1) the maximum function output found, and
      (2) the list of inputs that yielded the highest function output.
    """
    # Eleucidate func, inputs, and step_size:
    # print('func:', func.__name__)
    # print('inputs:', inputs)
    # print('step_size:', step_size)

    max_result = -INF
    for x in [inputs[0]-step_size, inputs[0], inputs[0]+step_size]:
        for y in [inputs[1]-step_size, inputs[1], inputs[1]+step_size]:
            for z in [inputs[2]-step_size, inputs[2], inputs[2]+step_size]:
                result = func(x, y, z)
                if (result > max_result):
                    max_result = result
                    max_inputs = [x, y, z]
    # print(' ASCENT STEP:', (max_result, max_inputs))
    return((max_result, max_inputs))


def get_back_prop_dependencies(net, wire):
    """
    Given:
      (1) A wire in a neural network
    Returns:
      (1) A set of inputs, neurons, and wires whose outputs/values
          are required to update this wire's weight.
    """
    # Eleucidate net, inputs, and threshold function:
    # print('\n\nNET:\n', net)
    # print(' * input wire:', wire)

    # Initialize queue:
    dependencies = []
    queue = [wire]

    # Breadth-first forward walk through net:
    while (queue):
        w = queue.pop(0)
        nodeA = w.startNode
        nodeB = w.endNode
        dependencies.extend([nodeA, nodeB, w])
        forwardWires = net.get_wires(nodeB)
        queue.extend(forwardWires)

    # print(' * DEPENDENCIES:', set(dependencies))
    return(set(dependencies))


def calculate_deltas(net, desired_output, neuron_outputs):
    """
    Given:
      (1) A neural net
      (2) A dictionary of neuron outputs from forward-propagation
    Does:
      Computes the update coefficient (delta_B) for each neuron in the net.
      Uses the sigmoid function to compute neuron output.
    Returns:
      (1) A dictionary mapping neuron names to update coefficient (the
          delta_B values).
    """

    # Eleucidate net, desired_output, and neuron_outputs:
    # print('\n\nNET:\n', net)
    # print(' * desired_output:', desired_output)
    # print(' * neuron_outputs:', neuron_outputs)

    neurons = net.topological_sort()

    # Iterate backward through nodes, starting with the output node:
    for n in reversed(neurons):
        outB = neuron_outputs[n]

        if (net.is_output_neuron(n)):
            deltas = {n: (outB*(1-outB) * (desired_output-outB))}
        else:
            forwardNodes = net.get_outgoing_neighbors(n)
            delta_sum = 0
            for forwardNode in forwardNodes:
                weight = net.get_wire(n, forwardNode).get_weight()
                delta_sum += weight * deltas[forwardNode]
            deltas[n] = outB*(1-outB) * delta_sum

    # print(' * deltaB values:', deltas)
    return(deltas)


def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """
    Does:
      Performs a single step of back-propagation.  Computes delta_B values
      and weight updates for entire neural net, then updates all weights.
      Uses the sigmoid function to compute neuron output.
    Returns:
      (1) The modified neural net, with the updated weights.
    """

    # print('\n\nNET:\n', net)
    # print(' * neuron_outputs:', neuron_outputs)

    wires = net.get_wires()
    deltas = calculate_deltas(net, desired_output, neuron_outputs)

    for wire in wires:
        nodeA = wire.startNode
        nodeB = wire.endNode
        outA = node_value(nodeA, input_values, neuron_outputs)
        deltaB = deltas[nodeB]
        old_weight = wire.get_weight()
        new_weight = old_weight + r*outA*deltaB
        wire.set_weight(new_weight)

    return(net)


def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """
    Does:
      Updates weights until accuracy surpasses minimum_accuracy. Uses the
      sigmoid function to compute neuron output.
    Returns a tuple containing:
      (1) the modified neural net, with trained weights
      (2) the number of iterations (that is, the number of weight updates)
    """

    print('\n\nNET:\n', net)

    output, neuron_outputs = forward_prop(net, input_values, sigmoid)
    current_accuracy = accuracy(desired_output, output)
    iterations = 0

    while (current_accuracy < minimum_accuracy):
        update_weights(net, input_values, desired_output, neuron_outputs, r)
        iterations += 1
        output, neuron_outputs = forward_prop(net, input_values, sigmoid)
        current_accuracy = accuracy(desired_output, output)
        print(' accuracy =', current_accuracy)

    print(' * TOTAL ITERATIONS:', iterations)
    return((net, iterations))


# Part 5: Training a Neural Net #############################################
ANSWER_1 = 18
ANSWER_2 = 19
ANSWER_3 = 8
ANSWER_4 = 284
ANSWER_5 = 50

ANSWER_6 = 1
ANSWER_7 = 'checkerboard'
ANSWER_8 = ['small', 'medium', 'large']
ANSWER_9 = 'B'

ANSWER_10 = 'D'
ANSWER_11 = ['A', 'C']
ANSWER_12 = ['A', 'E']


# SURVEY ####################################################################

NAME = 'Blake Cole'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 13
WHAT_I_FOUND_INTERESTING = 'The whole tamale.'
WHAT_I_FOUND_BORING = 'In my opinion, the get_back_prop_dependencies() function took too much time to implement, relative to the value of conceptual underpinnings it conveyed.  I felt like the same concepts could have been learned simply by observing the solution to "2015 Quiz 3, Problem 1, Part B".  '
SUGGESTIONS = 'I was a bit dissapointed that the "training" component of the lab was all swept into the training.py black-ish box.  Albeit, it was really cool to visualize the training progress; however, I couldnt help but feel like Id put in the hard work, and stopped just shy of the goaline of true understanding.  Perhaps it would be possible to offer a bonus lab for people interested in how training on multiple data works in greater detail, or cover training in a subsequent lab.'
