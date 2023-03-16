import numpy as np
import scipy as sci
import itertools

import scipy.sparse._coo


def matrix_solver(input_matrix_csr_format: np.array, b_array: np.array) -> np.array:
    x = sci.sparse.linalg.spsolve(input_matrix_csr_format, b_array)
    return x


# given text file convert it into integer dictionary of gates and pads along with additional information
def text_to_int(text_lines: list) -> (int, int, int, dict, dict):
    for i in range(len(text_lines)):
        current_line_split = list(map(int, text_lines[i].split()))
        if i == 0:  # first line containing number_of_gates and number_of_nets
            number_of_gates = current_line_split[0]
            number_of_nets = current_line_split[1]
            gate_dict = {}
            pad_dict = {}

        elif i > 0 and i <= number_of_gates:
            gate_info = []
            for ele in current_line_split:
                gate_info.append(ele)
            current_gate_num = gate_info[0]
            number_of_connections = gate_info[1]
            nets_connected = gate_info[2:]
            gate_dict[current_gate_num] = nets_connected

        elif i == number_of_gates + 1:
            number_of_pads = current_line_split[0]
        else:
            pad_info = []
            for ele in current_line_split:
                pad_info.append(ele)
            current_pad_num = pad_info[0]
            connected_net_num = pad_info[1]
            x_y_coordinate_tuple = tuple(pad_info[-2:])
            trait_dict = {'connected_net_num': connected_net_num, 'x_y_coordinate': x_y_coordinate_tuple}
            pad_dict[current_pad_num] = trait_dict

    return number_of_gates, number_of_nets, number_of_pads, gate_dict, pad_dict


# generate dictionary with key being net and value being gates, (net_num): [gate_val1, gate_val2, ...]
def net_gate_dict_generator(number_of_nets, gate_dict: dict) -> dict:
    net_gate_dict = {}
    # initialize empty dictionary with all of the nets as keys
    # value for dictionary will be the gates connected to the net itself
    for i in range(1, number_of_nets + 1):
        net_gate_dict[i] = []
    for gate in gate_dict.keys():
        adjacent_nets_to_gate = gate_dict[gate]
        for net in adjacent_nets_to_gate:
            net_gate_dict[net].append(gate)
    return net_gate_dict


# generate dictionary that maps pad to gate as well as encoding the coordinate of the pad itself
def pad_gate_dict_generator(pad_dict: dict, net_gate_dict: dict) -> dict:
    pad_gate_dict = {}
    for pad in pad_dict.keys():
        connected_net_num = pad_dict[pad]['connected_net_num']
        x_y_coordinate_tuple = pad_dict[pad]['x_y_coordinate']
        connected_gates = net_gate_dict[connected_net_num]
        trait_dict = {'connected_gates': connected_gates, 'x_y_coordinate': x_y_coordinate_tuple}
        pad_gate_dict[pad] = trait_dict
    return pad_gate_dict


# given a net_gate_dict which shows gates connected to the same net generate a set that contains all gate adjacent pairs
def adjacent_gate_pair_generator(net_gate_dict: dict) -> set:
    gate_adjacent_pair_set = set()
    for adjacent_gate_list in net_gate_dict.values():
        if len(adjacent_gate_list) > 1:
            adjacent_tuple_list = list(itertools.combinations(adjacent_gate_list, 2))
            gate_adjacent_pair_set.update(adjacent_tuple_list)
    return gate_adjacent_pair_set


# given adjacent_gate_pair generate gate_adjacency_dict
def gate_adjacency_dict_generator(gate_adjacent_pair_set: set) -> dict:
    gate_adjacency_dict = {}
    for (gate_0, gate_1) in gate_adjacent_pair_set:
        if gate_0 in gate_adjacency_dict:
            gate_adjacency_dict[gate_0].append(gate_1)
        else:
            gate_adjacency_dict[gate_0] = [gate_1]
        if gate_1 in gate_adjacency_dict:
            gate_adjacency_dict[gate_1].append(gate_0)
        else:
            gate_adjacency_dict[gate_1] = [gate_0]
    return gate_adjacency_dict


# convert gate_adjacent_pair_set into v r c array matrix format with all values decremented by 1, ex: 1 -> 0, 18->17...
def vrc_array_generator_from_tuple_list(gate_adjacent_pair_set: set) -> (np.array, np.array, np.array):
    v_list = []
    row_list = []
    col_list = []

    for u, v in gate_adjacent_pair_set:
        v_list.append(1)
        row_list.append(u - 1)
        col_list.append(v - 1)
        # additional entry as the matrix is symmetric
        v_list.append(1)
        row_list.append(v - 1)
        col_list.append(u - 1)

    V_array = np.array(v_list, dtype=int)
    R_array = np.array(row_list, dtype=int)
    C_array = np.array(col_list, dtype=int)

    return V_array, R_array, C_array


# convert gate_adjacency_dict into v r c array matrix format index based on gate_vector_map
# gate_vector_map: a 1D numpy array where gate is mapped to the index of matrix/vector
#   - format: [gate_num_0, gate_num_1, ...]
#       - gate_vector_map[(index used in vrc matrix and b vector)] = (gate_num)
def vrc_array_generator_from_gate_adjacency_dict(gate_adjacency_dict: dict, gate_vector_map: np.array) -> (
np.array, np.array, np.array):
    v_list = []
    row_list = []
    col_list = []
    gate_index_dict = {}
    for i, gate_num in enumerate(gate_vector_map):
        gate_index_dict[gate_num] = i
    for k in gate_adjacency_dict.keys():
        val_list = gate_adjacency_dict[k]
        for v in val_list:
            v_list.append(1)
            row_list.append(gate_index_dict[k])
            col_list.append(gate_index_dict[v])


    V_array = np.array(v_list, dtype=int)
    R_array = np.array(row_list, dtype=int)
    C_array = np.array(col_list, dtype=int)

    return V_array, R_array, C_array


# take a simple C matrix in coo format and incorporate pad details to generate a V matrix
def C_to_V_matrix_converter(A_matrix_csr_format: scipy.sparse._csr.csr_matrix,
                            pad_weight_list: np.array) -> scipy.sparse._csr.csr_matrix:
    row_sum_list = A_matrix_csr_format.sum(axis=1)
    col_sum_list = A_matrix_csr_format.sum(axis=0)

    row_sum = np.array(row_sum_list).flatten()
    col_sum = np.array(col_sum_list).flatten()
    diagonal_val = row_sum + (pad_weight_list)
    # print(diagonal_val, '\n')

    V_matrix_csr = -A_matrix_csr_format  # change sign for non-diagonal values
    V_matrix_lil = V_matrix_csr.tolil()
    V_matrix_lil.setdiag(diagonal_val, k=0)  # add diagonal values to get V matrix
    # V_matrix_csr.setdiag(sci.sparse.diags(diagonal_val)) # add diagonal values to get V matrix
    return V_matrix_lil.tocsr()


# given net_gate_dict and pad_dict generate pad_weight_array where the total pad weight of each gate is computed
def pad_weight_array_calculator(net_gate_dict: dict, pad_dict: dict) -> np.array:
    print(net_gate_dict.values())

    number_of_gates = max([x for sub_list in net_gate_dict.values() for x in sub_list])
    print(number_of_gates)
    pad_weight_list = [0] * number_of_gates
    pad_connected_net_list = []
    for key in pad_dict.keys():
        pad_connected_net_list.append(pad_dict[key]['connected_net_num'])

    for pad_net in pad_connected_net_list:
        pad_connected_gate = net_gate_dict[pad_net]
        for gate in pad_connected_gate:
            pad_weight_list[gate - 1] = pad_weight_list[gate - 1] + 1
    # print(pad_weight_list, '\n')
    return np.array(pad_weight_list)

# given pad_gate_dict generate pad_weight_array where the total pad weight of each gate is computed
# INPUT:
# pad_gate_dict: (pad_num): {'connected_gates': [(gate_1), (gate_2) ...], 'x_y_coordinate': (pad_x, pad_y)}, ...
# gate_vector_map: maps the index to the gate number through an array
def pad_gate_dict_pad_weight_array_calculator(pad_gate_dict: dict, gate_vector_map: np.array) -> np.array:
    # initialize empty pad_weight_array
    pad_weight_array = np.zeros(len(gate_vector_map), dtype=np.int32)
    for pad in pad_gate_dict.keys():
        pad_connected_gate_list = pad_gate_dict[pad]['connected_gates']
        for gate in pad_connected_gate_list:
            index = gate_vector_map == gate
            pad_weight_array[index] = pad_weight_array[index] + 1
    return pad_weight_array

# generate bx and by vectors from pad_dict and net_gate_dict bxi = wi * xi(pad coordinate)
# returns (bx, by) in tuple form stored in np.array
def b_vector_generator(net_gate_dict: dict, pad_dict: dict) -> (np.array, np.array):
    number_of_gates = max([x for sub_list in net_gate_dict.values() for x in sub_list])
    bx_array = np.zeros(number_of_gates)
    by_array = np.zeros(number_of_gates)
    connection_weight = 1  # default connection weight is always 1 in this assignment
    for current_pin in pad_dict.keys():
        connected_net_num = pad_dict[current_pin]['connected_net_num']
        pin_x_coordinate, pin_y_coordinate = pad_dict[current_pin]['x_y_coordinate']
        current_net_connected_gate_list = net_gate_dict[connected_net_num]
        for current_gate in current_net_connected_gate_list:
            bx_array[current_gate - 1] = bx_array[current_gate - 1] + pin_x_coordinate * connection_weight
            by_array[current_gate - 1] = by_array[current_gate - 1] + pin_y_coordinate * connection_weight
    return bx_array, by_array


# generate bx and by vectors from pad_dict and net_gate_dict bxi = wi * xi(pad coordinate)
# returns (bx, by) in tuple form stored in np.array
# uses standardized input data format compared to b_vector_generator() function
# INPUT:
# pad_gate_dict: (pad_num): {'connected_gates': [(gate_1), (gate_2) ...], 'x_y_coordinate': (pad_x, pad_y)}, ...
# gate_vector_map: maps the index to the gate number through an array
def pad_gate_dict_b_vector_generator(pad_gate_dict: dict, gate_vector_map) -> (np.array, np.array):
    number_of_gates = len(gate_vector_map)
    bx_array = np.zeros(number_of_gates, dtype=np.float32)
    by_array = np.zeros(number_of_gates, dtype=np.float32)
    connection_weight = 1 # default connection weight is always 1 in this assignment
    for pad in pad_gate_dict.keys():
        pad_x_coordinate, pad_y_coordinate = pad_gate_dict[pad]['x_y_coordinate']
        for gate in pad_gate_dict[pad]['connected_gates']:
            index = gate_vector_map == gate
            bx_array[index] = bx_array[index] + pad_x_coordinate * connection_weight
            by_array[index] = by_array[index] + pad_y_coordinate * connection_weight

    return bx_array, by_array


# take input file and reads gate, net, pad data, then encodes this information to various parameters
# number_of_gates: number of gates in input file
# number_of_nets: number of nets in input file
# gate_dict: original information from file, format: (gate_num): [net_1, net_2, ... ] (list of connected nets)
# pad_dict: original information from file,
#   - format: two layered dictionary
#       (pad_num): {
#           'connected_net_num': (net num that is connected to pad_num),
#           'x_y_coordinate': (x_coordinate_of_current_pad_num, y_coordinate_of_current_pad_num)
#       } ...
# pad_gate_dict: processed information, maps pad to connected gates as well coordinate of pad itself,
#   - format: two layered dictionary
#       (pad_num): {
#           'connected_gate_num': (gate num that is connected to pad_num),
#           'x_y_coordinate': (x_coordinate_of_current_pad_num, y_coordinate_of_current_pad_num)
#       } ...
# net_gate_dict: processed information, maps net to connected gates,
#   - format: (net_num): [gate_1, gate_2, ...] (list of connected gates)
# gate_adjacent_pair_set: set of all gate connections,
#   - each connection consists of (gate_1, gate_2) when gate_1 and gate_2 shares a net connection
#   - gate_1 < gate_2 rule is always followed, set of tuples to prevent any duplicates
# gate_adjacency_dict: adjacency list of gates in dictionary format
#   - format:
#       (gate_num): [adjacent_gate_1, adjacent_gate 2, ... ]
def input_file_parser(input_file_path: str):
    with open(input_file_path, 'r') as f:
        lines = f.readlines()
    # first line to retrieve number of gates and number of nets
    # number_of_gates = (lines[0].split())[0]
    # number_of_nets = (lines[0].split())[1]
    number_of_gates, number_of_nets, number_of_pads, gate_dict, pad_dict = text_to_int(lines)
    net_gate_dict = net_gate_dict_generator(number_of_nets, gate_dict)
    pad_gate_dict = pad_gate_dict_generator(pad_dict, net_gate_dict)
    print(net_gate_dict)
    gate_adjacent_pair_set = adjacent_gate_pair_generator(net_gate_dict)
    print(gate_adjacent_pair_set)
    print(len(gate_adjacent_pair_set))
    gate_adjacency_dict = gate_adjacency_dict_generator(gate_adjacent_pair_set)
    return number_of_gates, number_of_nets, number_of_pads, gate_dict, pad_dict, pad_gate_dict, net_gate_dict, gate_adjacent_pair_set, gate_adjacency_dict


# given x y coordinate array of gates generate a text file of output
def output_file_generator(gate_x_array, gate_y_array, file_name=r'default_output_placement_file.txt'):
    with open(file_name, 'w') as f:
        total_gate_num = len(gate_x_array)
        lines = []
        for current_gate_num in range(1, total_gate_num + 1):
            line_str = str(current_gate_num) + ' ' + str(round(gate_x_array[current_gate_num - 1], 8)) + ' ' + \
                       str(round(gate_y_array[current_gate_num - 1], 8))
            lines.append(line_str)
        for line in lines:
            f.write(line + '\n')


# generate a single sort key based on x and y vectors
# x_vector/y_vector : contains x y coordinates of each gate
# gate_vector_map : contains the mapping between the gate number and the index number of x_vector / y_vector
# x_vector /y_vector 1D array with (index) -> (coordinate)
# gate_vector_map : 1D array with (index) -> (gate_num)
# config: decides the partition and sort direction
#   - True (partition by left/right 100000 * x + y)
#   - False (partition by top/bottom 100000 * y + x)
# output: (part_0_x_vector, part_0_y_vector, part_0_gate_vector_map, part_1_x_vector, part_1_y_vector, part_1_gate_vector_map)
def sort_vector_generator(x_vector: np.array, y_vector: np.array, gate_vector_map: np.array, config=True,
                          multiplier=10000.0) -> (np.array, np.array, np.array, np.array, np.array, np.array):
    if config == True:  # right/left partition
        key_vector = multiplier * x_vector + y_vector
    else:  # top/bottom partition
        key_vector = multiplier * y_vector + x_vector

    # get indices that sort key_vector in ascending order
    idx = np.argsort(key_vector)
    # sort key_vector by idx
    sorted_key_vector = key_vector[idx]
    # sort gate_vector_map based on idx
    sorted_gate_vector_map = gate_vector_map[idx]
    # sort x and y vector based on idx
    sorted_x_vector = x_vector[idx]
    sorted_y_vector = y_vector[idx]

    split_index = len(idx) // 2
    part_0_x_vector = sorted_x_vector[:split_index]
    part_0_y_vector = sorted_y_vector[:split_index]
    part_0_gate_vector_map = sorted_gate_vector_map[:split_index]

    part_1_x_vector = sorted_x_vector[split_index:]
    part_1_y_vector = sorted_y_vector[split_index:]
    part_1_gate_vector_map = sorted_gate_vector_map[split_index:]
    return (
    part_0_x_vector, part_0_y_vector, part_0_gate_vector_map, part_1_x_vector, part_1_y_vector, part_1_gate_vector_map)


# generate a gate_adjacency_dict for each of the two partitions
# gate_adjacency_dict: encodes the adjacency relations before the partition
# part_0_gate_vector_map: contains the gate number for the partition 0
# part_1_gate_vector_map: contains the gate number for the partition 1
# Output:
# part_0_gate_adjacency_dict: contains the adjacency relations for the partition 0
# part_1_gate_adjacency_dict: contains the adjacency relations for the partition 1
# removes any references to gates in different partition while preserving relation for gates in same partition
def partition_adjacency_dict_generator(gate_adjacency_dict:dict, part_0_gate_vector_map: np.array, part_1_gate_vector_map: np.array) -> (dict, dict):
    part_0_gate_adjacency_dict = {}
    part_1_gate_adjacency_dict = {}
    for gate in part_0_gate_vector_map:
        part_0_gate_adjacency_dict[gate] = [x for x in gate_adjacency_dict[gate] if x in part_0_gate_vector_map]
    for gate in part_1_gate_vector_map:
        part_1_gate_adjacency_dict[gate] = [x for x in gate_adjacency_dict[gate] if x in part_1_gate_vector_map]
    return part_0_gate_adjacency_dict, part_1_gate_adjacency_dict


# generate pad_gate_dict for part_0 while propagating gates/pads from part_1
# INPUTS:
# gate_adjacency_dict: encodes an adjacency list consisting of gates (gate_num): [adjacent_gate1, ...]
# pad_gate_dict: stores information about gates adjacent to pads and coordinate of pads
# part_0_x_vector: contains the x coordinates of gates in partition 0
# part_0_y_vector: contains the y coordinates of gates in partition 0
# part_0_gate_vector_map: maps the index to gate number for partition 0
# cut_orientation: boolean value that specifies cut direction 1: Vertical(Right/Left) 0: Horizontal(Top/Bottom)
# cut_line: defines cut_line coordinates
def part_0_pad_gate_dict_generator(gate_adjacency_dict: dict, pad_gate_dict: dict, part_0_gate_vector_map: np.array,
                                   part_1_x_vector: np.array, part_1_y_vector: np.array,
                                   part_1_gate_vector_map: np.array, cut_orientation: bool, cutline: float) -> dict:
    # When cut_orientation == True(1), cutline vertical, part_0 left, part_1 right
    # When cut_orientation == False(0), cutline horizontal, part_0 top, part_1 bottom
    part_0_pad_gate_dict = {}
    max_pad_num = max(pad_gate_dict.keys())

    # 1: Left/Top-side(part_0) containment step
    # 1-1: find connected gates and pads that should be propagated to cutline
    connected_gates_in_part_1 = []
    for part_0_gate in part_0_gate_vector_map:
        connected_gates_in_part_1.append(x for x in gate_adjacency_dict[part_0_gate] if x in part_1_gate_vector_map)
    connected_pads_in_part_1 = [] # need to be propagated as it sits opposite of cutline
    connected_pads_in_part_0 = [] # preserve coordinate as it is both connected and not propagated, coordinate preserved
    for pad in pad_gate_dict:
        x_y_coordinate = pad_gate_dict[pad]['x_y_coordinate']
        # depending on the cut orientation the compare_coordinate should be either left or top of cutline
        # it is x or y coordinate of pad depending on cut orientation
        # x(0 idx) for vertical cut y(1 idx) for horizontal cut
        compare_coordinate = x_y_coordinate[cut_orientation]
        for pad_connected_gate in pad_gate_dict[pad]['connected_gates']:
            if pad_connected_gate in part_0_gate_vector_map:
                if compare_coordinate < cutline:
                    connected_pads_in_part_1.append(pad)
                    break # pad's status is already determined no need to check further
                else:
                    connected_pads_in_part_0.append(pad)
                    break # pad's status is already determined no need to check further
    # 1-2: Add the connected pads with no need for propagation(connected_pads_in_part_0) to part_0_pad_gate_dict
    for pad in connected_pads_in_part_0:
        # only add to gate list if that gate is in part_0
        gate_list = [x for x in pad_gate_dict[pad]['connected_gates'] if x in part_0_gate_vector_map]
        pad_trait = {'connected_gates': gate_list, 'x_y_coordinate': pad_gate_dict[pad]['x_y_coordinate']}
        part_0_pad_gate_dict[pad] = pad_trait
    # 1-3: Propagate the pads in part_1(connected_pads_in_part_1) to cutline
    for propagate_pad in connected_pads_in_part_1:
        # only add to gate list if that gate is in part_0
        gate_list = [x for x in pad_gate_dict[propagate_pad]['connected_gates'] if x in part_0_gate_vector_map]
        if cut_orientation == 1: # if cut_orientation is vertical left/right, propagate x coordinate to cutline
            x_y_coordinate = (cutline, pad_gate_dict[propagate_pad]['x_y_coordinate'][1])
        else: # if cut_orientation is horizontal top/bottom, propagate y coordinate to cutline
            x_y_coordinate = (pad_gate_dict[propagate_pad]['x_y_coordinate'][0], cutline)
        pad_trait = {'connected_gates': gate_list, 'x_y_coordinate': x_y_coordinate}
        part_0_pad_gate_dict[propagate_pad] = pad_trait
    # 1-4: Propagate the gates to cutline
    for propagate_gate in connected_gates_in_part_1:
        # only add to gate list if that gate is in part_0
        gate_list = [x for x in gate_adjacency_dict[propagate_gate] if x in part_0_gate_vector_map]
        if cut_orientation == 1: # if cut_orientation is vertical left/right, propagate x coordinate to cutline
            x_y_coordinate = (cutline, part_1_y_vector[int(np.where(part_1_gate_vector_map == propagate_gate))])
        else: # if cut_orientation is horizontal top/bottom, propagate y coordinate to cutline
            x_y_coordinate = (part_1_x_vector[int(np.where(part_1_gate_vector_map == propagate_gate))], cutline)
        pad_trait = {'connected_gates': gate_list, 'x_y_coordinate': x_y_coordinate}
        part_0_pad_gate_dict[propagate_gate] = pad_trait

    return part_0_pad_gate_dict

# generate gate adjacency dictionary and pad gate dictionary for the two partitions while also carrying out porpagation of gates and pads
# INPUTS:
# gate_adjacency_dict: encodes an adjacency list consisting of gates (gate_num): [adjacent_gate1, ...]
# pad_gate_dict: stores information about gates adjacent to pads and coordinate of pads
# part_0_x_vector: contains the x coordinates of gates in partition 0
# part_0_y_vector: contains the y coordinates of gates in partition 0
# part_0_gate_vector_map: maps the index to gate number for partition 0
# vice versa for part_1_x/y_vector and gate_vector_map
# cut_orientation: boolean value that specifies cut direction 1: Vertical(Right/Left) 0: Horizontal(Top/Bottom)
# cut_line: defines cut_line coordinates
def partition_generator(gate_adjacency_dict: dict, pad_gate_dict: dict, part_0_x_vector: np.array,
                                   part_0_y_vector: np.array, part_0_gate_vector_map: np.array,
                                   part_1_x_vector: np.array, part_1_y_vector: np.array,
                                   part_1_gate_vector_map: np.array, cut_orientation: bool, cutline: float) -> (
dict, dict, dict, dict):
    part_0_gate_adjacency_dict, part_1_gate_adjacency_dict = partition_adjacency_dict_generator(gate_adjacency_dict,part_0_gate_vector_map, part_1_gate_vector_map)

    part_0_pad_gate_dict = part_0_pad_gate_dict_generator()
    return part_0_gate_adjacency_dict, part_0_pad_gate_dict, part_1_gate_adjacency_dict, part_1_pad_gate_dict


# uses functions to implement the 3 Quadratic Placement
def core_3qp_placer(input_file_path: str) -> np.array:
    input_file_path = r'..\benchmarks\3QP\toy1'
    number_of_gates, number_of_nets, number_of_pads, gate_dict, pad_dict, pad_gate_dict, net_gate_dict, \
        gate_adjacent_pair_set, gate_adjacency_dict = input_file_parser(r'..\benchmarks\3QP\toy1')

    x_max = 100.0
    y_max = 100.0
    # QP1: conduct initial #1 placement
    V, R, C = vrc_array_generator_from_tuple_list(gate_adjacent_pair_set)

    # C_matrix_coo_format = sci.sparse.coo_matrix((V, (R ,C)), shape=(number_of_gates, number_of_gates))
    # C_matrix_csr_format = C_matrix_coo_format.tocsr()
    C_matrix_csr_format = sci.sparse.csr_matrix((V, (R, C)), shape=(number_of_gates, number_of_gates))

    pad_weight_array = pad_weight_array_calculator(net_gate_dict, pad_dict)
    V_matrix_csr_format = C_to_V_matrix_converter(C_matrix_csr_format, pad_weight_array)
    bx_array, by_array = b_vector_generator(net_gate_dict, pad_dict)
    x = sci.sparse.linalg.spsolve(V_matrix_csr_format, bx_array)
    y = sci.sparse.linalg.spsolve(V_matrix_csr_format, by_array)

    # find line number where pad information starts

    # return A_matrix_csr_format, b_x_array, b_y_array
    # output_file_generator(x, y)

    # QR2: conduct placement of left partition of QR1
    # 1. assign partition of QP1, left and right (x vectors)
    initial_gate_vector_map = np.array([(x + 1) for x in range(number_of_gates)])
    part_0_x_vector, part_0_x_vector, part_0_gate_vector_map, part_1_x_vector, part_1_y_vector, part_1_gate_vector_map = sort_vector_generator(x, y, initial_gate_vector_map)

    part_0_gate_num = len(part_0_gate_vector_map)
    part_1_gate_num = len(part_1_gate_vector_map)

    # conducts propagation for the gates and the pads and the related data structures
    # TODO LIST: Implement partition propagation function
    qr1_x_cutline = x_max / 2
    cut_orientation = True # True: vertical cut line (left/right), False: horizontal cut line (top/bottom)

    # calculate gate connectivity dictionary for both partitions
    part_0_gate_adjacency_dict, part_1_gate_adjacency_dict = partition_adjacency_dict_generator(gate_adjacency_dict,part_0_gate_vector_map, part_1_gate_vector_map)
    # propagate pads/gates from part_1 to setup pads for part_0
    part_0_pad_gate_dict = part_0_pad_gate_dict_generator(gate_adjacency_dict, pad_gate_dict, part_0_gate_vector_map, part_1_x_vector, part_1_y_vector, part_1_gate_vector_map, cut_orientation, qr1_x_cutline)
    # perform placement on part_0
    part_0_V, part_0_R, part_0_C = vrc_array_generator_from_gate_adjacency_dict(part_0_gate_adjacency_dict, part_0_gate_vector_map)

    part_0_C_matrix_csr = sci.sparse.csr_matrix((part_0_V, (part_0_R, part_0_C)), shape=(part_0_gate_num, part_0_gate_num))

    part_0_pad_weight_array = pad_gate_dict_pad_weight_array_calculator(part_0_pad_gate_dict, part_0_gate_vector_map)
    part_0_V_matrix_csr_format = C_to_V_matrix_converter(part_0_C_matrix_csr, part_0_pad_weight_array)
    part_0_bx_array, part_0_by_array = pad_gate_dict_b_vector_generator(part_0_pad_gate_dict, part_0_gate_vector_map)
    part_0_x = sci.sparse.linalg.spsolve(part_0_V_matrix_csr_format, part_0_bx_array)
    part_0_y = sci.sparse.linalg.spsolve(part_0_V_matrix_csr_format, part_0_by_array)

    # QR3: conduct right partition of QR1
    # propagate already placed pads/gates from part_0 to setup pads for part_1
    part_1_pad_gate_dict = part_1_pad_gate_dict_generator()

    # perform placement on part_1
    part_1_V, part_1_R, part_1_C = vrc_array_generator_from_gate_adjacency_dict(part_1_gate_adjacency_dict, part_1_gate_vector_map)
    part_1_C_matrix_csr = sci.sparse.csr_matrix((part_1_V, (part_1_R, part_1_C)), shape=(part_1_gate_num, part_1_gate_num))


    part_1_pad_weight_array = pad_gate_dict_pad_weight_array_calculator(part_1_pad_gate_dict)
    part_1_V_matrix_csr_format = C_to_V_matrix_converter(part_1_C_matrix_csr, part_1_pad_weight_array)
    part_1_bx_array, part_1_by_array = pad_gate_dict_b_vector_generator(part_1_pad_gate_dict, part_1_gate_vector_map)
    part_1_x = sci.sparse.linalg.spsolve(part_1_V_matrix_csr_format, part_1_bx_array)
    part_1_y = sci.sparse.linalg.spsolve(part_1_V_matrix_csr_format, part_1_by_array)

    merged_x, merged_y = merge_partition(part_0_x, part_0_y, part_1_x, part_1_y, part_0_gate_vector_map, part_1_gate_vector_map)



    # part_0_gate_adjacency_dict, part_0_pad_gate_dict, part_1_gate_adjacency_dict, part_1_pad_gate_dict = partition_generator(
    #     gate_adjacency_dict, pad_gate_dict, part_0_x_vector, part_0_y_vector, part_0_gate_vector_map, part_1_x_vector,
    #     part_1_y_vector, part_1_gate_vector_map, cut_orientation, qr1_x_cutline)

    # propagate connected gates and pads to cutline at x = 50

    # QP1_part_0_V, QP1_part_0_R, QP1_part_0_C = vrc_array_generator_from_gate_adjacency_dict(gate)


def main():
    number_of_gates, number_of_nets, number_of_pads, gate_dict, pad_dict, pad_gate_dict, net_gate_dict, gate_adjacent_pair_set, gate_adjacency_dict = \
        input_file_parser(r'..\benchmarks\3QP\toy1')
    V, R, C = vrc_array_generator_from_tuple_list(gate_adjacent_pair_set)
    initial_gate_vector_map = np.array([(x + 1) for x in range(number_of_gates)])
    V_alt, R_alt, C_alt = vrc_array_generator_from_gate_adjacency_dict(gate_adjacency_dict, initial_gate_vector_map)
    # C_matrix_coo_format = sci.sparse.coo_matrix((V, (R ,C)), shape=(number_of_gates, number_of_gates))
    # C_matrix_csr_format = C_matrix_coo_format.tocsr()
    C_matrix_csr_format = sci.sparse.csr_matrix((V, (R, C)), shape=(number_of_gates, number_of_gates))
    alt_C_matrix_csr_format = sci.sparse.csr_matrix((V_alt, (R_alt, C_alt)), shape=(number_of_gates, number_of_gates))
    print('comparison between C implementations', np.array_equal(C_matrix_csr_format.todense(), alt_C_matrix_csr_format.todense()))
    print((C_matrix_csr_format.todense()))
    print(alt_C_matrix_csr_format.todense())
    print(C_matrix_csr_format.todense() == alt_C_matrix_csr_format.todense())
    pad_weight_array = pad_weight_array_calculator(net_gate_dict, pad_dict)
    dict_based_pad_weight_array = pad_gate_dict_pad_weight_array_calculator(pad_gate_dict, initial_gate_vector_map)
    print('comparison between pad_weight_array implementations', np.array_equal(pad_weight_array, dict_based_pad_weight_array))
    print(pad_weight_array)
    print(dict_based_pad_weight_array)
    V_matrix_csr_format = C_to_V_matrix_converter(C_matrix_csr_format, pad_weight_array)
    bx_array, by_array = b_vector_generator(net_gate_dict, pad_dict)
    dict_bx_array, dict_by_array = pad_gate_dict_b_vector_generator(pad_gate_dict, initial_gate_vector_map)
    print('comparison between bx/by_array implementations', np.array_equal(bx_array, dict_bx_array) and np.array_equal(by_array, dict_by_array))
    x = sci.sparse.linalg.spsolve(V_matrix_csr_format, bx_array)
    y = sci.sparse.linalg.spsolve(V_matrix_csr_format, by_array)
    print(V_matrix_csr_format.todense())
    print(bx_array)
    print(by_array)
    print(x)
    print(y)
    # find line number where pad information starts
    print(gate_dict)
    print(net_gate_dict)
    print(pad_dict)
    print(gate_adjacent_pair_set)
    print(len(gate_adjacent_pair_set))
    print('pad_gate_dict\n',pad_gate_dict)
    # return A_matrix_csr_format, b_x_array, b_y_array
    output_file_generator(x, y)
    pass


if __name__ == '__main__':
    main()
