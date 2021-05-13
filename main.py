import math
import numpy as np
import time

class make_node():
    def __init__(self, state, parent, move, leveldepth, patheuristiccost, totalPatheuristiccost, heuristic_cost):
        self.state = state
        self.parent = parent  
        self.move = move 
        self.leveldepth = leveldepth  
        self.patheuristiccost = patheuristiccost  # Heuristic Cost
        self.totalPatheuristiccost = totalPatheuristiccost  # Total heuristic cost to visit current node
        self.heuristic_cost = heuristic_cost  # h(n), heuristic cost

        # children node
        self.up = None
        self.left = None
        self.down = None
        self.right = None

    def pathsearch(self, goal_State, algorithm_input):
        start=time.time() ##to start time track
        queue = [(self, 0)]  # queue containing not visited nodes ordered by pathcost
        queue_popped = 0  # count of nodes popped from queue
        queue_maxLength = 1  # max nodes in queue for space complexity measurement

        queueDepth = [(0, 0)]  #queue for node depth
        total_pathCost_queue = [0]  #queue for total path cost
        visited_states = set([])  # visited states memorization
        while queue:     
            queue = sorted(queue, key=lambda x: x[1]) # sorting queue based on path cost, in ascending order
            queueDepth = sorted(queueDepth, key=lambda x: x[1])
            total_pathCost_queue = sorted(total_pathCost_queue, key=lambda x: x) 
            ## change the mximum length of queue
            if len(queue) > queue_maxLength:
                queue_maxLength = len(queue)

            currentNode = queue.pop(0)[0]  # pop first node in queue
            queue_popped += 1
            depth_present = queueDepth.pop(0)[0]  # pop depth for current node
            current_patheuristiccost = total_pathCost_queue.pop(0)  # pop path cost for visiting current node
            visited_states.add(tuple(currentNode.state))  # avoid repeated state, which is represented as a tuple
            # when the goal state is found, print path, nodes, depth etc.
            if np.array_equal(currentNode.state, goal_State):
                currentNode.displaypath()
                print("\n \n \n -------------Reaching Goal State------------- \n \n \n")
                print('\n Total Nodes Expanded are', str(queue_popped))
                print('\n Maximum count of nodes in the queue at any time was ', str(queue_maxLength))
                print('\n Depth of Goal Node is ', str(depth_present + 1))
                end = time.time()
                executiontime = end - start
                return True

            else:
                # goal state is not found, move blank space to right
                if currentNode.traverseright():
                    fresh_state, right_tile = currentNode.traverseright()
                    # check if the resulting node is already visited
                    if tuple(fresh_state) not in visited_states:
                        # create a new child node
                        leveldepth = depth_present + 1
                        total_pathCost = current_patheuristiccost
                        if algorithm_input == "1":
                            heuristic = 0
                        elif algorithm_input == "2":
                             heuristic = self.heuristic_hamming(fresh_state, goal_State)
                        else:
                             heuristic = self.heuristic_manhattan(fresh_state, goal_State)
                        currentNode.move_right = make_node(state=fresh_state, parent=currentNode, move='right',
                                                           leveldepth=leveldepth,
                                                           patheuristiccost=leveldepth + 1, totalPatheuristiccost=total_pathCost,
                                                           heuristic_cost=heuristic)
                        queue.append((currentNode.move_right, current_patheuristiccost))
                        queueDepth.append((depth_present + 1, current_patheuristiccost))
                        total_pathCost_queue.append(total_pathCost)

                # goal state is not found, move blank space to left
                if currentNode.traverseleft():
                    fresh_state, left_tile = currentNode.traverseleft()
                    # check if the resulting node is already visited
                    if tuple(fresh_state) not in visited_states:
                        leveldepth = depth_present + 1
                        total_pathCost = current_patheuristiccost
                        if algorithm_input == "1":
                            heuristic = 0
                        elif algorithm_input == "2":
                            heuristic = self.heuristic_hamming(fresh_state, goal_State)
                        else:
                            heuristic = self.heuristic_manhattan(fresh_state, goal_State)
                        # create a new child node
                        currentNode.move_left = make_node(state=fresh_state, parent=currentNode, move='left',
                                                          leveldepth=leveldepth,
                                                          patheuristiccost=leveldepth + 1, totalPatheuristiccost=total_pathCost,
                                                          heuristic_cost=heuristic)
                        queue.append((currentNode.move_left, current_patheuristiccost))
                        queueDepth.append((depth_present + 1, current_patheuristiccost))
                        total_pathCost_queue.append(current_patheuristiccost)

                # goal state is not found, move blank space to down
                if currentNode.traversedown():
                    fresh_state, down_tile = currentNode.traversedown()
                    # check if the resulting node is already visited
                    if tuple(fresh_state) not in visited_states:
                        leveldepth = depth_present + 1
                        total_pathCost = current_patheuristiccost  # + down_tile
                        if algorithm_input == "1":
                            heuristic = 0
                        elif algorithm_input == "2":
                            heuristic = self.heuristic_hamming(fresh_state, goal_State)
                        else:
                            heuristic = self.heuristic_manhattan(fresh_state, goal_State)
                        # create a new child node
                        currentNode.move_down = make_node(state=fresh_state, parent=currentNode, move='down',
                                                          leveldepth=leveldepth,
                                                          patheuristiccost=leveldepth + 1, totalPatheuristiccost=total_pathCost,
                                                          heuristic_cost=heuristic)
                        queue.append((currentNode.move_down, current_patheuristiccost))
                        queueDepth.append((depth_present + 1, current_patheuristiccost))
                        total_pathCost_queue.append(current_patheuristiccost)

                # goal state is not found, move blank space to up
                if currentNode.traverseup():
                    fresh_state, up_tile = currentNode.traverseup()
                    # check if the resulting node is already visited
                    if tuple(fresh_state) not in visited_states:
                        leveldepth = depth_present + 1
                        total_pathCost = current_patheuristiccost
                        if algorithm_input == "1":
                            heuristic = 0
                        elif algorithm_input == "2":
                            heuristic = self.heuristic_hamming(fresh_state, goal_State)
                        else:
                            heuristic = self.heuristic_manhattan(fresh_state, goal_State)
                        # create a new child node
                        currentNode.move_up = make_node(state=fresh_state, parent=currentNode, move='up',
                                                        leveldepth=leveldepth,
                                                        patheuristiccost=leveldepth, totalPatheuristiccost=total_pathCost,
                                                        heuristic_cost=heuristic)
                        queue.append((currentNode.move_up, current_patheuristiccost))
                        queueDepth.append((depth_present + 1, current_patheuristiccost))
                        total_pathCost_queue.append(current_patheuristiccost)

        # return h(c) count of misplaced tiles
    def heuristic_hamming(self, fresh_state, goal_state):
        heuristiccost = np.sum(
            fresh_state != goal_state) - 1  # minus 1 to exclude the empty tile # check how many tiles are different
        if heuristiccost > 0:
            return heuristiccost
        else:
            return 0 
    def heuristic_manhattan(self, fresh_state, goal_state):   # return h(c): sum of Manhattan distance for reaching the goal state
        distance = 0
        number_of_pieces = len(fresh_state)
        c = int(math.sqrt(number_of_pieces))
        for i in range(1, number_of_pieces + 1):
            first = np.where(fresh_state == 0)
            second = np.where(goal_state == 0)
            first_x = first[0][0] % c
            first_y = first[0][0] / c
            second_x = second[0][0] % c
            second_y = second[0][0] / c
            distance += abs(first_x - second_x) + abs(first_y - second_y)
        return distance

    ##function to print the path
    def displaypath(self):
        state_trace = [self.state]
        action_trace = [self.move]
        depth_trace = [self.leveldepth]
        patheuristiccost_trace = [self.patheuristiccost]
        totalPatheuristiccost_trace = [self.totalPatheuristiccost]
        heuristic_cost_trace = [self.heuristic_cost]

        # PROVIDING node information while going up the tree
        while self.parent:
            self = self.parent
            state_trace.append(self.state)
            action_trace.append(self.move)
            depth_trace.append(self.leveldepth)
            patheuristiccost_trace.append(self.patheuristiccost)
            totalPatheuristiccost_trace.append(self.totalPatheuristiccost)
            heuristic_cost_trace.append(self.heuristic_cost)

        # print out the path
        step_counter = 0
        while state_trace:
            # print 'step', step_counter
            print("Here g(n)=" + str(patheuristiccost_trace.pop()) + "and h(n)=" + str(
                totalPatheuristiccost_trace.pop() + heuristic_cost_trace.pop()) + "  and the best state to expand is ...")
            state_to_print = state_trace.pop()
            array = [[state_to_print[j * numrows + i] for i in range(numrows)] for j in range(numrows)]
            for row in array:
                print(row)
            print(" \n Expanding node below \n")
            step_counter += 1

    ##to check up move
    def traverseup(self):
        indexofzero=np.where(self.state == 0)
        if indexofzero[0][0] > numrows-1:
            upval = self.state[indexofzero[0][0] - numrows]
            fresh_state = self.state.copy()
            fresh_state[indexofzero] = upval
            fresh_state[indexofzero[0][0] - numrows] = 0
            return fresh_state, upval
        else:
            return False
    ##checkleftmove
    def traverseleft(self):
        indexofzero = np.where(self.state == 0)
        if indexofzero[0][0] % numrows != 0:
            leftval = self.state[indexofzero[0][0] - 1]
            fresh_state = self.state.copy()
            fresh_state[indexofzero] = leftval
            fresh_state[indexofzero[0][0] - 1] = 0
            return fresh_state, leftval
        else:
            return False
    ##move down
    def traversedown(self):
        indexofzero = np.where(self.state == 0)
        if indexofzero[0][0] < numrows*2:
            downval = self.state[indexofzero[0][0] + numrows]
            fresh_state = self.state.copy()
            fresh_state[indexofzero] = downval
            fresh_state[indexofzero[0][0] + 3] = 0
            return fresh_state, downval
        else:
            return False
    ##moveright
    def traverseright(self):
        indexofzero = np.where(self.state == 0)
        if indexofzero[0][0] % numrows < numrows-1:
            rightval = self.state[indexofzero[0][0] + 1]
            fresh_state = self.state.copy()
            fresh_state[indexofzero] = rightval
            fresh_state[indexofzero[0][0] + 1] = 0
            return fresh_state, rightval
        else:
            return False
##taking algorithm input from user
def takealgoinput():
    print("\n \n Enter your choice of algorithm: \n 1. Uniform Cost Search \n 2. A* with Misplaced Tile heuristics \n 3. A* with Manhattan distance heuristics \n \n")
    algorithm_input = input()
    if algorithm_input == "1":
        print("\n \n Proceeding towards goal state with 1. Uniform Cost Search \n \n")
        basenode = make_node(state=input_state, parent=None, move=None, leveldepth=0, patheuristiccost=0, totalPatheuristiccost=0,
                             heuristic_cost=0)
        basenode.pathsearch(goal_State, "1")
    elif algorithm_input == "2":
        print("\n \n Proceeding towards goal state with 2. A* with Misplaced Tile heuristics\n \n")
        basenode = make_node(state=input_state, parent=None, move=None, leveldepth=0, patheuristiccost=0, totalPatheuristiccost=0,
                             heuristic_cost=0)
        basenode.pathsearch(goal_State, "2")

    else:
        print("\n \n Proceeding towards Goal State Using 3. A* with Manhattan distance heuristics \n \n")
        basenode = make_node(state=input_state, parent=None, move=None, leveldepth=0, patheuristiccost=0, totalPatheuristiccost=0,
                             heuristic_cost=0)
        basenode.pathsearch(goal_State, "3")


def isgoalreachable(input_state):
    n = len(input_state)
    count_inversions = 0
    zero_pos = np.where(input_state == 0)
    new_list_without_zero = np.delete(input_state, zero_pos)
    small_n = int(math.sqrt(n))
    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            if new_list_without_zero[i] > new_list_without_zero[j]:
                count_inversions += 1

    width = (n / math.sqrt(n)) % 2
    mod_inv = count_inversions % 2

    two_d_array = np.reshape(input_state, (small_n, small_n))
    zero_index = [i[0] for i in np.where(two_d_array == 0)]
    x = zero_index[0]
    if (n / math.sqrt(n) - x) % 2 == 0:  # even
        zero_position = 0
    else:  # odd
        zero_position = 1

    if ((width == 1 and mod_inv == 0) or (width == 0 and zero_position == 0 and mod_inv == 1) or (
            width == 0 and zero_position == 1 and mod_inv == 0)):
        print("\n \n Congratulations your puzzle is solvable \n \n")
        takealgoinput()
    else:
        print("\n \n Your puzzle can't be solved \n \n")

print("\n \n Hi! You are now entering Varun's 8 puzzle solver. \n \n")
puzzlechoice=int(input("\n \n What puzzle do you want to solve? Hit 8 for puzzle; 15 for 15 puzzle; or 24 for 24 puzzle \n \n"))
if puzzlechoice==8:
    print("\n \n Enter puzzle values row by row separated by tab/space and use 0 to represent an empty tile \n \n")
    row1 = input("\n \n Enter values for row 1 \n \n")
    row1_values=[int(x) for x in row1.split()]
    row2 = input("\n \nEnter values for row 2  \n \n")
    row2_values = [int(x) for x in row2.split()]
    row3 = input("\n \n Enter values for row 3  \n \n")
    row3_values = [int(x) for x in row3.split()]
    goal_State = np.array([1,2,3,4,5,6,7,8,0])
    input_state = row1_values + row2_values + row3_values
    input_state=np.array(input_state)

elif puzzlechoice==15:
    print("Enter puzzle values row by row separated by tab and use 0 to represent an empty tile \n ")
    row1 = input("\n \n Enter values for row 1 \n \n")
    row1_values = [int(x) for x in row1.split()]
    row2 = input("\n \nEnter values for row 2  \n \n")
    row2_values = [int(x) for x in row2.split()]
    row3 = input("\n \n Enter values for row 3  \n \n")
    row3_values = [int(x) for x in row3.split()]
    row4 = input("\n \n Enter values for row 4  \n \n")
    row4_values = [int(x) for x in row4.split()]
    goal_State = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0])
    input_state = row1_values + row2_values + row3_values + row4_values
    input_state = np.array(input_state)
elif puzzlechoice==24:
    print("Enter puzzle values row by row separated by tab and use 0 to represent an empty tile \n ")
    row1 = input("\n \n Enter values for row 1 \n \n")
    row1_values = [int(x) for x in row1.split()]
    row2 = input("\n \nEnter values for row 2  \n \n")
    row2_values = [int(x) for x in row2.split()]
    row3 = input("\n \n Enter values for row 3  \n \n")
    row3_values = [int(x) for x in row3.split()]
    row4 = input("\n \n Enter values for row 4  \n \n")
    row4_values = [int(x) for x in row4.split()]
    row5 = input("\n \n Enter values for row 5  \n \n")
    row5_values = [int(x) for x in row5.split()]
    goal_State = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0])
    input_state = row1_values + row2_values + row3_values + row4_values + row5_values
    input_state = np.array(input_state)
size_inputstate=len(input_state)
# print(math.sqrt(puzzlechoice+1))
print("Printing input state here")
numrows=int(math.sqrt(puzzlechoice+1))
inputstatelist = [[input_state[j * numrows + i] for i in range(numrows)] for j in range(numrows)]
for eachrow in inputstatelist:
    print(eachrow)
print("printing goal state here for your choice", goal_State)
goalstatelist=[[goal_State[j * numrows + i] for i in range(numrows)] for j in range(numrows)]
for eachrow in goalstatelist:
    print(eachrow)
edge=int(math.sqrt(size_inputstate))
isgoalreachable(input_state)




