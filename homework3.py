############################################################
# CIS 521: Homework 3
############################################################

student_name = "Zhengxuan Wu"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import time
import math
import Queue
import random
import copy


############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    tile = 1
    board = []
    for i in range(0,rows):
        temp = []
        for j in range(0,cols):
            if tile > rows * cols - 1:
                tile = 0
            temp.append(tile)
            tile += 1
        board.append(temp)
    return TilePuzzle(board)


class TilePuzzle(object):
    
    # Required
    # tested
    def __init__(self, board):
        '''Initialization for this game'''
        self.board = board
        self.r = len(board)
        # storing rows and cols numbers
        # storing 0-tile's position
        if self.r != 0:
            self.c = len(board[0])
            # storing every element's position
            # storing 0's position as well
            for i in range(0,self.r):
                for j in range(0,self.c):
                    if board[i][j] == 0:
                        self.zero_tile = (i, j)
            
    
    # tested
    def get_board(self):
        '''Call this whenever need to know the board'''
        # print self.board
        return self.board
    # tested
    def perform_move(self, direction):
        '''Perform moves on 0 tile with reasonable bounds'''
        if direction == "left":
            move_in_row = self.zero_tile[0]
            move_in_col = self.zero_tile[1] - 1
        elif direction == "right":
            move_in_row = self.zero_tile[0]
            move_in_col = self.zero_tile[1] + 1
        elif direction == "up":
            move_in_row = self.zero_tile[0] - 1
            move_in_col = self.zero_tile[1]
        elif direction == "down":
            move_in_row = self.zero_tile[0] + 1
            move_in_col = self.zero_tile[1]
        else:
            # the input is totally wrong, then output wrong
            return False
        if move_in_row >= 0 and move_in_col >= 0 and \
            move_in_row < self.r and move_in_col < self.c:
           # perform python swap a,b=b,a
           self.board[move_in_row][move_in_col], \
           self.board[self.zero_tile[0]][self.zero_tile[1]] = \
           self.board[self.zero_tile[0]][self.zero_tile[1]], \
           self.board[move_in_row][move_in_col]
           # again, store the new position of 0
           self.zero_tile = (move_in_row, move_in_col)
           return True
        else:
            # the input is out of boundary here
            return False

    # tested
    def scramble(self, num_moves):
        '''in order to initialize a new game, we random moving the board'''
        for i in range(0,num_moves):
            # using built in random choice function
            dir = random.choice(["up", "down", "left", "right"])
            self.perform_move(dir)

     # tested
    def is_solved(self):
        '''to check if the current state is a solved state'''
        temp1 = self.get_board()
        temp2 = create_tile_puzzle(self.r, self.c)
        tempb = temp2.get_board()
        for i in range(0,self.r):
            for j in range(0,self.c):
                if temp1[i][j] != tempb[i][j]:
                    return False
        return True
    
    def copy(self):
        '''use built in function to deep copy game board'''
        return copy.deepcopy(self)
    
    def successors(self):
        '''return all the possible successors of the current state'''
        for moves in ["up", "down", "left", "right"]:
            temp = self.copy()
            if temp.perform_move(moves) == True:
                yield (moves, temp)
    
    # Required
    def find_solutions_iddfs(self):
        depth = 0
        sol=[]
        while sol ==[]:
            print depth
            sol = self.iddfs_helper(depth)
            depth+=1;
        for s in sol:
            yield s

    def iddfs_helper(self, limit):
        '''Iterative Deepening Search'''
        # Initializations
        stack=[]
        explored_set=set()
        parent={}
        moves={}
        depthes={}
        solutions=[]
        # initialization all those properties
        stack.append((self,0))
        explored_set.add(tuple(tuple(x) for x in self.get_board()))
        parent[self]=self
        moves[self]=""
        depth_curr=0
        # loop with two conditions: not running out of elements
        # depth is not exceeding the limit
        while stack != []:
            # pop out the first element
            item=stack.pop(0)
            frontier = item[0]
            depth_curr = item[1]
            # check if solved, actually it is only valid for round 1
            if frontier.is_solved():
                move_list=[]
                node=frontier
                while (parent[node]!=node):
                    node=parent[node]
                    move_list.append(moves[node])
                # get one solution but not return it, because we are
                # potentially having another solution/s, we keep iterating
                solutions.append(list(reversed(move_list)))
            # if it is not solved, adding elements to the stack
            if depth_curr<limit:
                for move,new_p in frontier.successors():
                    if tuple(tuple(x) for x in new_p.get_board()) not in explored_set:
                        parent[new_p]=frontier
                        moves[new_p]=move
                        if new_p.is_solved():
                            move_list=[]
                            node=new_p
                            while (parent[node]!=node):
                                move_list.append(moves[node])
                                node=parent[node]
                            solutions.append(list(reversed(move_list)))
                        else:
                            stack.insert(0,(new_p,depth_curr+1))
                            explored_set.add(tuple(tuple(x) for x in new_p.get_board()))
        return solutions

    # Required
    def find_solution_a_star(self):
        queue = Queue.PriorityQueue()
        explored_set = set()
        parent = {}
        parent[self] = self
        moves = {}
        h={}
        moves[self] = ""
        goal = create_tile_puzzle(self.r, self.c)
        queue.put((self.h_fun(goal), 0, self))
        explored_set.add(tuple(tuple(x) for x in self.get_board()))
        h[tuple(tuple(x) for x in self.get_board())]=0
        while queue != []:
            current = queue.get()
            puzzle_instance = current[2]
            g = current[1]
            if puzzle_instance.is_solved():
                # backtrack through the solution to get the moves
                solution = []
                node = puzzle_instance
                while(parent[node] != node):
                    solution.append(moves[node])
                    node = parent[node]
                return list(reversed(solution))
            # print "Parent:"
            # print puzzle_instance.get_board()
            for move, successor in puzzle_instance.successors():
                cost = successor.h_fun(goal) + g + 1
                if tuple(tuple(x) for x in successor.get_board()) not in explored_set or cost < h[tuple(tuple(x) for x in successor.get_board())]:
                    parent[successor] = puzzle_instance
                    moves[successor] = move
                    h[tuple(tuple(x) for x in successor.get_board())]=cost
                    # add the node to the queue with it's scores and mark it as
                    # explored
                    queue.put(\
                        (cost, g + 1, successor))
                    explored_set.add(tuple(tuple(x)\
                              for x in successor.get_board()))
        return None


    def h_fun(self, goal):
        '''histerics function'''
        h = 0
        position_recorder = {}
        goal_recorder = {}
        curr_board = self.get_board()
        goal_board = goal.get_board()
        rows = len(curr_board)
        # record the current board
        if rows != 0:
            cols = len(curr_board[0])
            for i in range(0,rows):
                for j in range(0,cols):
                    position_recorder[curr_board[i][j]]=(i,j)
                    goal_recorder[goal_board[i][j]]=(i,j)
        for i in position_recorder:
            h += abs(position_recorder[i][0]-goal_recorder[i][0])+\
                abs(position_recorder[i][1]-goal_recorder[i][1])
        return h

# test
#p = create_tile_puzzle(3, 3)
#p.get_board()
#p.scramble(4)
#p.get_board()
#print p.is_solved()
#
#b = [[1,2,3], [4,0,5], [6,7,8]]
#p = TilePuzzle(b)
#for move, new_p in p.successors():
#    print move, new_p.get_board()
#
#b = [[4,1,3], [0,6,2], [7,8,5]]
#g = [[1,2,3],[4,5,6],[7,8,0]]
#p = TilePuzzle(b)
#g = TilePuzzle(g)
#p.h_fun(g)

#b = [[4, 5, 7], [6, 8, 2], [3, 0, 1]]
#p = TilePuzzle(b)
#print p.is_solved()
#print p.find_solution_a_star()
############################################################
# Section 2: Grid Navigation
############################################################


def find_path(start, goal, scene):
    # initialization
    start_r=start[0]
    start_c=start[1]
    goal_r=goal[0]
    goal_c=goal[1]
    # exam the corner cases
    if scene[start_r][start_c] == True or scene[goal_r][goal_c] == True:
        print 'No Solution Is Found'
        return None
    
    queue = Queue.PriorityQueue()
    explored_set = set()
    current_state=start
    parent = {}
    parent[current_state] = start
    costs={}
    costs[start] = 0
    queue.put((h(current_state,goal), 0, current_state))
    explored_set.add(start)
    solution = []
    while queue != []:
        item=queue.get()
        frontier = item[2]
        g = item[1]
        if frontier == goal:
            node = frontier
            while(parent[node]!=node):
                solution.append(node)
                node = parent[node]
            solution.append(start)
            return list(reversed(solution))
        # node visited, recorded in explored set
        for new_p in find_successors(frontier,scene,explored_set):
            cost = h(new_p,goal) + g + 1
            if new_p not in explored_set or cost < costs[frontier]:
                costs[new_p] = cost
                parent[new_p] = frontier
                queue.put((cost,g+1,new_p))
                explored_set.add(new_p)
    return None

# pass in a tuple representating position, and the scene
def find_successors(curr,scene, explored_set):
    potential_moves = [[curr[0]+1, curr[1]],\
                      [curr[0], curr[1]+1],\
                      [curr[0]-1, curr[1]],\
                      [curr[0], curr[1]-1],\
                      [curr[0]+1, curr[1]+1],\
                      [curr[0]-1, curr[1]-1],\
                      [curr[0]+1, curr[1]-1],\
                      [curr[0]-1, curr[1]+1]]
    for i in potential_moves:
        if i[0]>=0 and i[0]<len(scene) and i[1]>=0 and i[1]<len(scene[0]) and scene[i[0]][i[1]] == False:
            yield (i[0],i[1])

def h(curr,goal):
    ''' h function, calculating the norm2 distance'''
    x = (curr[1] - goal[1])*(curr[1] - goal[1])
    y = (curr[0] - goal[0])*(curr[0] - goal[0])
    return math.sqrt(x+y)

# print find_path((1, 0), (2, 1), scene)


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################


class DiskMovement(object):
    def __init__(self, disks, length, n):
        self.disks = list(disks)
        self.length = length
        self.n = n
    
    def successors(self):
        i = 0
        li = self.disks
        while i < len(self.disks):
            if li[i] != 0:
                if i + 1 < self.length:
                    if li[i+1] == 0:
                        temp = list(self.disks)
                        disk_to_move = temp[i]
                        temp[i] = 0
                        temp[i+1] = disk_to_move
                        yield((i, i+1), DiskMovement(temp,self.length,self.n))
                if i + 2 < self.length:
                    if li[i+2] == 0 and li[i+1] !=0:
                        temp = list(self.disks)
                        disk_to_move = temp[i]
                        temp[i] = 0
                        temp[i+2] = disk_to_move
                        yield((i, i+2), DiskMovement(temp,self.length,self.n))
                if i-1 >= 0:
                    if li[i-1] == 0:
                        temp = list(self.disks)
                        disk_to_move = temp[i]
                        temp[i] = 0
                        temp[i-1] = disk_to_move
                        yield((i, i-1), DiskMovement(temp,self.length,self.n))
                if i - 2 >= 0:
                    if li[i-2] == 0 and li[i-1] !=0:
                        temp = list(self.disks)
                        disk_to_move = temp[i]
                        temp[i] = 0
                        temp[i-2] = disk_to_move
                        yield((i, i-2), DiskMovement(temp,self.length,self.n))
            i += 1



#Problem Representation [diskId1, diskId2, diskId3, 0, 0 ..]
def solve_distinct_disks(length, n, initial_distinct_disk = []):
    initialDisks = []
    # Initialization of disk
    if initial_distinct_disk == []:
        #Disk numbers starting from 1
        initialDisks = [i+1 for i in xrange(n)]
        #fill empty slots with 0
        for i in xrange(length - n):
            initialDisks.append(0)
    else:
        initialDisks = initial_distinct_disk
    # Setting Up The Goal Conditions
    goalDisks = []
    for i in xrange(length - n):
        goalDisks.append(0)
    for element in list(reversed([i+1 for i in xrange(n)])):
        goalDisks.append(element)
    goal = DiskMovement(goalDisks, length, n)
    dm = DiskMovement(initialDisks, length, n)

    # Using PriorityQueue To Solve
    queue = Queue.PriorityQueue()
    parent = {}
    moves = {}
    explored_set = set()
    solution = []
    current_state = dm
    parent[dm] = dm
    moves[dm] = ()
    explored_set.add(tuple(dm.disks))
    queue.put((h_disk(current_state,goal), 0, current_state))
    # Check if the current node is the answer
    if is_solved_distinct(dm):
        return moves[dm]
    while queue != []:
        item = queue.get()
        diskInstance = item[2]
        g= item[1]
        if is_solved_distinct(diskInstance):
            node = diskInstance
            while(parent[node] != node):
                solution.append(moves[node])
                node = parent[node]
            return list(reversed(solution))
        for move, neighbor in diskInstance.successors():
            cost =h_disk(neighbor,goal) + g + 1
            if tuple(neighbor.disks) not in explored_set:
                parent[neighbor] = diskInstance
                moves[neighbor] = move
                explored_set.add(tuple(neighbor.disks))
                queue.put((cost, g+1, neighbor))
    return None

def is_solved_distinct(dm):
    i = len(dm.disks) - 1
    diskId = 1
    while diskId <= dm.n:
        if dm.disks[i] != diskId:
            return False
        i -= 1
        diskId += 1
    return True

# unique h function for every state
def h_disk(curr,goal):
    curr_disk = curr.disks
    goal_disk = goal.disks
    sum = 0;
    for element in [i+1 for i in range(0,curr.n)]:
        sum += abs(goal_disk.index(i) - curr_disk.index(i))
    return sum



# print solve_distinct_disks(5, 3)



############################################################
# Section 4: Dominoes Game
############################################################

def create_dominoes_game(rows, cols):
    board_temp  = [[False for i in range(cols)] for j in range(rows)]
    return DominoesGame(board_temp)

class DominoesGame(object):
    
    node_num = 0
    
    # Required
    def __init__(self, board):
        self.board = board;
        self.rows = len(board)
        if self.rows != 0:
            self.cols = len(board[0])
    
    def get_board(self):
        print self.board
        return self.board
    
    def reset(self):
        board_temp  = [[False for i in range(self.cols)] for j in range(self.rows)]
        self.board = board_temp
    
    def is_legal_move(self, row, col, vertical):
        if vertical:
            if row+1<self.rows and col<self.cols:
                if self.board[row + 1][col] == False and self.board[row][col] == False:
                    return True
                else:
                    return False
            else:
                return False
        else:
            if row<self.rows and col+1<self.cols:
                if self.board[row][col] == False and self.board[row][col + 1] == False:
                    return True
                else:
                    return False
            else:
                return False
    
    def legal_moves(self, vertical):
        for i in range(0,self.rows):
            for j in range(0,self.cols):
                if self.is_legal_move(i, j, vertical):
                    yield (i, j)
    
    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical):
            if vertical:
                self.board[row][col] = True
                self.board[row + 1][col] = True
            else:
                self.board[row][col] = True
                self.board[row][col + 1] = True
        else:
            print 'Move Illegal!'
    
    def game_over(self, vertical):
        temp = list(self.legal_moves(vertical))
        if temp == []:
            return True
        else:
            return False
    
    def copy(self):
        return copy.deepcopy(self)
    
    def successors(self, vertical):
        for moves in self.legal_moves(vertical):
            temp = self.copy()
            temp.perform_move(moves[0], moves[1], vertical)
            yield (moves, temp)
    
    def get_random_move(self, vertical):
        temp = list(self.legal_moves(vertical))
        print random.choice(temp)
        print temp
        return random.choice(temp)
    
    # Required
    def evaluate_board(self, curr_move):
        temp1 =len(list(self.legal_moves(curr_move)))
        temp2 =len(list(self.legal_moves(not curr_move)))
        return temp1 - temp2
    
    def alpha_beta_search(self, level_remain, alpha, beta, vertical, curr_move, Player):
        if self.game_over(vertical) or level_remain == 0:
            DominoesGame.node_num = DominoesGame.node_num + 1
            return ((0,0), self.evaluate_board(curr_move))
        if Player:
            return self.max_value(level_remain, alpha, beta, vertical, curr_move, Player)
        else:
            return self.min_value(level_remain, alpha, beta, vertical, curr_move, Player)

    # Required
    def get_best_move(self, vertical, limit):
        move, value = self.alpha_beta_search(limit, float("-inf"), float("inf"), vertical, vertical, True)
        temp = DominoesGame.node_num
        DominoesGame.node_num = 0
        return move, value, temp

    def max_value(self, level_remain, alpha, beta, vertical, curr_move, Player):
        v = float("-inf")
        required_move = tuple()
        for move, new_p in self.successors(vertical):
            position, temp = new_p.alpha_beta_search(level_remain - 1, alpha, beta, not vertical, curr_move,False)
            if temp > v:
                v = temp
                required_move = move
            alpha = max(alpha, v)
            if beta <= alpha:
                break
        return (required_move, v)

    def min_value(self, level_remain, alpha, beta, vertical, curr_move, Player):
        v = float("inf")
        required_move = tuple()
        for move, new_p in self.successors(vertical):
            position, temp = new_p.alpha_beta_search(level_remain - 1, alpha, beta, not vertical, curr_move, True)
            if temp < v:
                v = temp
                required_move = move
            beta = min(beta, v)
            if beta <= alpha:
                break
        return (required_move, v)

#b = [[False] * 3 for i in range(3)]
#g = DominoesGame(b)
#g.get_random_move(True)
#g.get_board()
#g.perform_move(0, 1, True)
#g.get_board()
#print g.get_best_move(False, 2)


############################################################
# Section 5: Feedback
############################################################

feedback_question_1 = """
    12 hrs
    """

feedback_question_2 = """
    The implementation of the knowledge
    """

feedback_question_3 = """
    I like all of them
    """