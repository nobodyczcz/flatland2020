"""
TransitionMap and derived classes.
"""

import numpy as np
from importlib_resources import path
from numpy import array

from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transitions import Transitions


class TransitionMap:
    """
    Base TransitionMap class.

    Generic class that implements a collection of transitions over a set of
    cells.
    """

    def get_transitions(self, cell_id):
        """
        Return a tuple of transitions available in a cell specified by
        `cell_id' (e.g., a tuple of size of the maximum number of transitions,
        with values 0 or 1, or potentially in between,
        for stochastic transitions).

        Parameters
        ----------
        cell_id : [cell identifier]
            The cell_id object depends on the specific implementation.
            It generally is an int (e.g., an index) or a tuple of indices.

        Returns
        -------
        tuple
            List of the validity of transitions in the cell.

        """
        raise NotImplementedError()

    def set_transitions(self, cell_id, new_transitions):
        """
        Replaces the available transitions in cell `cell_id' with the tuple
        `new_transitions'. `new_transitions' must have
        one element for each possible transition.

        Parameters
        ----------
        cell_id : [cell identifier]
            The cell_id object depends on the specific implementation.
            It generally is an int (e.g., an index) or a tuple of indices.
        new_transitions : tuple
            Tuple of new transitions validitiy for the cell.

        """
        raise NotImplementedError()

    def get_transition(self, cell_id, transition_index):
        """
        Return the status of whether an agent in cell `cell_id' can perform a
        movement along transition `transition_index (e.g., the NESW direction
        of movement, for agents on a grid).

        Parameters
        ----------
        cell_id : [cell identifier]
            The cell_id object depends on the specific implementation.
            It generally is an int (e.g., an index) or a tuple of indices.
        transition_index : int
            Index of the transition to probe, as index in the tuple returned by
            get_transitions(). e.g., the NESW direction of movement, for agents
            on a grid.

        Returns
        -------
        int or float (depending on Transitions used)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """
        raise NotImplementedError()

    def set_transition(self, cell_id, transition_index, new_transition):
        """
        Replaces the validity of transition to `transition_index' in cell
        `cell_id' with the new `new_transition'.


        Parameters
        ----------
        cell_id : [cell identifier]
            The cell_id object depends on the specific implementation.
            It generally is an int (e.g., an index) or a tuple of indices.
        transition_index : int
            Index of the transition to probe, as index in the tuple returned by
            get_transitions(). e.g., the NESW direction of movement, for agents
            on a grid.
        new_transition : int or float (depending on Transitions used)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """
        raise NotImplementedError()


class GridTransitionMap(TransitionMap):
    """
    Implements a TransitionMap over a 2D grid.

    GridTransitionMap implements utility functions.
    """

    def __init__(self, width, height, transitions: Transitions = Grid4Transitions([])):
        """
        Builder for GridTransitionMap object.

        Parameters
        ----------
        width : int
            Width of the grid.
        height : int
            Height of the grid.
        transitions : Transitions object
            The Transitions object to use to encode/decode transitions over the
            grid.

        """

        self.width = width
        self.height = height
        self.transitions = transitions

        self.grid = np.zeros((height, width), dtype=self.transitions.get_type())

    def get_full_transitions(self, row, column):
        """
        Returns the full transitions for the cell at (row, column) in the format transition_map's transitions.

        Parameters
        ----------
        row: int
        column: int
            (row,column) specifies the cell in this transition map.

        Returns
        -------
        self.transitions.get_type()
            The cell content int the format of this map's Transitions.

        """
        return self.grid[row][column]

    def get_transitions(self, row, column, orientation):
        """
        Return a tuple of transitions available in a cell specified by
        `cell_id' (e.g., a tuple of size of the maximum number of transitions,
        with values 0 or 1, or potentially in between,
        for stochastic transitions).

        Parameters
        ----------
        cell_id : tuple
            The cell_id indices a cell as (column, row, orientation),
            where orientation is the direction an agent is facing within a cell.
            Alternatively, it can be accessed as (column, row) to return the
            full cell content.

        Returns
        -------
        tuple
            List of the validity of transitions in the cell as given by the maps transitions.

        """
        return self.transitions.get_transitions(self.grid[row][column], orientation)

    def set_transitions(self, cell_id, new_transitions):
        """
        Replaces the available transitions in cell `cell_id' with the tuple
        `new_transitions'. `new_transitions' must have
        one element for each possible transition.

        Parameters
        ----------
        cell_id : tuple
            The cell_id indices a cell as (column, row, orientation),
            where orientation is the direction an agent is facing within a cell.
            Alternatively, it can be accessed as (column, row) to replace the
            full cell content.
        new_transitions : tuple
            Tuple of new transitions validitiy for the cell.

        """
        assert len(cell_id) in (2, 3), \
            'GridTransitionMap.set_transitions() ERROR: cell_id tuple must have length 2 or 3.'
        if len(cell_id) == 3:
            self.grid[cell_id[0]][cell_id[1]] = self.transitions.set_transitions(self.grid[cell_id[0]][cell_id[1]],
                                                                                 cell_id[2],
                                                                                 new_transitions)
        elif len(cell_id) == 2:
            self.grid[cell_id[0]][cell_id[1]] = new_transitions

    def get_transition(self, cell_id, transition_index):
        """
        Return the status of whether an agent in cell `cell_id' can perform a
        movement along transition `transition_index (e.g., the NESW direction
        of movement, for agents on a grid).

        Parameters
        ----------
        cell_id : tuple
            The cell_id indices a cell as (column, row, orientation),
            where orientation is the direction an agent is facing within a cell.
        transition_index : int
            Index of the transition to probe, as index in the tuple returned by
            get_transitions(). e.g., the NESW direction of movement, for agents
            on a grid.

        Returns
        -------
        int or float (depending on Transitions used in the )
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """

        assert len(cell_id) == 3, \
            'GridTransitionMap.get_transition() ERROR: cell_id tuple must have length 2 or 3.'
        return self.transitions.get_transition(self.grid[cell_id[0]][cell_id[1]], cell_id[2], transition_index)

    def set_transition(self, cell_id, transition_index, new_transition, remove_deadends=False):
        """
        Replaces the validity of transition to `transition_index' in cell
        `cell_id' with the new `new_transition'.


        Parameters
        ----------
        cell_id : tuple
            The cell_id indices a cell as (column, row, orientation),
            where orientation is the direction an agent is facing within a cell.
        transition_index : int
            Index of the transition to probe, as index in the tuple returned by
            get_transitions(). e.g., the NESW direction of movement, for agents
            on a grid.
        new_transition : int or float (depending on Transitions used in the map.)
            Validity of the requested transition (e.g.,
            0/1 allowed/not allowed, a probability in [0,1], etc...)

        """
        assert len(cell_id) == 3, \
            'GridTransitionMap.set_transition() ERROR: cell_id tuple must have length 3.'
        self.grid[cell_id[0]][cell_id[1]] = self.transitions.set_transition(
            self.grid[cell_id[0]][cell_id[1]],
            cell_id[2],
            transition_index,
            new_transition,
            remove_deadends)

    def save_transition_map(self, filename):
        """
        Save the transitions grid as `filename', in npy format.

        Parameters
        ----------
        filename : string
            Name of the file to which to save the transitions grid.

        """
        np.save(filename, self.grid)

    def load_transition_map(self, package, resource):
        """
        Load the transitions grid from `filename' (npy format).
        The load function only updates the transitions grid, and possibly width and height, but the object has to be
        initialized with the correct `transitions' object anyway.

        Parameters
        ----------
        package : string
            Name of the package from which to load the transitions grid.
        resource : string
            Name of the file from which to load the transitions grid within the package.
        override_gridsize : bool
            If override_gridsize=True, the width and height of the GridTransitionMap object are replaced with the size
            of the map loaded from `filename'. If override_gridsize=False, the transitions grid is either cropped (if
            the grid size is larger than (height,width) ) or padded with zeros (if the grid size is smaller than
            (height,width) )

        """
        with path(package, resource) as file_in:
            new_grid = np.load(file_in)

        new_height = new_grid.shape[0]
        new_width = new_grid.shape[1]

        self.width = new_width
        self.height = new_height
        self.grid = new_grid

    def is_dead_end(self, rcPos):
        """
        Check if the cell is a dead-end.

        Parameters
        ----------
        rcPos: Tuple[int,int]
            tuple(row, column) with grid coordinate
        Returns
        -------
        boolean
            True if and only if the cell is a dead-end.
        """
        nbits = 0
        tmp = self.get_full_transitions(rcPos[0], rcPos[1])
        while tmp > 0:
            nbits += (tmp & 1)
            tmp = tmp >> 1
        return nbits == 1

    def is_simple_turn(self, rcPos):
        """
        Check if the cell is a left/right simple turn

        Parameters
        ----------
            rcPos: Tuple[int,int]
                tuple(row, column) with grid coordinate
        Returns
        -------
            boolean
                True if and only if the cell is a left/right simple turn.
        """
        tmp = self.get_full_transitions(rcPos[0], rcPos[1])

        def is_simple_turn(trans):
            all_simple_turns = set()
            for trans in [int('0100000000000010', 2),  # Case 1b (8)  - simple turn right
                          int('0001001000000000', 2)  # Case 1c (9)  - simple turn left]:
                          ]:
                for _ in range(3):
                    trans = self.transitions.rotate_transition(trans, rotation=90)
                    all_simple_turns.add(trans)
            return trans in all_simple_turns

        return is_simple_turn(tmp)

    def check_path_exists(self, start, direction, end):
        # BFS - Check if a path exists between the 2 nodes

        visited = set()
        stack = [(start, direction)]
        while stack:
            node = stack.pop()
            node_position = node[0]
            node_direction = node[1]
            if node_position[0] == end[0] and node_position[1] == end[1]:
                return True
            if node not in visited:
                visited.add(node)

                moves = self.get_transitions(node_position[0], node_position[1], node_direction)
                for move_index in range(4):
                    if moves[move_index]:
                        stack.append((get_new_position(node_position, move_index),
                                      move_index))

        return False

    def cell_neighbours_valid(self, rcPos, check_this_cell=False):
        """
        Check validity of cell at rcPos = tuple(row, column)
        Checks that:
        - surrounding cells have inbound transitions for all the
            outbound transitions of this cell.

        These are NOT checked - see transition.is_valid:
        - all transitions have the mirror transitions (N->E <=> W->S)
        - Reverse transitions (N -> S) only exist for a dead-end
        - a cell contains either no dead-ends or exactly one

        Returns: True (valid) or False (invalid)
        """
        cell_transition = self.grid[tuple(rcPos)]

        if check_this_cell:
            if not self.transitions.is_valid(cell_transition):
                return False

        gDir2dRC = self.transitions.gDir2dRC  # [[-1,0] = N, [0,1]=E, etc]
        grcPos = array(rcPos)
        grcMax = self.grid.shape

        binTrans = self.get_full_transitions(*rcPos)  # 16bit integer - all trans in/out
        lnBinTrans = array([binTrans >> 8, binTrans & 0xff], dtype=np.uint8)  # 2 x uint8
        g2binTrans = np.unpackbits(lnBinTrans).reshape(4, 4)  # 4x4 x uint8 binary(0,1)
        gDirOut = g2binTrans.any(axis=0)  # outbound directions as boolean array (4)
        giDirOut = np.argwhere(gDirOut)[:, 0]  # valid outbound directions as array of int

        # loop over available outbound directions (indices) for rcPos
        for iDirOut in giDirOut:
            gdRC = gDir2dRC[iDirOut]  # row,col increment
            gPos2 = grcPos + gdRC  # next cell in that direction

            # Check the adjacent cell is within bounds
            # if not, then this transition is invalid!
            if np.any(gPos2 < 0):
                return False
            if np.any(gPos2 >= grcMax):
                return False

            # Get the transitions out of gPos2, using iDirOut as the inbound direction
            # if there are no available transitions, ie (0,0,0,0), then rcPos is invalid
            t4Trans2 = self.get_transitions(*gPos2, iDirOut)
            if any(t4Trans2):
                continue
            else:
                return False

        return True

    def fix_neighbours(self, rcPos, check_this_cell=False):
        """
        Check validity of cell at rcPos = tuple(row, column)
        Checks that:
        - surrounding cells have inbound transitions for all the
            outbound transitions of this cell.

        These are NOT checked - see transition.is_valid:
        - all transitions have the mirror transitions (N->E <=> W->S)
        - Reverse transitions (N -> S) only exist for a dead-end
        - a cell contains either no dead-ends or exactly one

        Returns: True (valid) or False (invalid)
        """
        cell_transition = self.grid[tuple(rcPos)]

        if check_this_cell:
            if not self.transitions.is_valid(cell_transition):
                return False

        gDir2dRC = self.transitions.gDir2dRC  # [[-1,0] = N, [0,1]=E, etc]
        grcPos = array(rcPos)
        grcMax = self.grid.shape

        binTrans = self.get_full_transitions(*rcPos)  # 16bit integer - all trans in/out
        lnBinTrans = array([binTrans >> 8, binTrans & 0xff], dtype=np.uint8)  # 2 x uint8
        g2binTrans = np.unpackbits(lnBinTrans).reshape(4, 4)  # 4x4 x uint8 binary(0,1)
        gDirOut = g2binTrans.any(axis=0)  # outbound directions as boolean array (4)
        giDirOut = np.argwhere(gDirOut)[:, 0]  # valid outbound directions as array of int

        # loop over available outbound directions (indices) for rcPos
        for iDirOut in giDirOut:
            gdRC = gDir2dRC[iDirOut]  # row,col increment
            gPos2 = grcPos + gdRC  # next cell in that direction

            # Check the adjacent cell is within bounds
            # if not, then this transition is invalid!
            if np.any(gPos2 < 0):
                return False
            if np.any(gPos2 >= grcMax):
                return False

            # Get the transitions out of gPos2, using iDirOut as the inbound direction
            # if there are no available transitions, ie (0,0,0,0), then rcPos is invalid
            t4Trans2 = self.get_transitions(*gPos2, iDirOut)
            if any(t4Trans2):
                continue
            else:
                self.set_transition((gPos2[0], gPos2[1], iDirOut), mirror(iDirOut), 1)
                return False

        return True

    def fix_transitions(self, rcPos):
        """
        Fixes broken transitions
        """
        gDir2dRC = self.transitions.gDir2dRC  # [[-1,0] = N, [0,1]=E, etc]
        grcPos = array(rcPos)
        grcMax = self.grid.shape

        # Transition elements
        transitions = RailEnvTransitions()
        cells = transitions.transition_list
        simple_switch_east_south = transitions.rotate_transition(cells[10], 90)
        simple_switch_west_south = transitions.rotate_transition(cells[2], 270)
        symmetrical = cells[6]
        double_slip = cells[5]
        three_way_transitions = [simple_switch_east_south, simple_switch_west_south, symmetrical]
        # loop over available outbound directions (indices) for rcPos
        self.set_transitions(rcPos, 0)

        incoming_connections = np.zeros(4)
        for iDirOut in np.arange(4):
            gdRC = gDir2dRC[iDirOut]  # row,col increment
            gPos2 = grcPos + gdRC  # next cell in that direction

            # Check the adjacent cell is within bounds
            # if not, then ignore it for the count of incoming connections
            if np.any(gPos2 < 0):
                continue
            if np.any(gPos2 >= grcMax):
                continue

            # Get the transitions out of gPos2, using iDirOut as the inbound direction
            # if there are no available transitions, ie (0,0,0,0), then rcPos is invalid
            connected = 0
            for orientation in range(4):
                connected += self.get_transition((gPos2[0], gPos2[1], orientation), mirror(iDirOut))
            if connected > 0:
                incoming_connections[iDirOut] = 1

        number_of_incoming = np.sum(incoming_connections)
        # Only one incoming direction --> Straight line
        if number_of_incoming == 1:
            for direction in range(4):
                if incoming_connections[direction] > 0:
                    self.set_transition((rcPos[0], rcPos[1], mirror(direction)), direction, 1)
        # Connect all incoming connections
        if number_of_incoming == 2:
            connect_directions = np.argwhere(incoming_connections > 0)
            self.set_transition((rcPos[0], rcPos[1], mirror(connect_directions[0])), connect_directions[1], 1)
            self.set_transition((rcPos[0], rcPos[1], mirror(connect_directions[1])), connect_directions[0], 1)

        # Find feasible connection for three entries
        if number_of_incoming == 3:
            transition = np.random.choice(three_way_transitions, 1)
            hole = np.argwhere(incoming_connections < 1)[0][0]
            transition = transitions.rotate_transition(transition, int(hole * 90))
            self.set_transitions((rcPos[0], rcPos[1]), transition)

        # Make a double slip switch
        if number_of_incoming == 4:
            rotation = np.random.randint(2)
            transition = transitions.rotate_transition(double_slip, int(rotation * 90))
            self.set_transitions((rcPos[0], rcPos[1]), transition)
        return True


def mirror(dir):
    return (dir + 2) % 4
# TODO: improvement override __getitem__ and __setitem__ (cell contents, not transitions?)
