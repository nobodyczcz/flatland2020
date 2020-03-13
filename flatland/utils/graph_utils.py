import networkx as nx
import numpy as np
import pandas as pd
import json
from numpy import array
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import cm

# turn a transition into a string of binary
def trans_int_to_binstr(intTrans):
    sbinTrans = format(intTrans, "#018b")[2:]
    return "_".join(["NESW"[i] + sbinTrans[i*4:(i*4 + 4)] for i in range(0, 4)])


# Turn a transition into a 4x4 array of 0s and 1s
def trans_int_to_4x4(intTrans):
    arrBytes = np.array([intTrans >> 8, intTrans & 0xff], dtype=np.uint8)
    #print(arrBytes)
    arrBool = np.array(np.zeros((4,4)), dtype=np.bool)
    arrBool = np.unpackbits(arrBytes)
    arrBool4x4 = arrBool.reshape((4,4))
    return arrBool4x4


# Turn a transition int into a string list, eg EE, EN, SW, WW
def trans_int_to_nesw(intTrans):
    astrNESW = np.array(list("NESW"))
    a2Trans = trans_int_to_4x4(intTrans)
    lstrTrans = [ 
        np.char.add(astrNESW[iInDir], astrNESW[np.where(aiOutDirs)[0]])
        for iInDir, aiOutDirs in enumerate(a2Trans) 
        ]
    return ",".join(list(np.concatenate(lstrTrans)))


def get_rail_transitions_df(env):
    ll = []
    for rowcol, iTrans in np.ndenumerate(env.rail.grid):
        ll.append([rowcol, iTrans, trans_int_to_binstr(iTrans), trans_int_to_nesw(iTrans)])
    df = pd.DataFrame(ll, columns=["rowcol", "Integer", "Binary", "NESW"])
    df = df[df.Integer > 0]
    return df

def neighbors(G, nbunch, edge_types=None, outer_edge_types=None):
    """
        From a graph G, and nodes nbunch, return the list of nodes who are neighbors 
        of nbunch via edges of types edge_types.  Include those edges, and edges between
        the neighbors of type types outer_edge_types, in the second return value.

        ***Unfinished***
    """
    lnNeighbors = []
    if edge_types is not None:
        if (type(edge_types) is str):
            edge_types = [edge_types]
    else:
        edge_types = ["grid", "hold", "dir"]
    
    if outer_edge_types is not None:
        if (type(outer_edge_types) is str):
            outer_edge_types = [outer_edge_types]
    else:
        outer_edge_types = ["grid", "hold", "dir"]
    
    for u in nbunch:
        for v in G.adj[u]:
            edge_type = G.adj[u][v]["type"]
            if edge_type in edge_types:
                lnNeighbors.append(v)
    
    # get all the edges from the original nbunch
    lEdges = [ (u,v,d) for u,v,d in G.edges(nbunch, data=True) if d["type"] in edge_types]
    
    Gneighbours = G.subgraph(lnNeighbors)
    
    # now get the edges between the neighbours
    lEdges += [ (u,v,d) for u,v,d in G.edges(lnNeighbors, data=True) if d["type"] in outer_edge_types and (u in Gneighbours) and (v in Gneighbours)]
    
    #G4d.add_nodes_from(list(G2.subgraph(lnRails).nodes(data=True)))
    #G4d.add_edges_from(lEdges)

    return lnNeighbors, lEdges

def grid_node_for_rails(G, nbunch):
    for n in nbunch:
        G.pred

def get_simple_path(G, u):
    """ Follow a linear path in G starting at u, terminating as soon as there is 
        more than (or less than) 1 successor, ie when the linear section ends.
        The branching node is not included.
    """
    visited = OrderedDict()
    visited[u] = 1
    v = u
    while True: 
        lSucc = list(G.successors(v))
        if len(lSucc) != 1:
            break
        v = lSucc[0]
        if v in visited:
            break
        visited[v] = 1
    return list(visited.keys())


def plotGraphEnv(G, env, aImg, space=0.3, figsize=(12,8),
                 show_labels=(), show_edges=("dir"),
                 show_nodes="all", node_colors=None, edge_colors=None, 
                 alpha_img=0.2,
                 node_size=300):

    xyDir = array([[0,1], [1,0], [0,-1], [-1,0]])
    xy2 = array([xyDir[(i+1) % 4,:] for i in range(4)])
    
    if figsize is not None:
        plt.figure(figsize=figsize)
        
    rows, cols = env.rail.grid.shape
    plt.imshow(aImg, extent=(-.5,cols-.5,.5-rows,0.5), alpha=alpha_img)
    
    if show_nodes == "all":
        nodelist = G.nodes()
    else:
        nodelist = [ n for n,d in G.nodes(data=True) if d["type"] in show_nodes]

    if node_colors is None:
        node_colors = {"grid":"red", "rail":"lightblue"}
    
    if edge_colors is None:
        edge_colors = {"grid":"gray", "hold":"blue", "dir":"green"}
    
    edgelist = [(u, v) for u, v, d in G.edges(data=True) if d["type"] in show_edges]
    dnDat = G.nodes(data=True)
    deDat = {(u, v): d for u, v, d in G.edges(data=True) if d["type"] in show_edges}
    
    nx.draw(G,
            labels={n:str(n) for n,d in G.nodes(data=True) if d["type"] in show_labels},
            node_color=[ node_colors[dnDat[n]["type"]] for n in nodelist], 
            pos={n:(
                    n[1] if len(n)==2 else n[1] - space * xy2[n[2],0],
                    -n[0] if len(n)==2 else -n[0] - space * xy2[n[2],1]  )
                for n in G.nodes()},
            edgelist=edgelist,
            edge_color=[edge_colors[deDat[(u,v)]["type"]] for u,v in edgelist],
            nodelist=nodelist,
            node_size=node_size
        )


def genStartTargetDirs(G, env, ):
    lGpaths = []        # list of paths as graphs
    llnPaths = []       # list of paths as list of nodes (rail nodes + dir edges)
    lltStartTarg = []   # [start, target] for each direction permutation 
                        # as list of 2-lists of 3-tuples (rail nodes)

    nAgents = len(env.agents)
    print("nAgents:", nAgents)
    if True:
        for iAgent, agent in enumerate(env.agents[:nAgents]):
            nStart = agent.initial_position  # grid node (row,col)
            nEnd = agent.target  # grid node
            lnStartDir = []

            # check the start grid node is in the graph
            if nStart in G and nEnd in G:
                # Find all the rail node neighbours
                lnNeighbours = list(G.neighbors(nStart))
                for nNbr in lnNeighbours:
                    if len(nNbr)==3:  # rail nodes are triples, grid nodes are two-ples.
                        lnStartDir.append(nNbr)

                lnEndDir = []
                lnNeighbours = list(G.neighbors(nEnd))
                for nNbr in lnNeighbours:
                    if len(nNbr)==3:  # rail nodes
                        lnEndDir.append(nNbr)

                print("agent:", iAgent, nStart, "poss starts:", lnStartDir, nEnd, "poss ends:", lnEndDir)
            else:
                print("Start / end not in graph:", nStart, nEnd)
        
            # iterate all the (start dir, end dir)s
            for nStartDir in lnStartDir[:]:
                for nEndDir in lnEndDir[:]:
                    lnPath=[]
                    try:
                        lnPath = nx.algorithms.shortest_path(G, source=nStartDir, target=nEndDir)
                    except nx.exception.NetworkXNoPath:
                        print("No path:", nStartDir, nEndDir)
                        continue
                        
                    Gpath = nx.induced_subgraph(G, lnPath)
                    lGpaths.append(Gpath)
                    llnPaths.append(lnPath)
                    lltStartTarg.append([nStartDir, nEndDir])
                    print("Path length:", len(lnPath))


def plotResourceUsage(G, llnPaths,
    llnAltPaths=None,
    nSteps=500, nStepsShow=200,
    contradir=False,
    nResources=50,
    figsize=(20,8), twostep=False, node_ticks=False, agent_increment=False, vmax=3,
    grid=True, cmap=None):

    # Infer the grid paths for the rail paths
    llnGridPaths = []
    for lnPath in llnPaths:
        lnGridPath = []
        for n in lnPath:
            # find the grid node for this rail node
            nGrid = [ n for n, d in G.pred[n].items() if d["type"]=="hold" ][0]
            lnGridPath.append(nGrid)
        llnGridPaths.append(lnGridPath)

    # the grid nodes, each with a vector of utilisation through time
    dResource = OrderedDict()

    # For each grid node, a list of direction nodes used
    dlRails = OrderedDict()    

    # For each grid node, a matrix of 4 x vectors of utilisation, one for each direction, indexed as above
    dg2Dirs = OrderedDict()

    for iPath, (lnGridPath, lnPath) in enumerate(list(zip(llnGridPaths, llnPaths))[:]):
        t=0
        
        # Create a resource for each grid node n
        # increment a counter for each step it is occupied by this agent
        for i in range(len(lnGridPath)):
            n = lnGridPath[i]
            nRail = lnPath[i]
            #print(i,n)
            if n not in dResource:
                dResource[n] = np.zeros(nSteps)
                dlRails[n] = [nRail]  # list of rail nodes used for entry
                dg2Dirs[n] = np.zeros((nSteps, 4)) # timesteps x "directions" (really, entry nodes)
                #print(n, nRail)
            
            # Ensure we have stored the rail node used for this grid node
            if nRail not in dlRails[n]:
                dlRails[n].append(nRail)
            iRail = dlRails[n].index(nRail)
            
            # node data for n, the grid node
            d = G.nodes()[n]
            if "l" in d:
                weight = d["l"]
            else:
                weight = 1
            
            g2Dirs = dg2Dirs[n]

            #print(iPath, n, weight, nRail, iRail)
            for t2 in range(t, t+weight):
                
                # increment the resource utilisation
                if agent_increment:
                    inc = iPath+1
                else:
                    inc = 1
                dResource[n][t2] += inc
                
                # Increment the direction utilisation
                g2Dirs[t2, iRail] += 1
            t += weight

    a2ResSteps = np.stack(list(dResource.values()))
    a3ResDirSteps = np.stack(list(dg2Dirs.values()))

    a2ResDirSteps = np.count_nonzero(a3ResDirSteps[:,:,:], axis=2)

    if cmap is None:
        cmap = cm.get_cmap("viridis")

    if not contradir:
        plt.figure(figsize=figsize)
        plt.imshow(a2ResSteps[:nResources,:nStepsShow], aspect="auto", vmax=vmax, cmap=cmap)
        plt.title("Rail Resource usage through time")

    else:
        plt.figure(figsize=figsize)
        plt.imshow(a2ResDirSteps[:nResources,:nStepsShow], aspect="auto", vmax=vmax, cmap=cmap)

        plt.xticks(range(0, nStepsShow, 5))    
        plt.title("Contra-Directional Rail Resource usage through time")

    plt.xlabel("Time steps into the future")
    plt.ylabel("Resource index")

    if grid: 
        plt.grid()
    plt.colorbar()

    if node_ticks:
        plt.yticks(range(len(dResource))[:nResources], labels=list(dResource.keys())[:nResources])

    return dResource, dlRails, dg2Dirs


def hammockPaths(G, nStart, nTarget, endSkip=0, preamble=True):
    lGpaths = []
    llnPaths = []
    # print("Start, End:", nStart, nEnd)

    # This is a generator of paths (shortest first)
    genPath = nx.algorithms.shortest_simple_paths(G, nStart, nTarget)
    # genPath = nx.algorithms.all_simple_paths(G, nStart, nEnd)

    # Walk along the shortest path
    for iPath, tPath in enumerate(genPath):
        # We only want the first ie shortest path
        if iPath >= 1: break

        # array of rail nodes in the path: node index x 3 coords (row, col, dir)
        g2Path = array(tPath)
        # plt.scatter(g2Path[:, 1], -g2Path[:, 0], label=str(len(tPath)))

        # Get the shortest path and record it in the lists
        Gpath = nx.induced_subgraph(G, tPath).copy()
        lGpaths.append(Gpath)
        llnPaths.append(tPath)

        for iStepStart in range(1, len(tPath) - endSkip):
            tPath2 = tPath[iStepStart:]
          
            # Start of the path at this point
            n = tPath2[0]
            
            # Check number of successors: If not a decision node, skip
            if len(G[n]) < 2:
                continue
            # print("Choices at step {} - {}".format(iStepStart, len(G[n])))
            
            # Find the "deviation" node - the successor node which is NOT on the shortest path
            # Next node in the shortest path
            n2 = tPath2[1]
            #print(n, n2, G[n])
            lSucc = set(G[n])  # the set of successor rail nodes
            lSucc.remove(n2)  # remove the next node in the shortest path
            nStart2 = lSucc.pop()  # retrieve the other choice - the deviation
            
            # Find the shortest path, having followed the deviation
            genPath2 = nx.algorithms.shortest_simple_paths(G, nStart2, nTarget, weight="l")

            if preamble:
                tPath3 = tPath[:iStepStart+1] + genPath2.__next__()
            else:
                tPath3 = [n] + genPath2.__next__()
            print(tPath3[0], tPath3[-1])
            Gpath = nx.induced_subgraph(G, tPath3)
            lGpaths.append(Gpath)
            
            llnPaths.append(tPath3)

    return llnPaths            

class RailEnvGraph(object):
    """
    Represent a RailEnv with a NetworkX DiGraph:

    Node types:
    - "grid" nodes, rows x cols, connected in a lattice / grid. eg (2,3)=row2, col3
    - "rail" nodes attached to grid nodes, one for each direction.

    Edge types:
    - "grid" edges between grid nodes, to give the grid structure
    - "hold" edges to hold a rail to a grid point,
        and to represent the resource occupied by an agent moving in any direction
    - "dir" edges (directional) between rail nodes

    So a RailEnv cell has a grid node showing its location, and two or more rail nodes
    representing the direction of entry, 
    eg (2,3,0) means row 2, col 3, entry direction north (ie from the south)

    An agent moves along "rail" edges between rail nodes, but occupies the whole grid node
    ie the whole complex of {grid node - hold edges - rail nodes}

    """

    def __init__(self, env):

        self.env = env

        # Create a grid of nodes matching (isomorphic to) the env rail grid
        # we use directed because we need directed edges to represent the agent/train direction
        self.G = nx.grid_2d_graph(*env.rail.grid.shape).to_directed()

        # give all these nodes and edges a type of grid.
        nx.set_node_attributes(self.G, name="type", values="grid")
        nx.set_edge_attributes(self.G, name="type", values="grid")

        self.add_entry_nodes()
        self.add_exit_edges()
        self.set_halts()
    
    def add_entry_nodes(self):
        """ Add a node for each inbound transition to a cell
        """
        for rowcol, trans in np.ndenumerate(self.env.rail.grid):
            # print(rowcol, type(rowcol), trans, G.node[rowcol])
            b44 = trans_int_to_4x4(trans)

            # for each inbound direction:
            for dirIn in range(4):
                # if we can enter in this direction (any exit)
                if b44[dirIn].any():  
                    # add a rail node for this entry, with the id (row, col, direction)
                    t3n_rail = (*rowcol, dirIn)
                    self.G.add_node(t3n_rail, type="rail")  

                    # add a "hold" edge to the grid node 
                    self.G.add_edge(rowcol, t3n_rail, type="hold") 

    def add_exit_edges(self):
        # a row,col vector for each direction NESW inbound
        gDirs = array([[-1,0], [0,1], [1,0], [0,-1]])

        # add edges to the direction nodes
        for rcIn, trans in np.ndenumerate(self.env.rail.grid):
            # print(rowcol, type(rowcol), trans, G.node[rowcol])
            if trans > 0:
                b44 = trans_int_to_4x4(trans)

                for dirIn in range(4):
                    for dirOut in range(4):
                        if b44[dirIn, dirOut]:
                            # get the rowcol of the destination cell
                            rcOut = tuple(array(rcIn) + gDirs[dirOut])
                            self.G.add_edge((*rcIn, dirIn), (*rcOut, dirOut), type="dir", l=1)

    def set_halts(self):
        stHalts = set()
        for agent in self.env.agents:
            if agent.initial_position is not None:
                stHalts.add(agent.initial_position)
            if agent.target is not None:
                stHalts.add(agent.target)
        
        for tHalt in stHalts:
            if tHalt in self.G:
                self.G.nodes()[tHalt]["halt"] = True

    def graph_rail_grid(self):
        """ returns a NX graph of rails only; includes:
            - grid nodes with rails
            - grid edges between grid nodes along rails (but not between adjacent rails)
            - rail nodes
            - hold edges
            - dir edges
            Excludes:
            - Grid nodes associated with empty rails
            - grid edges with empty grid nodes
        """
        G2 = nx.DiGraph()

        # Add the rail nodes and their direction edges.
        G2.add_nodes_from([(n, d) for n, d in self.G.nodes(data=True) 
            if d["type"] == "rail"])
        G2.add_edges_from([(u, v, d) for u, v, d in self.G.edges(data=True)
            if d["type"] == "dir"])

        # The "hold" edges are grid->rail
        # Copy the grid nodes connected to the rails, setting the type=grid
        G2.add_nodes_from([
                (u, self.G.nodes[u])  # node, attr dict
                for u, v, d in self.G.edges(data=True)
                if d["type"]=="hold"
            ])

        # Copy all the hold edges
        G2.add_edges_from([(u,v,d) for u, v, d in self.G.edges(data=True)
            if d["type"] == "hold"])

        # Include the grid edges which link the grid nodes (but not the grid links to no-rail grid nodes)
        # Add the grid edges for the grid nodes we have included:  u(grid) -- e(grid) -- v(grid)

        # This is no good because it also adds the edges between grid nodes which have rails.
        if False:
            G2.add_edges_from([
                (u, v, d) for u, v, d in self.G.edges(data=True) 
                if d["type"] == "grid" 
                and (u in G2)
                and (v in G2)])

        # Although we have a nested loop, mostly the inner loop will only execute once
        for nRail, dRail in G2.nodes(data=True):
            if dRail["type"]=="rail":
                # nGrid = [ nGrid for nGrid, d in G2.pred[nRail].items() if d["type"]=="hold" ][0]
                # alternatively, we simply remove the "direction" from the node id 3-tuple:
                nGrid = nRail[:2]
                
                # successors to rail nodes are always rail nodes joined by a dir edge
                # there may be more than 1
                for nRail2 in G2.succ[nRail]:
                    # Now get the grid node for the other rail node:
                    nGrid2 = nRail2[:2]
                    #print("add edge:", nGrid, nGrid2)
                    G2.add_edge(nGrid,nGrid2,type="grid")

        return G2

    def reduce_simple_paths(self):
        """
        Reduce *linear* paths, ie unbranched chains, into a single node, 
        preserving length in the edges "l" property
        """

        # Get the grid + rail nodes only, ie remove empty cells.
        G2 = self.graph_rail_grid()

        # G3 is just the grid nodes for cells with rails
        G3 = nx.induced_subgraph(G2, [ n for n, d in G2.nodes(data=True) if d["type"]=="grid" ])

        # It seems we need to copy a subgraph to really get rid of the unwanted nodes
        G3b = G3.copy()

        # G4 is a subgraph consisting only of the grid nodes with only 2 neighbours
        # (degree is 4 because undirected edges in a DiGraph are counted twice)
        lnGridSimple = [ n for n, d in G3b.nodes(data=True) if (G3b.degree[n]==4) and ("halt" not in d) ]
        G4 = G3b.subgraph(lnGridSimple)  # copies the nodes + data, and the (grid) edges + data.

        # Get the rail nodes (held by hold edges) and edges and their dir edges
        lnRails, lEdges = neighbors(G2, G4.nodes(), edge_types="hold", outer_edge_types="dir")
        G4d = G4.copy()

        # Add these into G4d
        #G4d.add_nodes_from(G2.subgraph(lnRails))  # doesn't copy data
        # lnRails just has the node ids; use subgraph and .nodes to pull the data from G2.
        G4d.add_nodes_from(list(G2.subgraph(lnRails).nodes(data=True)))
        G4d.add_edges_from(lEdges)

        G3c = G3b.copy()  # grid nodes 
        G4e = G4d.copy()  # the simple paths augmented with rails
        G5 = G2.copy()  # the full graph, mutable, so we can remove things:
        #print (G4.degree)

        # Iterate over all the linear paths (which are disconnected from each other in G4)
        for nSet in nx.components.strongly_connected_components(G4): # G4 is the grid simple paths
            #print("comp:", nSet)
            
            # Don't contract simple paths of length 1
            if len(nSet)==1:
                continue
            igComp = nx.induced_subgraph(G4,nSet)

            #print("deg:", igComp.degree)
            # The "inner" nodes (degree=4) excluding the ends (degree=2)
            lnInner = [ n for n,d in igComp.degree if d==4 ]
            #print("inner:", lnInner)
            
            #igCompRail = nx.induced_subgraph(G4d, )
            #lnInnerRails = [ n for n,d in nx.induce]
            
            # Find the ends of the chain of grid nodes by their degree of 2
            # (The undirected edges created in the grid are counted twice in this DiGraph)
            lnEnds = [ n for n,d in igComp.degree if d==2 ]
            #print("ends:", lnEnds)
            
            # Remove the inner Grid nodes
            G3c.remove_nodes_from(lnInner)

            # This removes the grid nodes but not the rail nodes...
            G5.remove_nodes_from(lnInner)
            
            # Find the start and end of the two rail paths (one in each direction)
            # corresponding to the grid path
            lnPathStart = []
            lnPathEnd = []
            
            # We now need to remove both directions of rail nodes.
            # First find the start and end of each rail chain.
            # look at each end of this grid path
            for grid_end in lnEnds:
                #grid_end = lnEnds[0] # look at the first end
                #print("grid_end", grid_end)

                # Look at all the adjacent nodes to this (grid) end
                for rail_end in G4e.adj[grid_end]: # look at the rail at this end
                    #print("rail_end", rail_end, G4e.adj[grid_end][rail_end])
                    if G4e.adj[grid_end][rail_end]["type"] == "hold":  # select rail, discard grid
                        #print(rail_end)
                        outedges = G4e.edges([rail_end])

                        # If it has 1 outedge, it's the start, otherwise (0) means it's the end
                        if len(outedges)==1:
                            #print("outedges", outedges)
                            lnPathStart.append(rail_end)
                        else:
                            lnPathEnd.append(rail_end)
            
            #print("pathStart:", lnPathStart)
            #print("pathEnd:", lnPathEnd)

            lnPathEnd.reverse()  # the ends are the opposite order to the starts...
            
            for nPathStart in lnPathStart:
                lnPath = get_simple_path(G4d, nPathStart)
                #print("lnPath", lnPath)
                if len(lnPath)>2:
                    # Remove the "inner" section of the path (if any)
                    G5.remove_nodes_from(lnPath[1:-1])
                # Join the start and end of the rail chain into a single node (lnPath[0])
                G5 = nx.minors.contracted_nodes(G5, lnPath[0], lnPath[-1], self_loops=False)

                # There should just be one outedge
                nNext = list(G5.successors(lnPath[0]))[0]
                # Record the length of the removed path
                G5.edges[lnPath[0], nNext]["l"] = len(lnInner)+2
            
            # Join up (identify, ie make identical) the ends of the grid chain into a single node
            G3c = nx.minors.contracted_nodes(G3c, *lnEnds, self_loops=False)
            G5 = nx.minors.contracted_nodes(G5, *lnEnds, self_loops=False)

            # Record the length of the simple path we have removed
            # This records it in the node which will be ignored in shortest path!
            # We use it for the length of the track
            G5.nodes()[lnEnds[0]]["l"] = len(lnInner)+2

            # We need to record it in an edge... but which edge?
            # There should just be one outedge
            nNext = list(G5.successors(lnEnds[0]))[0]
            G5.edges[lnEnds[0], nNext]["l"] = len(lnInner)+2

        return G5

    



    def savejson(self, filename="graph.json", bKeepId=True, alt_graph=None):
        if alt_graph is None:
            G = self.G
        else:
            G = alt_graph

        if bKeepId:
            # This version keeps the original id (row, col, dir)
            dNodeToIndex = { oNode:str(oNode) for iNode, oNode in enumerate(G.nodes()) }
        else:
            # d3 doesn't seem to like names like "(0, 1)" (stringified tuples) so use node indices
            # This is a dict from the tuple node id (row, col) to its (integer) index
            dNodeToIndex = { oNode:iNode for iNode, oNode in enumerate(G.nodes()) }

        ldNodes = [{# 'name': dNodeToIndex[oNode],
                    'id': dNodeToIndex[oNode],
                    #'title': ( int(iNode) for iNode in oNode ), # elements of tuple
                    'title': str(oNode), 
                    #"type": np.random.randint(2)
                    #"type": self.G.node[oNode].get("type")
                    "type": data["type"],
                } for oNode, data in G.nodes(data=True) ]
        
        
        ldLinks = [{'source': dNodeToIndex[u], 
                    'target': dNodeToIndex[v],
                    #"type": len(u[1]),
                    #"type": g.node[u[1]].get("type") # get the type of the node
                    "type": d["type"] # get the type of the edge
                }  for u,v,d in G.edges(data=True)]
        
        djG = {'nodes': ldNodes, 'links': ldLinks}
        #print(json.dumps(djG))
        with open(filename, 'w') as fOut:
            json.dump(djG, fOut, indent=4,)        
