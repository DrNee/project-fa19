import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import random as rand

from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[2]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    #paramters
    k = max(int(len(adjacency_matrix) / 10), 4)
    #k = 4

    G, message = adjacency_matrix_to_graph(adjacency_matrix)   
    if message: return message
    
    list_of_homes = set(list_of_homes)
    TAHs = [] # indexes of TA homes
    for i, j in enumerate(list_of_locations):
        if len(TAHs) == len(list_of_homes):
            break
        elif j in list_of_homes:
            TAHs.append(i);
    start = list_of_locations.index(starting_car_location) # index of start
    D = nx.floyd_warshall_numpy(G, nodelist=range(len(list_of_locations)))
    clusters, dropoffs = getClusters(D, k, TAHs) # ta cluster: locations in that cluster, ta cluster: tas in that cluster
    centers = clustersToCentroids(clusters, D) # ta cluster center: ta cluster number
    route = planRoute(list(centers.keys()), D, start)
    while route[0] != start: route = rotate(route, 1) #get circuit to begin at start node
    
    predecessors, _ = nx.floyd_warshall_predecessor_and_distance(G)
    path = []
    for i in range(len(route)):
        if i+1 < len(route):
            path += nx.reconstruct_path(route[i], route[i+1], predecessors)
            del path[-1]
        else:
            path += nx.reconstruct_path(route[i], route[0], predecessors)
    out = (path, {cen: dropoffs[centers[cen]] for cen in centers.keys()})
    #plotSol(G, path, clusters)
    return out

    
def rotate(l, n):
    return l[n:] + l[:n]
    
def adjacencyToDistances(adjacency_matrix):
    Graph = nx.convert_matrix.from_numpy_matrix(np.matrix(adjacency_matrix))
    return nx.floyd_warshall_numpy(Graph, nodelist=range(adjacency_matrix.shape[0]))

def getClusters(D, k, TAHs):    
    distArray = ssd.squareform(D)
    lm = sch.linkage(distArray, method='complete')
    #return sch.fcluster(lm, 2, depth=10) 
    c = sch.fcluster(lm, t=D.max()/k, criterion='distance')
    #c = sch.fcluster(lm, k, criterion='maxclust')
    
    TAclusters = {c[t] for t in TAHs}
    num_clusters = max(c)
    clusters = {tac:[] for tac in TAclusters} # maps TA clusters to places in the in those clusters
    dropoffs = {tc:[ta for ta in TAHs if (c[ta]==tc)] for tc in TAclusters} # maps TA clusters to TAs who live in those clusters
    for p in range(len(c)):
        cluster = c[p]
        if cluster in TAclusters:
            clusters[cluster] += [p]
    return clusters, dropoffs

def clustersToCentroids(clusters, distances):
    centroids = dict()
    nodes = [i for i in range(0, distances.shape[0])]
    for key, cluster in clusters.items():
        centroid = min(nodes, key=lambda x: sum([distances.item((clusters[key][j], x)) for j in range(len(clusters[key]))]))
        centroids[centroid] = key
    return centroids # maps centers to cluster id
                
def planRoute(centers, D, start): # list of cluster center indices, distance matrix of G
    if start not in centers: centers.append(start) #add start to list of centers (we have to start there)
    T = np.empty((len(centers), len(centers)))
                
    for i in range(len(centers)):
        for j in range(len(centers)):
            T[i, j] = D[centers[i], centers[j]]
    
    path = nearest_neighbor(T).tolist()
    path = two_opt(T, path, verbose=False).tolist()
    return [centers[i] for i in path]

def plotSol(G, path, clusters):
    clusters = list(clusters.values())
    pos = nx.spring_layout(G)
    nx.draw(G,pos,node_color='k')
    # draw path in red
    path_edges = list(zip(path,path[1:]))
    nx.draw_networkx_nodes(G,pos,nodelist=path,node_color='r')
    # draw TA clusters in blue
    for cluster in clusters: 
        clr = "#{:02x}{:02x}{:02x}".format(rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        nx.draw_networkx_nodes(G,pos,nodelist=cluster,node_color='clr')
    nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',width=3)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.axis('off')
    plt.savefig('solution.png', bbox_inches='tight', format="png")

"""
======================================================================
Taken from https://github.com/nschloe/tspsolve/blob/master/tspsolve/
Credits to nschloe under MIT License
======================================================================
"""
def nearest_neighbor(d):
    """Classical greedy algorithm. (Start somewhere and always take the nearest item.)
    """
    n = d.shape[0]
    idx = np.arange(n)
    path = np.empty(n, dtype=int)
    mask = np.ones(n, dtype=bool)

    last_idx = 0
    path[0] = last_idx
    mask[last_idx] = False
    for k in range(1, n):
        last_idx = idx[mask][np.argmin(d[last_idx, mask])]
        path[k] = last_idx
        mask[last_idx] = False
    return path


def two_opt(d, path, verbose=False):
    """https://en.wikipedia.org/wiki/2-opt
    """
    path = np.array(path)

    edges = np.stack([path[:-1], path[1:]])
    min_path_cost = np.sum(d[tuple(edges)])
    n = d.shape[0]
    while True:
        found_new = False
        for i in range(n - 1):
            for k in range(i + 2, n + 1):
                new_path = np.concatenate([path[:i], path[i:k][::-1], path[k:]])
                edges = np.stack([new_path[:-1], new_path[1:]])
                path_cost = np.sum(d[tuple(edges)])
                if path_cost < min_path_cost:
                    if verbose:
                        print(
                            "Found better path ({} > {})".format(
                                min_path_cost, path_cost
                            )
                        )
                    path = new_path
                    min_path_cost = path_cost
                    # Go back to outmost loop
                    found_new = True
                    break
            if found_new:
                break
        if not found_new:
            break
    return path
                
"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__ =="__main_":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)