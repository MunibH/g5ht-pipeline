import tifffile as tif
import matplotlib.pyplot as plt
from skimage import morphology, measure
import numpy as np
import networkx as nx
import json
import tqdm
import sys
import os

r_dilation = 8
r_erosion = 32

def get_dilated_seg(seg, r=r_dilation):
    label_img = measure.label(seg)
    props = measure.regionprops(label_img)
    main_label = max(props, key=lambda p: p.area).label
    seg = (label_img == main_label)
    seg = morphology.isotropic_dilation(seg, r)
    return seg

def get_ske(seg, r=(r_erosion + r_dilation)):
    seg = morphology.isotropic_erosion(seg.copy(), r)
    ske = morphology.skeletonize(seg)
    return ske

def make_graph(ske):
    G = nx.Graph()
    rows, cols = [i.tolist() for i in np.where(ske)]
    for r, c in zip(rows, cols):
        G.add_node((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (r + dr, c + dc)
            if 0 <= neighbor[0] < ske.shape[0] and 0 <= neighbor[1] < ske.shape[1]:
                if ske[neighbor]:
                    G.add_edge((r, c), neighbor)
    return G

def get_largest(G):
    nodes_to_remove = [node for node, degree in G.degree() if degree > 2]
    G.remove_nodes_from(nodes_to_remove)
    largest_subgraph = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_subgraph)

def follow_nodes(G):
    start_node = next(node for node, degree in G.degree() if degree == 1)
    nodes_in_order = list(nx.bfs_edges(G, source=start_node))
    ordered_nodes = [start_node] + [v for u, v in nodes_in_order]
    return ordered_nodes

def visualize_frame(seg, nodes, spline_dilation=4):
    out = np.zeros(seg.shape, np.bool)
    for node in nodes:
        out[node] = True
    out = morphology.isotropic_dilation(out, spline_dilation)
    return np.logical_and(seg, np.logical_not(out))

def main():

    label_path = sys.argv[1]
    #reads label
    label = tif.imread(label_path)

    #initializes outputs
    dilated_label = np.zeros(label.shape, np.bool)
    visualization = np.zeros(label.shape, np.bool)
    spline_dict = {}

    #iterates through frames
    for i, frame in tqdm.tqdm(enumerate(label), total=len(label)):
        dilated_seg = get_dilated_seg(frame)
        ske = get_ske(dilated_seg)
        nodes = follow_nodes(get_largest(make_graph(ske)))

        #writes to outputs
        dilated_label[i] = dilated_seg
        visualization[i] = visualize_frame(dilated_seg, nodes)
        spline_dict[i] = nodes

    #saves outputs
    tif.imwrite('dilated.tif', dilated_label)
    tif.imwrite('visual.tif', visualization)
    with open('spline.json', 'w') as f:
        json.dump(spline_dict, f, indent=4)

