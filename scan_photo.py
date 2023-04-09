import os
from path import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import distance
import networkx as nx
import networkx.algorithms.isomorphism as iso

def load_image(file_name: str):
    """
    :param file_name:
    :return:  -> (int, int), double, double
    """
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


def scan_image(img_name, threshold=100, s_min=5, s_max=100):
    image = load_image(img_name)
    # make CSV file name from these params
    # time_stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    if not os.path.exists("./output"):
        os.mkdir("./output")
    image_name = img_name.strip("./").lower().split(".")[0]
    logfile = open(f'./output/{image_name}.csv', 'w+')
    logfile.write('i, x, y, r , b\n')

    # threshold
    th, th_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    # find contours
    contours = cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # filter by area
    i = 1
    p = []
    for c in contours:
        area = cv2.contourArea(c)
        if s_min < area < s_max:
            # cv2.drawContours(img, [c], -1, (255, 0, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            p.append((cx, cy))

            cv2.circle(th_img, (cx, cy), 40, (255, 0, 0), 3)
            cv2.putText(th_img, f'{i}', (cx - 20, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            logfile.write(f'{i}, {cx}, {cy}, {area}, {image[cy,cx] / 255}\n')
            i += 1

    create_mst(np.asarray(p), 750, image_name)

    # save images
    images = np.concatenate((image, th_img), axis=1)
    cv2.imwrite(f'./output/{image_name}_th.jpg', images)
    # return the threshold image
    return th_img


# def neighbors(data, image):
#     tree = cKDTree(data=data)
#     K = data.shape[0]
#     results = tree.query(data, k=K)
#     distances, indices = results
#     n = 0
#     index = indices[n]
#     neighbours = data[index]
#
#     # plt.clf()
#     # plt.scatter(data[1], data[0], color="red")
#     # plt.scatter(neighbours[:, 1], neighbours[:, 0], color="blue")
#     # plt.show()


def create_mst(data, radius=1000, image_name="_"):
    # distance matrix
    dist = distance.cdist(data, data, 'euclidean')
    # ignore diagonal values
    np.fill_diagonal(dist, np.nan)
    # extract i,j pairs where distance < threshold
    # paires = np.argwhere(dist <= radius)
    # groupby index
    # tmp = np.unique(paires[:, 0], return_index=True)
    # neighbors = np.split(paires[:, 1], tmp[1])[1:]
    # indices = tmp[0]
    G = nx.Graph()
    for ind in range(len(data)):
        paires = np.argwhere(dist <= radius)
        # G.add_nodes_from(indices)
        # G.add_edges_from(paires)
        vxs = list()
        edges = list()
        for vx in paires:
            if vx[0] != ind:
                continue
            vxs.append(vx[0]+1)
            edges.append((vx[0]+1, vx[1]+1, int(dist[vx[0], vx[1]])))
        G.add_nodes_from(vxs)
        G.add_weighted_edges_from(edges)
        T = nx.minimum_spanning_tree(G)
        sorted(T.edges(data=True))

        nx.write_weighted_edgelist(T, f'./output/T_{image_name}_{ind+1}.edgelist')

        edge_labels = nx.get_edge_attributes(G, "weight")
        # nx.draw(T, with_labels=True, font_weight='bold')
        pos = nx.spring_layout(T, seed=0)
        nx.draw(
            T, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9,
            labels={node: node for node in T.nodes()}
        )
        nx.draw_networkx_edge_labels(
            T, pos,
            edge_labels=edge_labels,
            font_color='red'
        )

        # nx.draw_networkx_edge_labels(T, pos, edge_labels)
        plt.savefig(f'./output/{image_name}_{ind+1}_T.jpg')
        G.clear()
        plt.clf()
    # plt.show()


def find_match(graph_list):
    T = {}
    for graph_name in graph_list:
        f_name = Path(graph_name).name.splitext()[0].strip()
        T[f_name] = nx.read_weighted_edgelist(graph_name)

    key1 = '3046'
    key2 = '3047'
    T1, T2 = {}, {}
    for key in T:
        if key1 in key:
            T1[key] = (T[key])
        else:
            T2[key] = (T[key])

    # em = iso.categorical_edge_match('weight', 'weight')
    em = lambda x,y: abs(x['weight'] - y['weight']) < 100
    matches = []
    for k1 in T1:
        for k2 in T2:
            if nx.is_isomorphic(T1[k1], T2[k2], edge_match=em):  # match weights
            # if nx.is_isomorphic(T1[k1], T2[k2]):
                matches.append((k1, k2))
                print(f'isomorphic: {k1}, {k2}')
    return matches


def draw_matches(matches):
    im1 = load_image('img_3047_th.jpg')
    im2 = load_image('img_3047_th.jpg')

def find_local_max(image):
    h = image.shape[0]
    w = image.shape[1]
    local_max = 0
    y1, x1 = 0, 0
    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            if image[y, x] >= local_max:
                local_max = image[y, x]
                y1, x1 = y, x
    return y1, x1
