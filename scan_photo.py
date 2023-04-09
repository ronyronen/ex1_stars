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


def rotate_image(image):
    return cv2.rotate(image, cv2.ROTATE_180)


def scan_image(img_name, threshold=100, radius=750, s_min=5, s_max=100, rotate=False):
    image = load_image(img_name)
    if rotate:
        image = rotate_image(image)
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

    # save images
    images = np.concatenate((image, th_img), axis=1)
    cv2.imwrite(f'./output/{image_name}_th.jpg', images)

    trees = create_mst(np.asarray(p), radius, image_name)

    # return the threshold image
    return trees, th_img, p


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
    trees = {}
    for ind in range(len(data)):
        G.clear()
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
        trees[f'{image_name}_{ind+1}'] = T

        # nx.write_weighted_edgelist(T, f'./output/T_{image_name}_{ind+1}.edgelist')
        #
        # edge_labels = nx.get_edge_attributes(G, "weight")
        # # nx.draw(T, with_labels=True, font_weight='bold')
        # pos = nx.spring_layout(T, seed=0)
        # nx.draw(
        #     T, pos, edge_color='black', width=1, linewidths=1,
        #     node_size=500, node_color='pink', alpha=0.9,
        #     labels={node: node for node in T.nodes()}
        # )
        # nx.draw_networkx_edge_labels(
        #     T, pos,
        #     edge_labels=edge_labels,
        #     font_color='red'
        # )
        #
        # # nx.draw_networkx_edge_labels(T, pos, edge_labels)
        # plt.savefig(f'./output/{image_name}_{ind+1}_T.jpg')
        # plt.clf()
    # plt.show()
    return trees


def find_match(scans, names, threshold=100):
    T1 = scans[0][0]
    T2 = scans[1][0]
    # em = iso.categorical_edge_match('weight', 'weight')
    em = lambda x,y: abs(x['weight'] - y['weight']) < threshold
    # nm = lambda x, y: abs(x['weight'] - y['weight']) < threshold
    matches = []
    for k1 in T1:
        for k2 in T2:
            # GM = iso.GraphMatcher(T1[k1], T2[k2], edge_match=em)
            # if GM.subgraph_is_isomorphic():
            #     # draw_graph(T1[k1], names)
            #     # draw_graph(T2[k2], names)
            #     # plt.show()
            #     # print((k1, k2))
            if nx.is_isomorphic(T1[k1], T2[k2], edge_match=em):  # match weights
            # if nx.is_isomorphic(T1[k1], T2[k2]):
                matches.append((k1, k2))
            else:
                if T1[k1].number_of_nodes() == T2[k2].number_of_nodes() + 1:
                    for n in T1[k1].nodes:
                        T_tmp = nx.Graph(T1[k1])
                        T_tmp.remove_node(n)
                        if nx.is_isomorphic(T2[k2], T_tmp, edge_match=em):  # match weights
                            # if nx.is_isomorphic(T1[k1], T2[k2]):
                            matches.append((k1, k2))
                            break
                elif T2[k2].number_of_nodes() == T1[k1].number_of_nodes() + 1:
                    for n in T2[k2].nodes:
                        T_tmp = nx.Graph(T2[k2])
                        T_tmp.remove_node(n)
                        if nx.is_isomorphic(T1[k1], T_tmp, edge_match=em):  # match weights
                            # if nx.is_isomorphic(T1[k1], T2[k2]):
                            matches.append((k1, k2))
                            break
            # print(f'isomorphic: {k1}, {k2}')

    th_mg1 = scans[0][1]
    th_mg1 = cv2.cvtColor(th_mg1, cv2.COLOR_GRAY2RGB)
    th_mg2 = scans[1][1]
    th_mg2 = cv2.cvtColor(th_mg2, cv2.COLOR_GRAY2RGB)
    p1 = scans[0][2]
    p2 = scans[1][2]

    cv2.putText(th_mg1, names[0],  (100,100) , cv2.FONT_HERSHEY_SIMPLEX, 4, color=(0,0,255), thickness=3)
    cv2.putText(th_mg2, names[1],  (100,100) , cv2.FONT_HERSHEY_SIMPLEX, 4, color=(0,0,255), thickness=3)
    w = 75
    for m in matches:
        vx1 = m[0].split('_')[-1]
        vx2 = m[1].split('_')[-1]
        px1 = p1[int(vx1)-1]
        px2 = p2[int(vx2)-1]
        color = np.random.choice(range(256), size=3).tolist()
        cv2.rectangle(th_mg1, (px1[0] - w, px1[1]- w), (px1[0] + w, px1[1] + w), color=color, thickness=4)
        cv2.rectangle(th_mg2, (px2[0]- w, px2[1]- w), (px2[0] + w, px2[1] + w), color=color, thickness=4)
        cv2.putText(th_mg1, f'{vx2}', (px1[0] + 100, px1[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, color=color, thickness=3)
        cv2.putText(th_mg2, f'{vx1}', (px2[0] + 100, px2[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, color=color, thickness=3)

    white = [255, 255, 255]
    th_mg1 = cv2.copyMakeBorder(th_mg1, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=white)
    th_mg2 = cv2.copyMakeBorder(th_mg2, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=white)
    images = np.concatenate((th_mg1, th_mg2), axis=1)
    nm1 = names[0].split('.')[0]
    nm2 = names[1].split('.')[0]
    cv2.imwrite(f'./output/matches_{nm1}_{nm2}.jpg', images)

    return matches

def draw_graph(T:nx.Graph, image_names):
    edge_labels = nx.get_edge_attributes(T, "weight")
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

    plt.savefig(f'./output/T_{image_names[0]}_{image_names[1]}')

    # nx.draw_networkx_edge_labels(T, pos, edge_labels)
    # plt.clf()


