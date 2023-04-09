import os

from scan_photo import *

def list_images(path="."):
    image_list = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            # print(os.path.join("/.", file))
            image_list.append(os.path.join(path, file))
    return image_list


def list_graphs(path="./output"):
    graph_list = []
    for file in os.listdir(path):
        if file.endswith(".edgelist"):
            # print(os.path.join("/.", file))
            graph_list.append(os.path.join(path, file))
    return graph_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_list  = list_images()
    for img_name in image_list:
        scan_img = scan_image(img_name, threshold=160,s_min=1,s_max=300)

    graph_list = list_graphs()
    matches = find_match(graph_list)
    draw_matches()


    # images = np.concatenate((orig_img, scan_img), axis=1)
    # cv2.imshow("img", images)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
