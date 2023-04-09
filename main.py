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
    image_list = list_images()
    # settings
    brightness = 145 # brightness
    radius = 700 # px
    s_min = 5 # px
    s_max = 300 # px
    dist_error = 30 # px
    print(f'Settings:\nbrightness: {brightness}\nradius: {radius}\ns_min: {s_min}\ns_max: {s_max}\ndist_error: {dist_error}')

    for i in range(len(image_list)):
        im1_name = image_list[i].strip('./')
        for j in range(i+1, len(image_list)):
            im2_name = image_list[j].strip('./')
            print(f'Match: {im1_name} -> {im2_name}')

            scans = []
            scans.append(scan_image(im1_name, threshold=brightness, radius=radius, s_min=s_min, s_max=s_max, rotate=False, draw_graph_flag=True))
            scans.append(scan_image(im2_name, threshold=brightness, radius=radius, s_min=s_min, s_max=s_max, rotate=True, draw_graph_flag=True))

            matches = find_match(scans, (im1_name, im2_name), dist_error)
            print(f'Results: {matches}')
