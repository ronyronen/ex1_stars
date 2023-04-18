from scan_photo import *
import sys

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
    image_list = []
    if len(sys.argv) <= 1:
        image_list = list_images()
    else:
        image_list = sys.argv[1:]

    # settings
    brightness = 150 # brightness
    radius = 750 # px
    s_min = 1 # px
    s_max = 40 # px
    dist_error = 20 # px
    print(f'Settings:\nbrightness: {brightness}\nradius: {radius}\ns_min: {s_min}\ns_max: {s_max}\ndist_error: {dist_error}')

    if not os.path.exists("./output"):
        os.mkdir("./output")
    logfile = open(f'./output/results.csv', 'w+')
    logfile.write('im_1, im_2, s_1, s_2\n')

    for i in range(len(image_list)):
        im1_name = image_list[i].strip('./')
        scan1 = scan_image(im1_name, threshold=brightness, radius=radius, s_min=s_min, s_max=s_max, rotate=False,
                                draw_graph_flag=False)
        for j in range(i+1, len(image_list)):
            scans = []
            scans.append(scan1)
            im2_name = image_list[j].strip('./')
            print(f'Match: {im1_name} -> {im2_name}')
            scans.append(scan_image(im2_name, threshold=brightness, radius=radius, s_min=s_min, s_max=s_max, rotate=False, draw_graph_flag=False))

            # There are two function: alg1 and alg2.
            matches = find_match_alg1(scans, (im1_name, im2_name))
            # matches = find_match_alg2(scans, (im1_name, im2_name))
            for m in matches:
                s_1 = m.split('_')[-1]
                s_2 = matches[m][0].split('_')[-1]
                logfile.write(f'{im1_name}, {im2_name}, {s_1}, {s_2}\n')
            print(f'Results: {matches}')
