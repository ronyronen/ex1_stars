import sys
from scan_photo import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n = len(sys.argv)
    if n < 2:
        print("Usage: main_3 <img name 1> <image name 2>")
        exit(1)

    im1_name = sys.argv[1]
    im2_name = sys.argv[2]

    print(f'Match: {im1_name} -> {im2_name}')

    scans = []
    scans.append(scan_image(im1_name, threshold=140, radius=700, s_min=5, s_max=250))
    scans.append(scan_image(im2_name, threshold=140, radius=700, s_min=5, s_max=250))

    matches = find_match(scans, (im1_name, im2_name), 2)

    # images = np.concatenate((orig_img, scan_img), axis=1)
    # cv2.imshow("img", images)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
