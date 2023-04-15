# Matala ex1_stars

## Q1
Write a simple and effective algorithm to match two pictures - one with hundreds of stars and the other with 10-20 stars. Choose the simplest existing method in this section.

Implemented a simple algorithm based on this article:
- Method: https://link.springer.com/article/10.1007/s40747-021-00619-z
- The paper proposes a new approach to star identification for spacecraft attitude determination, using spectral graph matching. The approach constructs a neighbor graph for each main star and uses rough search and graph matching to dynamically search for the most similar neighbor graph.
- Algorithm main functionality:
- For each image:
  - Load the image and convert it to grayscale.
  - Calculate the brightness threshold for the image.
  - Convert the image to black and white based on the brightness threshold.
  - Find contours in the image. Each contour is assumed to be a star.
  - For each contour, check its size, brightness, and position, and save it into a file. Filter out contours whose areas are less than or greater than the initial parameters.
  - Construct a neighbor graph for each star and build a minimum spanning tree (MST) from this graph. Node properties include star radius and brightness, and edge properties include distance.
  - Compare each pair of MSTs from two images using graph isomorphism.
  - Return a list of matching stars.

## Q2
Create a library that takes a star image and converts it to a file of coordinates x, y, r, b where x and y represent the coordinates of each star while r represents the radius and b represents the brightness.

- The library takes a list of images and creates a CSV file for each image containing the x, y, r, b parameters.

## Q3
Create a library that takes two images and calculates the best match between them by generating a list of coordinate pairs that point to the same star in each image.

- The library compare each two images and return a matching in case it detect two similar stars in both files. The results are saved into a result file.