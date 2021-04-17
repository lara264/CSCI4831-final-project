import numpy as np
import cv2

# https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
points = []
def click(event, x, y, flags, params):
    if(event == cv2.EVENT_LBUTTONDOWN):
        print(x, "", y)
        points.append((x,y))
    if(event == cv2.EVENT_RBUTTONDOWN):
        print(x,y)
        points.append((x,y))

def get_rect():
    """ Getting rect """
    global points
    points = []
    print("Select the top left and bottom right corners of the person")
    cv2.imshow("image", image)
    cv2.setMouseCallback("image", click)
    cv2.waitKey(0)

    if(len(points) == 2):
        #print(points)
        return (points[0][0], points[0][1], points[1][0], points[1][1])

    raise Exception("Incorrect number of points selected")

def get_face_mask(mask):

    """ Getting person's face """
    global points
    points = []
    print("Select the top left and bottom right corners of the face")
    cv2.imshow("image", image)
    cv2.setMouseCallback("image", click)
    cv2.waitKey(0)

    if(len(points) == 2):
        #print(points)
        minX = points[0][0]
        maxX = points[1][0]
        minY = points[0][1]
        maxY = points[1][1]

        for i in range(minY, maxY):
            for j in range(minX, maxX):
                mask[i][j] = 1

        cv2.imshow("mask", mask)
        cv2.waitKey(0)

        return mask

    raise Exception("Incorrect number of points selected")

if __name__ == "__main__":

    # https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/
    image = cv2.imread("../People_Images/Person_1.png")
    mask = np.zeros(image.shape[:2], dtype="uint8")
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)

    rect = get_rect()
    known_foreground = get_face_mask(mask)

    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.grabCut(image, mask, rect, backgroundModel, foregroundModel, 10, cv2.GC_INIT_WITH_RECT)
    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    outputMask = (outputMask * 255).astype("uint8")
    output = cv2.bitwise_and(image, image, mask=outputMask)
    keep_me = cv2.bitwise_xor(image, output, mask=known_foreground)
    output = cv2.bitwise_or(output, keep_me)
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.destroyAllWindows()