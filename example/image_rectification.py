import cv2

from dscamera import DSCamera

if __name__ == "__main__":
    json_file = "./calibration.json"
    cam = DSCamera(json_file)
    img = cv2.imread("./sample.jpg")
    perspective = cam.to_perspective(img)
    equirect = cam.to_equirect(img)

    # Display
    cv2.imshow("orig", img)
    cv2.imshow("perspective", perspective)
    cv2.imshow("equirect", equirect)
    cv2.waitKey()
    cv2.destroyAllWindows()
