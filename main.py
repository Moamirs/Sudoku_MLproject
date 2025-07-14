
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(img, title="Image"):
    plt.figure(figsize=(8, 8))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def preprocess_for_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    return dilated

def find_largest_sudoku_like_contour(thresh):
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)
            if 0.8 < ratio < 1.2 and area > max_area:
                max_area = area
                best_cnt = approx

    if best_cnt is None:
        raise Exception("جدول سودوکو پیدا نشد.")

    return best_cnt

def reorder_points(pts):
    pts = pts.reshape(4, 2)
    new_pts = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    new_pts[0] = pts[np.argmin(s)]     # top-left
    new_pts[2] = pts[np.argmax(s)]     # bottom-right
    new_pts[1] = pts[np.argmin(diff)]  # top-right
    new_pts[3] = pts[np.argmax(diff)]  # bottom-left

    return new_pts

def warp_perspective(image, points):
    rect = reorder_points(points)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def draw_red_box_on_original(image, contour):
    image_with_box = image.copy()
    cv2.drawContours(image_with_box, [contour], -1, (0, 0, 255), 10)
    return image_with_box

# ------------------ اجرای نهایی ------------------

def process_sudoku_image(image_path):
    image = cv2.imread(image_path)
    original = image.copy()

    thresh = preprocess_for_contours(image)
    contour = find_largest_sudoku_like_contour(thresh)

    image_with_box = draw_red_box_on_original(original, contour)
    warped = warp_perspective(original, contour)

    show_image(image_with_box, "Original with Red Box (All Types Supported)")
    show_image(warped, "Warped Sudoku Grid")

    return warped
sudoku = process_sudoku_image(r"C:\Users\masar\Desktop\Project\sample.webp")
