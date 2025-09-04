import cv2
import numpy as np
import sys

# HSV color ranges
COLOR_RANGES = {
    "black":  ([0, 0, 0], [180, 255, 30]),
    "white":  ([0, 0, 200], [180, 30, 255]),
    "blue":   ([100, 150, 50], [130, 255, 255]),
    "yellow": ([15, 40, 100], [35, 255, 255]),
    "green":  ([40, 100, 100], [80, 255, 255]),
    "red":    [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
}

def color_detection(hsv_img, lower_bound, upper_bound, mask):
    lower = np.array(lower_bound, dtype=np.uint8)
    upper = np.array(upper_bound, dtype=np.uint8)
    mask_color = cv2.inRange(hsv_img, lower, upper)
    mask_color = cv2.bitwise_and(mask_color, mask_color, mask=mask)
    return np.sum(mask_color)

def detect_color(hsv_img, mask):
    best_color = "unknown"
    best_value = 0
    total_area = np.sum(mask) / 255
    if total_area < 100:
        return best_color
    for name, bounds in COLOR_RANGES.items():
        if name == "red":
            value = 0
            for lo, hi in bounds:
                value += color_detection(hsv_img, lo, hi, mask)
        else:
            lo, hi = bounds
            value = color_detection(hsv_img, lo, hi, mask)
        threshold = total_area * (0.1 * 255 if name == "yellow" else 0.2 * 255)
        if value > best_value and value > threshold:
            best_value = value
            best_color = name
    return best_color

def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return None, None
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    area = cv2.contourArea(contour)
    if area < 100:
        return None, None
    v = len(approx)
    if v == 3:
        return "Triangle", approx
    if v == 4:
        pts = approx.reshape(-1, 2).astype(np.float32)
        d01 = np.linalg.norm(pts[0] - pts[1])
        d12 = np.linalg.norm(pts[1] - pts[2])
        ratio = d01 / d12 if d12 != 0 else 0
        if 0.85 <= ratio <= 1.15:
            return "Square", approx
        return "Rectangle", approx
    if v > 4:
        _, radius = cv2.minEnclosingCircle(contour)
        if radius > 0:
            circ_area = np.pi * radius * radius
            circularity = float(area) / float(circ_area)
            if circularity > 0.7:
                return "Circle", approx
    return None, None

def annotate(img, cnt, shape_label, color_label):
    # bounding box of the shape
    x, y, w, h = cv2.boundingRect(cnt)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    # labels
    shape_text = shape_label
    color_text = color_label

    # text sizes
    (stw, sth), _ = cv2.getTextSize(shape_text, font, scale, thickness)
    (ctw, cth), _ = cv2.getTextSize(color_text, font, scale, thickness)

    # shape text above the shape
    shape_x = x + w // 2 - stw // 2
    shape_y = y - 10 if y - 10 > sth else y + sth + 5

    # color text inside the shape (centered)
    center_x = x + w // 2 - ctw // 2
    center_y = y + h // 2 + cth // 2

    # background for better visibility
    cv2.rectangle(img, (shape_x - 5, shape_y - sth - 5),
                  (shape_x + stw + 5, shape_y + 5), (0, 0, 0), -1)
    cv2.putText(img, shape_text, (shape_x, shape_y),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.rectangle(img, (center_x - 5, center_y - cth - 5),
                  (center_x + ctw + 5, center_y + 5), (0, 0, 0), -1)
    cv2.putText(img, color_text, (center_x, center_y),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def process(image_path, save_path="result_shapes.png", show=True):
    # load image
    img = cv2.imread(image_path)
    if img is None:
        print("image not found")
        return
    out = img.copy()

    # preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    # find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # shape and color detection
    for c in cnts:
        label, _ = classify_shape(c)
        if not label:
            continue
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        color = detect_color(hsv, mask)
        if color != "unknown":
            annotate(out, c, label, color)
            cv2.drawContours(out, [c], -1, (0, 0, 0), 2)
            cv2.drawContours(out, [c], -1, (255, 255, 255), 1)

    # save result
    cv2.imwrite(save_path, out)
    print(f"saved to {save_path}")

    # show window
    if show:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    process(path)
