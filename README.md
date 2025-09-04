
# Task 11.3 — Shape and Color Detection

## Run
1) Install
   pip install opencv-python numpy

2) Execute
   python shapes.py test.jpg
   Output saved as result_shapes.png

## Method
- Preprocess
  grayscale, blur, Canny, dilate
- Contours
  find external contours
- Shape
  approxPolyDP, vertex count, side ratio, circularity
- Color
  HSV thresholds, majority inside the contour

## Tune
- thresholds in shapes.py
- min area in classify_shape
- HSV ranges for your lighting
