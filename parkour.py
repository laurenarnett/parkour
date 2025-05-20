#!/usr/bin/env python3

import cv2
import numpy as np

img = cv2.imread('/home/cameraftp/uploads/2025/05/20/Reolink Argus 3 Pro 1_00_20250520180259.jpg')
edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
cv2.imwrite('/home/cameraftp/uploads/2025/05/20/output_edges.jpg', edges)
