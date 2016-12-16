# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:11:27 2016

@author: Cristina
"""

import cv2
import numpy as np

size = 200, 200
m = np.zeros(size, dtype=np.uint8) # ?
m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
p1 = (0,0)
p2 = (200, 200)
cv2.line(m, p1, p2, (0, 0, 255), 10)

img = cv2.imread('../../Databases/MIT_split/test/Opencountry/fie12.jpg');

cv2.namedWindow("draw", cv2.CV_WINDOW_AUTOSIZE)
while True:
    cv2.imshow("draw", img)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
cv2.destroyAllWindows()