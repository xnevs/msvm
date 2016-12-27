#!/usr/bin/python

import sys
import numpy as np

num = int(sys.argv[1])

w = np.array([-1,1])
b = 0

m = 1

points = []
while len(points) < num:
    p = 4 * (np.random.rand(2) - 0.5)
    x = np.dot(w,p) + b
    if x >= m:
        points.append(p.tolist() + [1])
    elif x <= -m: 
        points.append(p.tolist() + [-1])

for p in points:
    print(','.join(map(str,p)))
