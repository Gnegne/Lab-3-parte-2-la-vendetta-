import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from lab import *
from iandons import *

# setup

camera_to_ruler = .568
bulb_radius = .05
bulb_thickness = .001
centre_to_ruler = .11

# contis

n = 1.5
d = bulb_thickness
r1 = -bulb_radius
r2 = -bulb_radius - d
p = (n-1) * (1/r1 - 1/r2 + (n-1)*d/(n * r1 * r2))
f = 1/p

vo = 1/bulb_radius
vi = p - vo
di = 1/vi

M = f/(f - bulb_radius)
img_from_camera = camera_to_ruler - centre_to_ruler - bulb_radius - di
k = img_from_camera / camera_to_ruler

print(M, k, img_from_camera, di)
