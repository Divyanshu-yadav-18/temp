import struct
import matplotlib.pyplot as plt

IMAGE_FILE = "train-images.idx3-ubyte"
COMPONENTS = 14
IMAGE_INDEX = 0

def read_one_image(filename, index):
    f = open(filename, "rb")
    magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
    f.seek(16+index*rows*cols)
    image = []

