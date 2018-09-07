
# interesting code in python

import matplotlib.pyplot as plt
import numpy as np

x_axis_1 = [-1, 1, 1.5, 9, 0, .9, .6, -9]

np.savetxt("x_axis_1.csv", x_axis_1, delimiter=",", fmt='%s', header="x_axis_1")

plt.plot(x_axis_1)

plt.xlabel("x label of value list on x-axis")
plt.ylabel("y label of value list on y-axis")

# save graph in image file
#if graph saved after show(), image file will not show the graph

plt.savefig('a_simple_graph.png')

plt.show()





