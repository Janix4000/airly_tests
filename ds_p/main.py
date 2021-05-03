# %%
from skle
from geopy.distance import distance as geo_dist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
april = pd.read_csv('data/april-2017.csv')
# %%
april.head()
# %%


def idw_point(point, points, weights, dist, p):
    dists = np.apply_along_axis(lambda row: dist(point, row), 1, points)
    W = (1 / dists) ** p
    A = np.dot(weights, W)
    B = np.sum(W)
    return A / B


# %%
sensors = pd.read_csv('data/sensor_locations.csv')

# %%
sensors.head()
# %%
sensors[sensors.id == 3]
# %%
sensors_head = sensors.head().to_numpy()
# %%
pm10 = april.filter(items=['3_pm10, 140_pm10'])

# %%
pm10.head()
# %%
march = pd.read_csv('data/march-2017.csv')
# %%
march.head()
# %%
march = march.filter(regex='pm10', axis=1).to_numpy()
# %%


# %%
sensors_np = sensors.to_numpy()[:, [1, 2]]
# %%


def test_idw(points, weights):
    proper = ~np.isnan(weights)
    points = points[proper]
    weights = weights[proper]

# %%


test_idw(sensors_np, march[0, :])
# %%
