# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
april = pd.read_csv('data/april-2017.csv')
# %%
april.head()
# %%


def idw_point(point, data, dist, p):
    points, weights = data
    dists = np.array([dist(point, p) for p in points])
    W = (1 / dists) ** p
    A = np.dot(weights, W)
    B = np.sum(W)
    return A / B


# %%
data = [
    np.array([
        [0, 1],
        [2, 1],
        [3, 1],
        [0, 4],
        [-3, 5],
    ]),
    np.array([
        12,
        3,
        4,
        2,
        16
    ])
]

# %%
point = np.array([1, 2])
idw_point(point, data, lambda x, y: np.linalg.norm(x - y), 2)
# %%
a = np.array([
    [1, 2],
    [2, 3]
])
for p in a:
    print(p)
# %%
pd.summary(april)

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
march = march.filter(regex='pm10', axis=1)
# %%
march.head()
# %%


def harvesine(lon1, lat1, lon2, lat2):
    rad = math.pi / 180  # degree to radian
    R = 6378.1  # earth average radius at equador (km)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat1 * rad) * \
        math.cos(lat2 * rad) * (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return(d)
