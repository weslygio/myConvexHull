import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from myConvexHull import ConvexHull
from itertools import cycle

data = datasets.load_wine()

# create a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = pd.DataFrame(data.target)

plt.figure(figsize=(10, 6))
colors = ['b','r','g']
plt.title('Nonflavanoid phenols vs Total phenols')
plt.xlabel(data.feature_names[7])
plt.ylabel(data.feature_names[5])

for i in range(len(data.target_names)):
    bucket = df[df['Target'] == i]
    bucket = bucket.iloc[:,[7,5]].values
    hull = ConvexHull(bucket)
    plt.scatter(bucket[:, 0], bucket[:, 1], label=data.target_names[i])

    hull_iter = cycle(hull)
    p1 = next(hull_iter)
    for _ in range(len(hull)):
        p2 = next(hull_iter)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], colors[i])
        p1 = p2

plt.legend()
plt.show()

plt.clf()

plt.figure(figsize=(10, 6))
plt.title('Alcohol vs Total phenols')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[5])

for i in range(len(data.target_names)):
    bucket = df[df['Target'] == i]
    bucket = bucket.iloc[:,[0,5]].values
    hull = ConvexHull(bucket)
    plt.scatter(bucket[:, 0], bucket[:, 1], label=data.target_names[i])

    hull_iter = cycle(hull)
    p1 = next(hull_iter)
    for _ in range(len(hull)):
        p2 = next(hull_iter)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], colors[i])
        p1 = p2

plt.legend()
plt.show()
