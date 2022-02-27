import numpy as np


def search_min(points: np.ndarray) -> np.ndarray:
    minpoint = points[0]
    for i in range(1, len(points)):
        if points[i, 0] < minpoint[0]:
            minpoint = points[i]
        elif points[i, 0] == minpoint[0]:
            if points[i, 1] < minpoint[1]:
                minpoint = points[i]
    return minpoint


def search_max(points: np.ndarray) -> np.ndarray:
    maxpoint = points[0]
    for i in range(1, len(points)):
        if points[i, 0] > maxpoint[0]:
            maxpoint = points[i]
        elif points[i, 0] == maxpoint[0]:
            if points[i, 1] > maxpoint[1]:
                maxpoint = points[i]
    return maxpoint


def dist_point_to_line(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """ Calculate distance from point p0 to a line segment from p1 to p2 """
    A = 0.5 * np.linalg.det([[p1[0], p1[1], 1],
                             [p2[0], p2[1], 1],
                             [p0[0], p0[1], 1]])
    b = np.linalg.norm(p2 - p1)
    h = 2 * A / b
    if abs(h) < 1e-5:
        return 0
    else:
        return h


def angle(P: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """ Calculate measure of ∠PQR in radian """
    QP = P - Q
    QR = R - Q
    return np.arccos(np.dot(QP, QR) / (np.linalg.norm(QP) * np.linalg.norm(QR)))


def ConvexHull(points: np.ndarray) -> np.ndarray:
    """
    Create convex hull from given points. Let p1, p2, ..., pn be the return set of points,
    then the visualization of the convex hull is p1-p2-...-pn-p1.

    :param points: a set of points which is to be made its convex hull
    :return: a set of ordered points which is the convex hull
    """
    minpoint = search_min(points)
    maxpoint = search_max(points)

    hull1 = ConvexHull2(points, minpoint, maxpoint)
    hull2 = ConvexHull2(points, maxpoint, minpoint)
    return np.append(hull1, hull2, axis=0)


def ConvexHull2(points: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Create convex hull from given points in which p1 and p2 are the initial extreme points.
    The resulting convex hull is the union of convex hull of outer points and p2.

    :param points: a set of points which is to be made its convex hull on the outer side
    :param p1: first extreme point
    :param p2: second extreme point
    :return: a set of ordered points which is the convex hull
    """
    subpoints: np.ndarray = np.array([]).reshape(0,2)
    p_extreme = p1
    max_dist = 0

    # Ambil point yang mengarah ke luar
    for testpoint in points:
        dist = dist_point_to_line(testpoint, p1, p2)
        if dist > 0:
            subpoints = np.append(subpoints, testpoint.reshape(1,2), axis=0)
            if dist > max_dist:
                max_dist = dist
                p_extreme = testpoint
            elif dist == max_dist:
                if angle(testpoint, p1, p2) > angle(p_extreme, p1, p2):
                    p_extreme = testpoint

    if subpoints.size == 0:
        return np.array([p2])

    hull1 = ConvexHull2(subpoints, p1, p_extreme)
    hull2 = ConvexHull2(subpoints, p_extreme, p2)

    return np.append(hull1, hull2, axis=0)
