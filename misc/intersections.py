import numpy as np

# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are colinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p, q, r):

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    # for details of below formula.
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    if (val == 0): return 0 # 
    if val > 0:
        return 1
    else:
        return 2

# The main function that returns true if line segment 'p1q1'
# and 'p2q2' intersect.
def helperDoIntersect(p1, q1, p2, q2):

    # Find the four orientations needed for general and
    # special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2 and o3 != o4):
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 and onSegment(p1, p2, q1)): return True

    # p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 and onSegment(p1, q2, q1)): return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 and onSegment(p2, p1, q2)): return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 and onSegment(p2, q1, q2)): return True

    return False # Doesn't fall in any of the above cases

def scale_dimension(pt1, pt2, factor):
    base_length = pt2 - pt1
    ret1 = pt1 - (base_length * (factor-1) / 2)
    return ret1

def doIntersect(p1, q1, p2, q2):

    p1 = np.array(p1)
    q1 = np.array(q1)
    p2 = np.array(p2)
    q2 = np.array(q2)

    # same line segment - intersect
    if ((np.array_equal(p1, p2)) and (np.array_equal(q1, q2))) or\
        ((np.array_equal(p1, q2)) and (np.array_equal(q1, p2))):
            return True

    # line segment shares only one of the endpoints
    if (np.array_equal(p1, p2)):
        np1 = scale_dimension(p1, q1, .999)
        np2 = scale_dimension(p2, q2, .999)
        return helperDoIntersect(np1, q1, np2, q2)
    if (np.array_equal(p1, q2)):
        np1 = scale_dimension(p1, q1, .999)
        nq2 = scale_dimension(q2, p2, .999)
        return helperDoIntersect(np1, q1, p2, nq2)
    if (np.array_equal(q1, p2)):
        nq1 = scale_dimension(q1, p1, .999)
        np2 = scale_dimension(p2, q2, .999)
        return helperDoIntersect(p1, nq1, np2, q2)
    if (np.array_equal(q1, q2)):
        nq1 = scale_dimension(q1, p1, .999)
        nq2 = scale_dimension(q2, p2, .999)
        return helperDoIntersect(p1, nq1, p2, nq2)

    # do not share any endpoint
    return helperDoIntersect(p1, q1, p2, q2)

if __name__ == '__main__':

    p1 = (-10, -10)
    q1 = (10, 10)
    p2 = (-10, 10)
    q2 = (10, -10)

    print(doIntersect(p1, q1, p2, q2))

    p1 = (0, 0)
    q1 = (-10, -10)
    p2 = (-10, -10)
    q2 = (-10, -15)

    print(doIntersect(p1, q1, p2, q2))

    # p1 = (10, 0)
    # q1 = (0, 10)
    # p2 = (0, 0)
    # q2 = (10, 10)

    # print(doIntersect(p1, q1, p2, q2))

    # p1 = (10, 0)
    # q1 = (0, 10)
    # p2 = (0, 0)
    # q2 = (10, 10)

    # print(doIntersect(p1, q1, p2, q2))

    # p1 = (10, 0)
    # q1 = (0, 10)
    # p2 = (0, 0)
    # q2 = (10, 10)

    # print(doIntersect(p1, q1, p2, q2))
    # p1 = (-5, -5)
    # q1 = (0, 0)
    # p1 = np.array(p1)
    # q1 = np.array(q1)
    # # p2 = (0, 0)
    # # q2 = (10, 10);
    # p2 = (-5, -5)
    # q2 = (10, 10)
    # p2 = np.array(p2)
    # q2 = np.array(q2)
    # print(doIntersect(p1, q1, p2, q2))
    # print(scale_dimension(p1, q1, .999))
    # print(scale_dimension(p2, q2, 1.001))