import numpy as np
from utils import extract_edges, visualize_sample, preprocess, check_polygon_connectivity, check_polygon_intersection, iou_polygon_intersection, split_edge, slide_wall, remove_colinear_edges, valid_layout
from simanneal import Annealer
from intersections import doIntersect
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import copy
from scipy.spatial import ConvexHull, convex_hull_plot_2d

class FloorplanVectorization(Annealer):

    def __init__(self, state, conns):
        super(FloorplanVectorization, self).__init__(state, conns)  # important!
        self.conns = conns

    def move(self):
        # initial_energy = self.energy()
        new_polys = copy.deepcopy(self.state)

        # pick a polygon
        p_ind = np.random.choice(range(len(new_polys)))

        # move only if polygon is not empty
        _, es = new_polys[p_ind]
        if len(es) == 0:
            return
        
        # perform step
        new_polys = split_edge(new_polys, p_ind)
        new_polys = slide_wall(new_polys, p_ind)
        new_polys = remove_colinear_edges(new_polys, p_ind)

        # check if modified layout is valid
        if valid_layout(new_polys, p_ind) == False:
            return

        self.state = copy.deepcopy(new_polys)
        return 

    def energy(self):

        # number of self intersections
        polys = copy.deepcopy(self.state)
        n_intersec = 0
        all_edges = []
        # for p in polys:
        #     cs, es = p
        #     for e in es:
        #         x0, y0 = cs[e[0]]
        #         x1, y1 = cs[e[1]]
        #         all_edges.append(np.array((x0, y0, x1, y1))/255.0)

        # for k, e1 in enumerate(all_edges):
        #     for l, e2 in enumerate(all_edges):
        #         if k > l:
        #             x0, y0, x1, y1 = e1
        #             x2, y2, x3, y3 = e2
        #             if doIntersect(np.array([x0, y0]), np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3])):
        #                 n_intersec += 1

        # compute connectivity mistakes
        conn_mistakes = 0
        for c in self.conns:
            # check connectivity
            n1, val, n2 = c
            p1 = polys[n1]
            p2 = polys[n2]
            if (check_polygon_connectivity(p1, p2) == False) and (val == 1):
                conn_mistakes += 10
            elif (check_polygon_connectivity(p1, p2) == True) and (val == -1):
                conn_mistakes += 10

        # compute overlapping regions
        overlaps = 0
        for k, p1 in enumerate(polys):
            for l, p2 in enumerate(polys):
                if k > l:
                    overlaps += 100*iou_polygon_intersection(p1, p2)

        # compute area hull
        all_corners = [np.array(cs) for cs, _ in polys if len(cs) > 0]
        corner_penalty = 0
        for cs in all_corners:
            corner_penalty += len(cs)/10.0

        points = np.concatenate(all_corners, 0)
        hull = ConvexHull(points)
        points_hull = [(x, y) for x, y in zip(points[hull.vertices, 0], points[hull.vertices, 1])]
        hull_im = Image.new('L', (256, 256))
        dr = ImageDraw.Draw(hull_im)
        dr.polygon(points_hull, fill='white')
        hull_arr = np.array(hull_im)/255.0

        reg_im = Image.new('L', (256, 256))
        dr = ImageDraw.Draw(reg_im)
        for p in polys:
            if len(p[0]) > 2:
                points = [(x, y) for x, y in p[0]]
                dr.polygon(points, fill='white')
        reg_arr = np.array(reg_im)/255.0
        shape = 100*(hull_arr-reg_arr).sum()/reg_arr.sum()
        # plt.imshow(reg_im)
        # plt.show()

        # # dr.polygon(points_hull, fill='white')
        # # plt.imshow(hull_im)
        # # plt.show()


        print('e inter: {}, conn miss: {}, overlaps: {}, shape: {}, corners: {}'.format(n_intersec, conn_mistakes, overlaps, shape, corner_penalty))
        return float(n_intersec) + float(conn_mistakes) + float(overlaps) + float(shape) + float(corner_penalty)

if __name__ == '__main__':

    # load initial state
    types, polys, conns = np.load('/home/nelson/Workspace/autodesk/housegan2/raster/0_0.npy', allow_pickle=True)
    polys_raw = extract_edges(polys)
    polys = preprocess(polys_raw)

    # set params
    fp2vec = FloorplanVectorization(polys, conns)
    fp2vec.copy_strategy = "deepcopy"
    fp2vec.Tmax = 0.001
    fp2vec.Tmin = 0.0001
    fp2vec.steps = 200000
    # fp2vec.updates = 10  
    new_polys, e = fp2vec.anneal()

    # display output
    import matplotlib.pyplot as plt
    print('final energy: {}'.format(e))

    print('rectified')
    for cs, es in polys:
        print(cs)

    print('final')
    for cs, es in new_polys:
        print(cs)

    raw_im = visualize_sample(types, polys_raw, conns)
    before_im = visualize_sample(types, polys, conns)
    after_im = visualize_sample(types, new_polys, conns)

    plt.figure()
    plt.imshow(raw_im)

    plt.figure()
    plt.imshow(before_im)

    plt.figure()
    plt.imshow(after_im)

    plt.show()