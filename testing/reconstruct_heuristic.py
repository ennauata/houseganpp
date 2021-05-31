import numpy as np
from utils import extract_edges, visualize_sample, preprocess, check_polygon_connectivity, check_polygon_intersection, iou_polygon_intersection, split_edge, slide_wall, remove_colinear_edges, valid_layout
from utils import vectorize_heuristic, visualize_vector, draw_graph_with_types
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import torch

im_list = []

# initial polygon
for k in range(150, 160):
	for l in range(4):
		types, polys, conns = np.load('/home/nelson/Workspace/autodesk/housegan2/raster/{}_{}.npy'.format(k, l), allow_pickle=True)
		polys = extract_edges(polys)

		# add graph
		if l == 0:
			graph_arr = draw_graph_with_types(types, conns)
			im_list.append(torch.tensor(graph_arr.transpose((2, 0, 1)))/255.0) 

		# add images
		raw_arr = visualize_sample(types, polys)
		vec_arr = vectorize_heuristic(types, polys)

		im_list.append(torch.tensor(raw_arr.transpose((2, 0, 1)))/255.0) 
		im_list.append(torch.tensor(vec_arr.transpose((2, 0, 1)))/255.0) 

im_tensor = torch.stack(im_list)
save_image(im_tensor, 'out_img.png', nrow=9, padding=2)

