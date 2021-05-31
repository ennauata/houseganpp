import torch
import torch.nn as nn
import torch.nn.init as init
from models.models_exp_high_res import Generator
import onnx
import onnxruntime
import numpy as np
from viz import draw_graph, draw_masks
import matplotlib.pyplot as plt

# Initialize model
checkpoint = '../exp_random_types_attempt_3_A_500000_G.pth'
generator = Generator()
generator.load_state_dict(torch.load(checkpoint), strict=False)
generator = generator.eval()

# Convert by tracing input
z = torch.randn(16, 128, requires_grad=True)
given_masks = torch.full((16, 1, 64, 64), -1.0, requires_grad=True)
inds_masks = torch.zeros_like(given_masks)
given_masks_in = torch.cat([given_masks, inds_masks], 1) 

given_nds = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]],
       device='cpu')

given_eds = torch.tensor([[ 0, -1,  1],
        [ 0, -1,  2],
        [ 0, -1,  3],
        [ 0, -1,  4],
        [ 0, -1,  5],
        [ 0, -1,  6],
        [ 0,  1,  7],
        [ 0, -1,  8],
        [ 0, -1,  9],
        [ 0, -1, 10],
        [ 0,  1, 11],
        [ 0, -1, 12],
        [ 0, -1, 13],
        [ 0, -1, 14],
        [ 0, -1, 15],
        [ 1, -1,  2],
        [ 1, -1,  3],
        [ 1, -1,  4],
        [ 1, -1,  5],
        [ 1, -1,  6],
        [ 1,  1,  7],
        [ 1, -1,  8],
        [ 1, -1,  9],
        [ 1,  1, 10],
        [ 1, -1, 11],
        [ 1, -1, 12],
        [ 1, -1, 13],
        [ 1, -1, 14],
        [ 1, -1, 15],
        [ 2, -1,  3],
        [ 2, -1,  4],
        [ 2,  1,  5],
        [ 2, -1,  6],
        [ 2, -1,  7],
        [ 2,  1,  8],
        [ 2, -1,  9],
        [ 2, -1, 10],
        [ 2, -1, 11],
        [ 2, -1, 12],
        [ 2, -1, 13],
        [ 2, -1, 14],
        [ 2, -1, 15],
        [ 3, -1,  4],
        [ 3, -1,  5],
        [ 3, -1,  6],
        [ 3,  1,  7],
        [ 3, -1,  8],
        [ 3, -1,  9],
        [ 3, -1, 10],
        [ 3, -1, 11],
        [ 3,  1, 12],
        [ 3, -1, 13],
        [ 3, -1, 14],
        [ 3, -1, 15],
        [ 4, -1,  5],
        [ 4,  1,  6],
        [ 4, -1,  7],
        [ 4, -1,  8],
        [ 4,  1,  9],
        [ 4, -1, 10],
        [ 4, -1, 11],
        [ 4, -1, 12],
        [ 4, -1, 13],
        [ 4, -1, 14],
        [ 4, -1, 15],
        [ 5, -1,  6],
        [ 5,  1,  7],
        [ 5,  1,  8],
        [ 5, -1,  9],
        [ 5, -1, 10],
        [ 5, -1, 11],
        [ 5, -1, 12],
        [ 5,  1, 13],
        [ 5, -1, 14],
        [ 5, -1, 15],
        [ 6,  1,  7],
        [ 6, -1,  8],
        [ 6,  1,  9],
        [ 6, -1, 10],
        [ 6, -1, 11],
        [ 6, -1, 12],
        [ 6, -1, 13],
        [ 6,  1, 14],
        [ 6, -1, 15],
        [ 7, -1,  8],
        [ 7, -1,  9],
        [ 7,  1, 10],
        [ 7,  1, 11],
        [ 7,  1, 12],
        [ 7,  1, 13],
        [ 7,  1, 14],
        [ 7,  1, 15],
        [ 8, -1,  9],
        [ 8, -1, 10],
        [ 8, -1, 11],
        [ 8, -1, 12],
        [ 8, -1, 13],
        [ 8, -1, 14],
        [ 8, -1, 15],
        [ 9, -1, 10],
        [ 9, -1, 11],
        [ 9, -1, 12],
        [ 9, -1, 13],
        [ 9, -1, 14],
        [ 9, -1, 15],
        [10, -1, 11],
        [10, -1, 12],
        [10, -1, 13],
        [10, -1, 14],
        [10, -1, 15],
        [11, -1, 12],
        [11, -1, 13],
        [11, -1, 14],
        [11, -1, 15],
        [12, -1, 13],
        [12, -1, 14],
        [12, -1, 15],
        [13, -1, 14],
        [13, -1, 15],
        [14, -1, 15]])



# Export the model
torch.onnx.export(generator.float(),               # model being run
                  (z.float(), given_masks_in.float(), given_nds.float(), given_eds.long()), # model input (or a tuple for multiple inputs)
                  "houseganpp.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['z', 'given_masks_in', 'given_nds', 'given_eds'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'z' : {0 : 'n'},    # variable lenght axes
                 				'given_masks_in' : {0 : 'n'},
                 				'given_nds' : {0 : 'n'},
                                'given_eds' : {0 : 'm'},
                                'output' : {0 : 'n'}})

# # Checking onnx
# onnx_model = onnx.load("houseganpp.onnx")
# onnx.checker.check_model(onnx_model)

# ort_session = onnxruntime.InferenceSession("houseganpp.onnx")

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy().astype('float32') if tensor.requires_grad else tensor.cpu().numpy().astype('float32')

# # Run pytorch
# torch_out = generator(z, given_masks_in, given_nds, given_eds)

# # compute ONNX Runtime output prediction
# ort_inputs = {'z': to_numpy(z), 'given_masks_in': to_numpy(given_masks_in), 'given_nds': to_numpy(given_nds), 'given_eds': to_numpy(given_eds).long()}
# ort_outs = ort_session.run(None, ort_inputs)

# real_nodes = np.where(given_nds.detach().cpu()==1)[-1]
# real_edges = given_eds.detach().cpu().numpy()

# _, rgb_im = draw_graph((real_nodes, real_edges))
# fp_img = draw_masks(ort_outs[0], real_nodes, im_size=256)

# plt.imshow(rgb_im)
# plt.imshow(fp_img)
# plt.show()
