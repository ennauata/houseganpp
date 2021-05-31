import os
os.environ['OPENBLAS_CORETYPE']='Haswell'

from pytorch_fid.fid_score import calculate_fid_given_paths
fid_value = calculate_fid_given_paths(['./FID/gt/', './FID/test/debug/'], 2, 'cpu', 2048)
print(fid_value)