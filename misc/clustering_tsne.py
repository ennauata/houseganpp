import numpy as np
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import operator
from sklearn.cluster import KMeans
from models.models_exp_high_res import Autoencoder
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

autoencoderTest = Autoencoder()
autoencoderTest.load_state_dict(torch.load('./checkpoints/exp_autoencoder_A_72900_ae.pth'))
autoencoderTest = autoencoderTest.eval()
autoencoderTest.cuda()

# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse

def extract_features(mks, _types):

	mks_tensor = torch.tensor(mks).float().cuda().unsqueeze(1)
	mks_tensor[mks_tensor<=0] = 0.0
	mks_tensor[mks_tensor>0] = 1.0
	gen_rec, feat = autoencoderTest(mks_tensor)
	gen_rec = torch.sigmoid(gen_rec)
	
	inds = np.where(_types <= 10)[0]
	feat = feat[inds, :]
	feat = feat.view(-1)

	# ### DEBUG
	# all_images = []
	# for m1, m2 in zip(mks_tensor, gen_rec):
	# 	all_images.append(m1)
	# 	all_images.append(m2)
	# save_image(all_images, "./test.png", padding=2, pad_value=255, nrow=32, scale_each=True, normalize=True)

	return feat

def compute_cm(mks):
	cms = []
	for m in mks:
		yc, xc = np.mean(np.where(m>0), -1)
		cms.append((yc, xc))
	return cms

# list files
fp_files = glob.glob('./clustering_exp/floorplans_output/*.npy')

# get node with most conenctions
data = np.load(fp_files[0], allow_pickle=True).item()
nds, eds = data['nodes'], data['edges']
counts = defaultdict(int)
_types = np.where(nds==1)[1]
for e in eds:
	if e[1] > 0:
		n1, n2 = int(e[0]), int(e[2])
		if _types[n1] <= 10 and _types[n2] <= 10:
			counts[n1] += 1
			counts[n2] += 1
sorted_x = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
n_max = sorted_x[0]

# get centers
fp_info = []
for fname in fp_files:
	l = fname.split('/')[-1].split('_')[0]
	data = np.load(fname, allow_pickle=True).item()
	mks = data['masks']
	feat = extract_features(mks, _types)
	cms = compute_cm(mks)
	fp_info.append([int(l), cms, mks, feat])

X = np.array([feat.detach().cpu().numpy() for _, _, _, feat in fp_info])
labels = np.array([l for l, _, _, _ in fp_info])
X_embedded = TSNE(n_components=2, n_iter=5000).fit_transform(X)
# X_embedded = PCA(n_components=2).fit_transform(X)

# zip joins x and y coordinates in pairs
plt.figure()
colors = []
for x, y, l in zip(X_embedded[:, 0], X_embedded[:, 1] , labels):
	label = "{}".format(l)
	if l == 999:
		plt.annotate(label, # this is the text
			(x, y), # this is the point to label
			textcoords="offset points", # how to position the text
			xytext=(0,10), # distance from text to points (x,y)
			ha='center',
			color='red') # horizontal alignment can be left, right or center
		colors.append('red')

	elif l == 1000:
		plt.annotate(label, # this is the text
			(x, y), # this is the point to label
			textcoords="offset points", # how to position the text
			xytext=(0,10), # distance from text to points (x,y)
			ha='center',
			color='green') # horizontal alignment can be left, right or center		
		colors.append('green')

	else:
		plt.annotate(label, # this is the text
			(x, y), # this is the point to label
			textcoords="offset points", # how to position the text
			xytext=(0,10), # distance from text to points (x,y)
			ha='center',
			color='blue') # horizontal alignment can be left, right or center
		colors.append('blue')

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
plt.savefig('./tsne.png')

plt.figure()
sse = calculate_WSS(X_embedded, kmax=100)
plt.plot(sse)
plt.savefig('./sse.png')

plt.figure()
y_pred = KMeans(n_clusters=20).fit_predict(X_embedded)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pred)
plt.savefig('./clusters.png')