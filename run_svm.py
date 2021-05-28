from scripts.svm import SVM
svm = SVM("data/train4/patches9x9", "data/train4/2010/labels", 200,  shrink = 5)
svm.train(30)
res_2010_svm = svm.draw("data/train4/2010/images_2010_R", "results/pac_9x9_ep26_R2_2010.png", (9,9), 5000)
res_2020_svm = svm.draw("data/train4/2020/images_2020_R", "results/pac_9x9_ep26_R2_2020.png", (9,9), 5000)

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pylab as pl

np.save("results/res2010_pac_9x9_ep30.npy", res_2010_svm)
np.save("results/res2020_pac_9x9_ep30.npy", res_2020_svm)

res2010 = np.load("results/res2010_cnn_9x9_R2.npy")
res2020 = np.load("results/res2020_cnn_9x9_R2.npy") 


img2010 = cv2.imread("data/pictures/mrd_085_eos_vis_20101023_1200.png")[:,:,0]
img2020 = cv2.imread("data/pictures/mrd_085_eos_vis_20201021_1205.png")[:,:,0]
mask = np.logical_and(img2010!=0, img2020!=0).astype(int)

diff = np.zeros(res2010.shape)
diff[np.logical_and(res2010!=4, res_2010_svm==4)] = 1 # ハイマツ以外 -> ハイマツ
diff[np.logical_and(res2010==4, res_2010_svm!=4)] = 2 # ハイマツ -> ハイマツ以外
cmap = plt.get_cmap("tab20", 3)
diff = diff * mask
plt.imsave("results/diff_haimatsu_cnn2pac.png", diff, cmap = cmap)

diff_labels = [
    "出現",
    "消滅"
]
pl.cla()
for i, l in zip(range(2), diff_labels):
    color = cmap(i+1)
    pl.plot(0, 0, "-", c = color, label = l, linewidth = 10)
pl.legend(loc = "center", prop = {"family": "MigMix 1P"})
pl.savefig("results/diff_legend.png")

diff = np.zeros(res2010.shape)
diff[np.logical_and(res2010!=0, res_2010_svm==0)] = 1 # ササ以外 -> ササ
diff[np.logical_and(res2010==0, res_2010_svm!=0)] = 2 # ササ -> ササ以外 
diff = diff * mask
plt.imsave("results/diff_sasa_cnn2pac.png", diff, cmap = cmap)