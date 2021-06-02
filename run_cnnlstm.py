from scripts.utils import set_patches
from scripts.cnn_lstm import CNNLSTM
import tensorboardX
# set_patches("data/train4/2010/labels", "data/train4/2010/images_2010_R", "data/train4/patches9x9/", (9,9), batch_size=50, unlabelled=False, shrink=5)
# set_patches("data/train4/2010/labels", "data/train4/2020/images_2020_R", "data/train4/patches9x9/", (9,9), batch_size=50, unlabelled=False, shrink=5)

lstm = CNNLSTM("data/train4/patches9x9", "data/train4/2010/labels", (9,9), 200, device="cuda", shrink = 5)
lstm.draw_legend("results/legend.png")
lstm.draw_teacher("results/teacher_2010_shrink0.png", (5616,3744))

lstm.train(200, "cnn_lstm9x9")

res2010, _ = lstm.draw("data/train4/2010/images_2010_R", "results/cnn_9x9_R2_2010.png", (9,9), 50000)
res2020, _ = lstm.draw("data/train4/2020/images_2020_R", "results/cnn_9x9_R2_2020.png", (9,9), 50000)

lstm.draw_legend("legend.png")
lstm.draw_teacher("teacher_2015_full_shrink01.png", (5616,3744))

import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
import cv2
np.save("results/res2010_cnn_9x9_R3.npy", res2010)
np.save("results/res2020_cnn_9x9_R3.npy", res2020)

lstm.labels
lstm.class_to_idx

img2010 = cv2.imread("data/pictures/mrd_085_eos_vis_20101023_1200.png")[:,:,0]
img2020 = cv2.imread("data/pictures/mrd_085_eos_vis_20201021_1205.png")[:,:,0]
mask = np.logical_and(img2010!=0, img2020!=0).astype(int)

diff = np.zeros(res2010.shape)
diff[np.logical_and(res2010!=4, res2020==4)] = 1 # ハイマツ以外 -> ハイマツ
diff[np.logical_and(res2010==4, res2020!=4)] = 2 # ハイマツ -> ハイマツ以外
cmap = plt.get_cmap("tab20", 3)
diff = diff * mask
plt.imsave("results/diff_haimatsu_cnn_9x9_R2.png", diff, cmap = cmap)

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
diff[np.logical_and(res2010!=0, res2020==0)] = 1 # ササ以外 -> ササ
diff[np.logical_and(res2010==0, res2020!=0)] = 2 # ササ -> ササ以外 
diff = diff * mask
plt.imsave("results/diff_sasa_cnn_9x9_R2.png", diff, cmap = cmap)