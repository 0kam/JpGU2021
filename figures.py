import numpy as np
from PIL import Image
from glob import glob
# Animation of images 

images = [Image.open(i).convert("P", palette=Image.ADAPTIVE, colors=256) for i in sorted(glob("data/train2/2010/images_2010/*"))]
images = [i.resize((800,600), Image.BICUBIC) for i in images]
images[0].save("results/animation_2010.gif", save_all=True, append_images=images[1:],  duration=300, loop=0)


images = [Image.open(i).convert("P", palette=Image.ADAPTIVE, colors=256) for i in sorted(glob("data/train2/2020/images_2020/*"))]
images = [i.resize((800,600), Image.BICUBIC) for i in images]
images[0].save("results/animation_2020.gif", save_all=True, append_images=images[1:],  duration=300, loop=0)

# Animation of patches
def animate_patches(in_path, out_path):
    patch = np.load(in_path).reshape(-1, 16, 3, 9, 9)[25,:,:,:,:].transpose(0,2,3,1)
    patch = [patch[i,:,:,:] for i in range(patch.shape[0])]
    patch = [Image.fromarray(i).convert("P", palette=Image.ADAPTIVE, colors=256) for i in patch]
    patch = [i.resize((200,200), Image.NEAREST) for i in patch]
    patch[0].save(out_path, save_all=True, append_images=patch[1:],  duration=300, loop=0)
animate_patches("data/train2/patches9x9/labelled/1/images_2010_31238.npy", "results/animation_sasa.gif")
animate_patches("data/train2/patches9x9/labelled/2/images_2010_21185.npy", "results/animation_rachi.gif")
animate_patches("data/train2/patches9x9/labelled/3/images_2010_22678.npy", "results/animation_sonota.gif")
animate_patches("data/train2/patches9x9/labelled/4/images_2010_10215.npy", "results/animation_sora.gif")
animate_patches("data/train2/patches9x9/labelled/5/images_2010_14997.npy", "results/animation_haimatsu.gif")

# Count changed numbers
## haimatsu
diff_h = np.asarray(Image.open("results/diff_haimatsu_cnn_9x9.png"))[1488:,:,:]
h_increase = diff_h[diff_h[:,:,0]==140].shape[0]
h_decrease = diff_h[diff_h[:,:,0]==158].shape[0]

## sasa
diff_s = np.asarray(Image.open("results/diff_sasa_cnn_9x9.png"))
s_increase = diff_s[diff_s[:,:,0]==140].shape[0]
s_decrease = diff_s[diff_s[:,:,0]==158].shape[0]

# Count Pixels
res2010 = np.asarray(Image.open("results/cnn_9x9_2010.png"))
res2020 = np.asarray(Image.open("results/cnn_9x9_2020.png"))
h_2010 = res2010[res2010[:,:,0]==158].shape[0]
h_2020 = res2020[res2020[:,:,0]==158].shape[0]

h_2010_rm_top = res2010[1488:,:,:][res2010[1488:,:,0]==158].shape[0] # 上部を除く
h_2020_rm_top = res2020[1488:,:,:][res2020[1488:,:,0]==158].shape[0] # 上部を除く

s_2010 = res2010[res2010[:,:,0]==31].shape[0]
s_2020 = res2020[res2020[:,:,0]==31].shape[0]