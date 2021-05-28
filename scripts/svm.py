import torch
from scripts.utils import read_sses, DrawDS, LabelledDS, draw_legend, draw_teacher
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from glob import glob
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.decomposition import PCA


def cf_labelled(batch):
    x, y = list(zip(*batch))
    x = torch.stack(x)
    y = torch.stack(y)
    x = x.view(-1, x.shape[2]*x.shape[3])
    y = y.view(-1)
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y

class SVM():
    def __init__(self, data_dir, labels_dir, batch_size, num_workers=10, label="all", shrink=0.0):
        self.classes = [Path(n).name for n in glob(data_dir + "/labelled/*")]
        _, labels = read_sses(labels_dir, (9999,9999), label=label)
        self.label = label
        self.shrink = shrink
        self.labels = labels
        self.labels_dir = labels_dir
        ## labelled data loader
        labelled = LabelledDS(data_dir + "/labelled/")
        train_indices, val_indices = train_test_split(list(range(len(labelled.dataset.targets))), test_size=0.2, stratify=labelled.dataset.targets)
        train_dataset = torch.utils.data.Subset(labelled, train_indices)
        val_dataset = torch.utils.data.Subset(labelled, val_indices)
        x, y = train_dataset[0]
        self.x_dim = x.shape[2]
        self.y_dim = len(self.classes)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=cf_labelled)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=cf_labelled)
        self.class_to_idx = self.train_loader.dataset.dataset.dataset.class_to_idx
        self.classes =  list(self.class_to_idx.values())

        self.model = PassiveAggressiveClassifier(C=1, n_jobs=10)
    
    def train(self, epochs):
        for epoch in range(epochs):
            for x, y in tqdm(self.train_loader):
                self.model.partial_fit(x, y, self.classes)
            acc = []
            for x, y in tqdm(self.val_loader):
                acc.append(self.model.score(x,y))
            print(epoch, ":", np.mean(acc))
        
    def draw(self, image_dir, out_path, kernel_size, batch_size):
        with Image.open(glob(image_dir+"/*")[0]) as img:
            w, h = img.size
        dataset = DrawDS(image_dir, kernel_size)
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
        pred_ys = []
        for x in tqdm(loader):
            x = np.asarray(x.view(x.shape[0], -1))
            y = self.model.predict(x)
            pred_ys.append(torch.tensor(y))
        
        seg_image = torch.cat(pred_ys).reshape([h,w]).numpy()
        cmap = plt.get_cmap("tab20", len(self.classes))
        plt.imsave(out_path, seg_image, cmap = cmap)
        return seg_image
    
    def draw_teacher(self, out_path, image_size):
        draw_teacher(out_path, self.labels_dir, self.class_to_idx, image_size, self.label, self.shrink)
    
    def draw_legend(self, out_path):
        draw_legend(out_path, self.labels_dir, self.class_to_idx, self.label)