from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms
import pandas as pd
import os.path as path
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
from torch.utils.data import Dataset, DataLoader
import os 
from sklearn.model_selection import train_test_split

min_max_scaler = MaxAbsScaler()

kwargs = {"shuffle": True, "num_workers": 16, "pin_memory": True}

def ImageLoader(dataname='celebA'):
    if dataname == 'celebA':
        root_dir = '/data/celebA/CelebA'

        data = pd.read_csv(os.path.join(root_dir, 'Anno/list_attr_celeba.csv'))
        split_data = pd.read_csv(os.path.join(root_dir, 'Anno/list_eval_partition.csv'))
        
        train_data = data[split_data['partition'] == 0]
        valid_data = data[split_data['partition'] == 1]
        test_data = data[split_data['partition'] == 2]
        
        return train_data.reset_index(drop=True), valid_data.reset_index(drop=True), test_data.reset_index(drop=True)

    if dataname == 'celebahq':
        data = pd.read_csv('/data/celeba-hq/hq_to_small.csv')
        data['idx'] = data['idx'].apply(lambda x : '/data/celeba-hq/CelebA-HQ/combined/imgHQ' + str(x).zfill(5) + '.npy')
            
        train_data, test_data = train_test_split(data, random_state = 2021, test_size = 0.9)
        
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)



class ImageDataset(Dataset):
    def __init__(self, data, dataname, sens_idx, label_idx, root_dir = None, transform = None):
        self.transform = transform
        self.data = data
        self.dataname = dataname
        
        if self.dataname == 'celeba':
            self.sens_idx = sens_idx
            self.label_idx = label_idx
            self.root_dir = root_dir
        elif self.dataname == 'celebahq':
            self.label_idx = data.columns.get_loc(label_idx)
            self.sens_idx = data.columns.get_loc(sens_idx)      
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.data.iloc[idx, 0]
        
        if self.dataname == 'celeba':
            img_name = os.path.join(self.root_dir, img_name)

            image = Image.open(img_name)
            sens = self.data[self.sens_idx][idx]
            label = self.data[self.label_idx][idx]

        elif self.dataname == 'celebahq':
            image = np.load(img_name)
            image = image.reshape(3, 1024, 1024)
            image = np.transpose(image, (1, 2, 0))
            image = Image.fromarray(image)
            
            sens = np.array(self.data.iloc[idx][self.sens_idx], dtype = float)
            label = np.array(self.data.iloc[idx][self.label_idx], dtype = float)
            name = self.data['idx'][idx].split('/')[-1].split('.')[0]           
            
        if self.transform:
            image = self.transform(image)

        return  image, max(int(sens), 0), max(int(label), 0)
    

class digital(data.Dataset):
    # with size in 32x32
    def __init__(self, subset, transform=None):
        file_dir = "./data/{}.txt".format(subset)
        self.data_dir = open(file_dir).readlines()
        self.transform = transform

    def __getitem__(self, index):
        img_dir, label = self.data_dir[index].split()
        img = Image.open(img_dir)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(np.int64(label)).long()

        return img, label

    def __len__(self):
        return len(self.data_dir)


def get_digital(args, subset, reverse = False):
    transform = transforms.Compose([
        transforms.Resize(32), transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    
    if reverse:
        transform = transforms.Compose([
        transforms.Resize(32), transforms.ToTensor(),
                                    transforms.Normalize((0.8625,), (0.2935,))
                                    ])
   
    data = digital(subset, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=args.bs,
        **kwargs
    )

    return data_loader


def mnist_usps(args):
    train_0 = get_digital(args, "train_mnist")
    train_1 = get_digital(args, "train_usps")
    train_data = [train_0, train_1]

    return train_data


def mnist_reverse(args):
    train_0 = get_digital(args, "train_mnist")
    train_1 = get_digital(args, "train_reverse_mnist", reverse = True)
    train_data = [train_0, train_1]

    return train_data


class FaceLandmarksDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file, sep=" ", header=None,
                                           names=["#image path", "#x1","#x2","#x3","#x4","#x5","#y1","#y2","#y3"
                                               ,"#y4","#y5","#gender"," #smile", "#wearing glasses", "#head pose"])
        
        self.transform = transforms.Compose(
                       [transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((89.93/255, 99.5/255, 119.78/255), (1., 1., 1.))])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_dir, sens, label = path.join('./data/MTFL/', self.data['#image path'][index].replace('\\', '/')), self.data['#wearing glasses'][index], self.data['#gender'][index]
        img = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        label = torch.tensor(np.int64(label)).long()
        sens = torch.tensor(np.int64(sens)).long()
        return img, sens, label

from sklearn.preprocessing import normalize
from torch.utils import data

class TabDataset(data.Dataset):
    def __init__(self, dataset, sens_idx):
        self.label = dataset.labels.squeeze(-1).astype(int)
        
        self.feature_size = dataset.features.shape[1]
        sens_loc = np.zeros(self.feature_size).astype(bool)
        if isinstance(sens_idx, list):
            for sens in sens_idx:
                sens_loc[sens] = 1
        else:
            sens_loc[sens_idx] = 1

        self.feature = dataset.features[:,~sens_loc] #data without sensitive
        self.feature = normalize(self.feature)
        
        self.sensitive = dataset.features[:,sens_loc]
        #n_values = int(np.max(self.label) + 1)
        #self.label = np.eye(n_values)[self.label.astype(int)].squeeze(1)
        self.enc = dict()
        for i, idx in enumerate(np.unique(self.sensitive, axis = 0)):
            self.enc[str(idx)] = i   
            
    def __getitem__(self, idx):
        y = self.label[idx]
        x = self.feature[idx]
        a = self.enc[str(self.sensitive[idx])]
        
        return x, a, y
    
 
    def __len__(self):
        return len(self.label)