import csv
import os
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

# PIAA Dataset
class FlickrAESPIAADataset(Dataset):
    def __init__(self, root_dir, label_file, image_size, transform=None, 
                 split='train', worker='', train_list=None, ):

        self.root_dir = root_dir
        # This might throw an error in a ddp setting. Simply run the script again.
        if not os.path.isfile(os.path.join(root_dir, f"{label_file.split('.')[0]}_metadata.pkl")):
            with open(os.path.join(root_dir, label_file), 'r') as f:
                reader = csv.reader(f)
                lines = [i for i in reader]
            self.data_list = [{'worker':line[0], 'MOS':float(line[2]), 'image_name':line[1]} for line in lines[1:]]
            with open(os.path.join(root_dir, f"{label_file.split('.')[0]}_metadata.pkl"), 'wb') as f:
                pickle.dump(self.data_list, f)
        else:
            with open(os.path.join(root_dir, f"{label_file.split('.')[0]}_metadata.pkl"), 'rb') as f:
                self.data_list = pickle.load(f)

        train_list = [i for i in self.data_list if i['image_name'] in train_list and i['worker'] == worker]

        if split == 'train':
            self.data_list = train_list
        elif split == 'test':
            self.data_list = [i for i in self.data_list if i not in train_list and i['worker'] == worker]

        print(f'Flickr AES dataset for worker: {worker}')
        print(f'Flickr AES dataset size: {len(self.data_list)}')

        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = {'image': None, 'MOS': None, 'image_name': '', 'worker': ''}
        meta = self.data_list[idx]

        image = Image.open(os.path.join(self.root_dir, '40K', meta['image_name']))
        image = image.convert('RGB')
        image = self.transform(image)

        data['image'] = image
        data['image_name'] = meta['image_name']
        data['MOS'] = float(meta['MOS'])
        data['worker'] = meta['worker']

        return data
    
class REALCURDataset(Dataset):
    def __init__(self, root_dir, label_file, image_size, transform=None, 
                 split='train', worker='', train_list=None, ):

        self.root_dir = root_dir
        # This might throw an error in a ddp setting. Simply run the script again.
        if not os.path.isfile(os.path.join(root_dir, f"metadata.pkl")):
            with open(os.path.join(root_dir, label_file), 'r') as f:
                reader = csv.reader(f)
                lines = [i for i in reader]
            self.data_list = [{'worker':line[0], 'MOS':float(line[2]), 'image_name':line[1]} for line in lines[1:]]
            with open(os.path.join(root_dir, "metadata.pkl"), 'wb') as f:
                pickle.dump(self.data_list, f)
        else:
            with open(os.path.join(root_dir, "metadata.pkl"), 'rb') as f:
                self.data_list = pickle.load(f)

        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

        self.worker = worker
        train_list = [i for i in self.data_list if i['image_name'] in train_list and i['worker'] == worker]

        if split == 'train':
            self.data_list = train_list
        elif split == 'test':
            self.data_list = [i for i in self.data_list if i not in train_list and i['worker'] == worker]

        print(f'REAL-CUR dataset for worker: {worker}')
        print(f'REAL-CUR dataset size: {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = {'image': None, 'MOS': None, 'image_name': '', 'worker': ''}
        meta = self.data_list[idx]

        # data is from the Flickr AES dataset
        image = Image.open(os.path.join(self.root_dir, 'images', self.worker, str(int(meta['MOS'])), meta['image_name']))
        image = image.convert('RGB')
        image = self.transform(image)

        data['image'] = image
        data['image_name'] = meta['image_name']
        data['MOS'] = float(meta['MOS'])
        data['worker'] = meta['worker']

        return data

class PARAPIAADataset(Dataset):
    def __init__(self, root_dir, label_file, image_size, transform=None, 
                 split='train', worker='', train_list=None, ):

        self.root_dir = root_dir
        # This might throw an error in a ddp setting. Simply run the script again.
        if not os.path.isfile(os.path.join(root_dir, f"metadata.pkl")):
            with open(os.path.join(root_dir, label_file), 'r') as f:
                reader = csv.reader(f)
                lines = [i for i in reader]
            self.data_list = [{'worker':line[0], 'MOS':2 * float(line[2]), 
                               'image_name':line[1], 'session_id': line[3]} for line in lines[1:]]
            with open(os.path.join(root_dir, "metadata.pkl"), 'wb') as f:
                pickle.dump(self.data_list, f)
        else:
            with open(os.path.join(root_dir, "metadata.pkl"), 'rb') as f:
                self.data_list = pickle.load(f)

        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

        self.worker = worker
        train_list = [i for i in self.data_list if i['image_name'] in train_list and i['worker'] == worker]

        if split == 'train':
            self.data_list = train_list
        elif split == 'test':
            self.data_list = [i for i in self.data_list if i not in train_list and i['worker'] == worker]

        print(f'PARA PIAA dataset for worker: {worker}')
        print(f'PARA PIAA dataset size: {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = {'image': None, 'MOS': None, 'image_name': '', 'worker': ''}
        meta = self.data_list[idx]

        # data is from the Flickr AES dataset
        image = Image.open(os.path.join(self.root_dir, 'imgs_40', meta['session_id'], meta['image_name']))
        image = image.convert('RGB')
        image = self.transform(image)

        data['image'] = image
        data['image_name'] = meta['image_name']
        data['MOS'] = float(meta['MOS'])
        data['worker'] = meta['worker']

        return data

class AADBDataset(Dataset):
    def __init__(self, root_dir, label_file, image_size, transform=None, 
                 split='train', worker='', train_list=None, ):

        self.root_dir = root_dir
        # This might throw an error in a ddp setting. Simply run the script again.
        if not os.path.isfile(os.path.join(root_dir, f"metadata.pkl")):
            with open(os.path.join(root_dir, label_file), 'r') as f:
                reader = csv.reader(f)
                lines = [i for i in reader]
            self.data_list = [{'worker':line[0], 'MOS':float(line[2]), 'image_name':line[1]} for line in lines[1:]]
            with open(os.path.join(root_dir, "metadata.pkl"), 'wb') as f:
                pickle.dump(self.data_list, f)
        else:
            with open(os.path.join(root_dir, "metadata.pkl"), 'rb') as f:
                self.data_list = pickle.load(f)

        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

        self.worker = worker
        train_list = [i for i in self.data_list if i['image_name'] in train_list and i['worker'] == worker]

        if split == 'train':
            self.data_list = train_list
        elif split == 'test':
            self.data_list = [i for i in self.data_list if i not in train_list and i['worker'] == worker]

        print(f'AADB dataset for worker: {worker}')
        print(f'AADB dataset size: {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = {'image': None, 'MOS': None, 'image_name': '', 'worker': ''}
        meta = self.data_list[idx]

        # data is from the Flickr AES dataset
        image = Image.open(os.path.join(self.root_dir, 'datasetImages_originalSize', meta['image_name']))
        image = image.convert('RGB')
        image = self.transform(image)

        data['image'] = image
        data['image_name'] = meta['image_name']
        data['MOS'] = float(meta['MOS'])
        data['worker'] = meta['worker']

        return data