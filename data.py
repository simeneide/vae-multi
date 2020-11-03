#%%
"""
pip install pycocotools
# Download files frm https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

"""
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer
# %%
from PIL import Image
import os
import os.path
from torch.utils.data import DataLoader
#import wget

class CocoCaptionsDict(dset.VisionDataset):
    
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoCaptionsDict, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Tokenize:
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Tokenize captions:
        target = [ann['caption'] for ann in anns]
        target = self.tokenizer((target[0]), padding='max_length',max_length=20, truncation=True, return_tensors="pt")['input_ids']
        target = target

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return {'image' : img, 'text' : target.squeeze()}

    def __len__(self):
        return len(self.ids)

class COCOCaptionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str ="./data", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.prepare_data()
        self.dims = (1, 32, 32)
        self.transform = transforms.Compose([
            transforms.Resize( self.dims[1:]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        self.setup()

    def prepare_data(self):
        pass
        # Annotations:
        #http://images.cocodataset.org/annotations/annotations_trainval2014.zip
        #http://images.cocodataset.org/zips/train2014.zip
        #http://images.cocodataset.org/zips/val2014.zip
        #wget.download("http://images.cocodataset.org/zips/test2014.zip", out=f"{self.data_dir}/")
        
    def setup(self, stage=None):
        self.datasets = {}
        for stage in ['train','val']: # ,'test'
            self.datasets[stage] = CocoCaptionsDict(root = f'{self.data_dir}/{stage}2014',
                            annFile = f'{self.data_dir}/annotations/captions_{stage}2014.json',
                            transform=self.transform)
    def build_dataloader(self, stage):
        return DataLoader(self.datasets[stage], batch_size=self.batch_size, num_workers=8)
    def train_dataloader(self):
        return self.build_dataloader("train")

    def val_dataloader(self):
        return self.build_dataloader("val")

    def test_dataloader(self):
        return None

### CIFAR
"""
from pl_bolts.datamodules import CIFAR10DataModule
dataset = CIFAR10DataModule(data_dir=".")
batch =next(iter(dataset.train_dataloader()))
x, y = batch
"""
#%%
if __name__ == "__main__":
    dm = COCOCaptionDataModule()
    batch = dm.datasets['train'][2]
    print('Number of samples: ', len(batch['caption']))

    print("Image Size: ", batch['image'].size())
    print(batch['caption'])
    import matplotlib.pyplot as plt
    plt.imshow(batch['image'].permute(1,2,0))
    # %%
