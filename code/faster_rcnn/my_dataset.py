from torch.utils.data import Dataset
from PIL import Image
from os import path
import json
import torch
import torchvision



class CropDataset(Dataset):
    def get_boxes_inside_box(self,big_box, boxes, min_overlap=0.5):
        """
        get all the bboxes that have more than min_overlap part of their area inside big_box
        box := [x_min,y_min,x_max,y_max]
        """
        inside_bboxes = []
        temp_ids = []
        for idx, bbox in enumerate(boxes):
            x_min = max(bbox[0],big_box[0])
            y_min = max(bbox[1],big_box[1])
            x_max = min(bbox[2],big_box[2])
            y_max = min(bbox[3],big_box[3])
            if x_max-x_min > 0 and y_max-y_min > 0:
                bbox_area = (bbox[0]-bbox[2])*(bbox[1]-bbox[3])
                if (x_max-x_min)*(y_max-y_min) > min_overlap * bbox_area:
                    inside_bboxes.append([x_min,y_min,x_max,y_max])
                    temp_ids.append(idx)
        return inside_bboxes,temp_ids
    
    def __init__(self, img_path, annotation_path, dataset='none', crop_size =224, 
                 stride = 100, transforms=None, splits = [.25,.15], split_direction = 'v'):
        self.img = Image.open(img_path)
        with open(annotation_path,'r') as f:
            annotations = json.loads(f.read())
        all_bboxes = []
        all_bbox_ids = []
        all_bbox_classes = []
        all_bbox_areas = []
        for ann in annotations['annotations']:
            x_min = ann['bbox'][0]
            y_min = ann['bbox'][1]
            x_max = ann['bbox'][0]+ann['bbox'][2] 
            y_max = ann['bbox'][1]+ann['bbox'][3]
            all_bboxes.append([x_min,y_min,x_max,y_max])
            all_bbox_ids.append(ann['id'])
            all_bbox_classes.append(ann['category_id'])
            all_bbox_areas.append(ann['area'])
        test_frac=splits[0]
        val_frac=splits[1]
        width = self.img.width
        height = self.img.height
        if split_direction == 'v':
            test_split = int(width*(1-test_frac))
            val_split = int(test_split*(1-val_frac))
            if dataset == 'train':
                self.set_box = [0,0,val_split,height]
            elif dataset == 'validation':
                self.set_box = [val_split,0,test_split,height]
            elif dataset == 'test':
                self.set_box = [test_split,0,width,height]
            else:
                self.set_box = [0,0,width,height]
                print('Warning: using entire image with no split!')
        elif split_direction == 'h':
            test_split = int(height*(1-test_frac))
            val_split = int(test_split*(1-val_frac))
            if dataset == 'train':
                self.set_box = [0,0,width,val_split]
            elif dataset == 'validation':
                self.set_box = [0,val_split,width,test_split]
            elif dataset == 'test':
                self.set_box = [0,test_split,width,height]
            else:
                self.set_box = [0,0,width,height]
                print('Warning: using entire image with no split!')
        else:
            print('split direction must be \'h\' or \'v\'')
            raise()
        n_width = int((self.set_box[2]-self.set_box[0] - crop_size) / stride + 1)
        n_height = int((self.set_box[3]-self.set_box[1]  - crop_size) / stride + 1)
        
        self.crop_boxes = []
        self.crop_ids = []
        self.bboxes = []
        self.classes = []
        self.areas = []
        crop_id = 0
        for j in range(n_height):
            for i in range(n_width):
                crop_box = [i * stride + self.set_box[0], 
                            j * stride + self.set_box[1], 
                            i * stride + crop_size + self.set_box[0], 
                            j * stride + crop_size + self.set_box[1]]
                self.crop_boxes.append(crop_box)
                ct_temp,bbox_ids = self.get_boxes_inside_box(crop_box,all_bboxes,0.3)
                ct=[]
                for i,box in enumerate(ct_temp):
                    ct.append([box[0]-crop_box[0],box[1]-crop_box[1],box[2]-crop_box[0],box[3]-crop_box[1]])
                    
                self.bboxes.append(ct)
                self.crop_ids.append(crop_id)
                self.areas.append([all_bbox_areas[bbox_id] for bbox_id in bbox_ids])
                self.classes.append([all_bbox_classes[bbox_id] for bbox_id in bbox_ids])
                crop_id += 1
        if transforms is None:
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        else:
            transforms.append(torchvision.transforms.ToTensor())
            self.transforms = torchvision.transforms.Compose(transforms)

    def __getitem__(self, index):
        
        # ID
        crop_id = torch.tensor(self.crop_ids[index])
        
        # Crop location
        crop_box = self.crop_boxes[index]
        
        # Generate the crop
        crop_img = self.img.crop(crop_box)
        
        # Bounding boxes for objects (created at initialization)
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        
        boxes = torch.as_tensor(self.bboxes[index], dtype=torch.float32)
        
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((len(boxes),),dtype=torch.int64)
        
         # Size of bbox (Rectangular)
        areas = torch.as_tensor(self.areas[index], dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = crop_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            crop_img = self.transforms(crop_img)
        
        return crop_img, my_annotation

    def __len__(self):
        return len(self.crop_ids)

def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    data_path = '/data/leavesboxes1image/'

    annotatio_path = path.join(data_path,'via_export_coco_cat.json')

    image_path = path.join(data_path,'20200307_160518.jpg')

    my_dataset = CropDataset(image_path,annotatio_path)

    test_crop, ann = my_dataset[1]
    f,ax = plt.subplots(1,1,figsize=(10,10))

    plt.imshow(test_crop)
    p=[]
    for box in ann['boxes']:
        p.append(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1]))
        ax.add_collection(PatchCollection(p,linewidth=1,edgecolor='r',facecolor='none'))
    plt.show()
if __name__ == '__main__':
    main()