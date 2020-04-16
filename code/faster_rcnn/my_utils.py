import torch
from pycocotools.coco import COCO

def train_one_epoch(model, optimizer, data_loader, device, print_freq):
    model.train()
    i = 0
    len_dataloader = len(data_loader)
    avg_losses = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0
    avg_losses = {}
    avg_losses['loss_combined'] = 0
    avg_losses['loss_classifier'] = 0
    avg_losses['loss_box_reg'] = 0
    avg_losses['loss_objectness'] = 0
    avg_losses['loss_rpn_box_reg'] = 0
    for imgs, annotations in data_loader:
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())
        avg_losses['loss_classifier'] += loss_dict['loss_classifier']
        avg_losses['loss_box_reg'] += loss_dict['loss_box_reg']
        avg_losses['loss_objectness'] += loss_dict['loss_objectness']
        avg_losses['loss_rpn_box_reg'] += loss_dict['loss_rpn_box_reg']
        avg_losses['loss_combined'] += losses
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if i % print_freq == 0:
            
            loss_str = '; '.join([f'{k[5:]}={v:.3f}' for k,v in avg_losses.items()])
            print(f"({i}/{len_dataloader}) loss: " + loss_str)
            for k in avg_losses.keys():
                avg_losses[k] = 0 
            
                  
#             print(f"Iteration: {i}/{len_dataloader}, Avg_Loss: {avg_losses/print_freq},"
#                   f"classifier_loss: {loss_classifier/print_freq},"
#                   f"Box_regression_loss: {loss_objectness/print_freq},"
#                   f"Objectness_loss: {loss_objectness/print_freq},"
#                   f"RPN_box_regression_loss: {loss_objectness/print_freq}")
#             avg_losses=0
#             loss_classifier = 0
#             loss_box_reg = 0
#             loss_objectness = 0
#             loss_rpn_box_reg = 0

def infer_on_dataset(model, data_loader, device, print_freq, is_coco):
    with torch.no_grad():
        model.eval()
        i = 0
        len_dataloader = len(data_loader)
        results = []
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            predictions = model(imgs)
            if is_coco:
                boxes = predictions[0]['boxes'].to('cpu').numpy()
                boxes[:,2:]= boxes[:,-2:] - boxes[:,:2]
                boxes= boxes.tolist()
            else:
                boxes = predictions[0]['boxes'].tolist()
            scores = predictions[0]['scores'].tolist()
            labels = predictions[0]['labels'].tolist()
            image_id = annotations[0]['image_id'].item()
            for bbox,score,label in zip(boxes, scores,labels):
                results.append({'image_id': image_id,
                                'category_id': label,
                                'bbox': bbox,
                                'score': score
                               })
            if i % print_freq == 0:
                print(f"({i}/{len_dataloader})")

    return results

def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        num_objs = len(bboxes)
        if num_objs <1:
            continue
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds
            
def collate_fn(batch):
    return tuple(zip(*batch))
