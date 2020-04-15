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