import argparse
import os
from pathlib import Path


import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import apply_external_classifier, check_dataset, check_file, check_img_size, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.torch_utils import select_device, time_synchronized


@torch.no_grad()
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         compute_loss=None,
         half_precision=True,
         opt=None):
    
    set_logging()
    device = select_device(opt.device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

        
    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.safe_load(f)
    
    check_dataset(data)  # check
    #nc = 1 if single_cls else int(data['nc'])  # number of classes
            
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    dataloader = create_dataloader(data["test"], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr('Test: '))[0]

    seen = 0
        
    loss = torch.zeros(3, device=device)
    stats = []
    for batch_i, (batch_imgs, batch_labels, paths, shapes) in enumerate(tqdm(dataloader)):
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        batch_imgs = batch_imgs.half() if half else batch_imgs.float()  # uint8 to fp16/32
        batch_imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        batch_labels = batch_labels.to(device)
        nb, _, height, width = batch_imgs.shape  # batch size, channels, height, width

        # Run model
        
        out, train_out = model(batch_imgs, augment=augment)  # inference and training outputs
        #print(out.shape)
        #print(len(train_out))

        
        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], batch_labels)[1][:3]  # box, obj, cls

        # Run NMS
        batch_labels[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [batch_labels[batch_labels[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        #print(out[0].shape,out[1].shape,out[2].shape)
        
        # Statistics per image
        for idx, pred in enumerate(out):
            labels = batch_labels[batch_labels[:, 0] == idx, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            #print("length of labels",nl)
            #print("labels ::",labels)
            path = Path(paths[idx])
            seen += 1

            
            # Predictions
            if single_cls:
                pred[:, 5] = 0     # setting the class to 0.
            
            pred_native = pred.clone()
            scale_coords(batch_imgs[idx].shape[1:], pred_native[:, :4], shapes[idx][0], shapes[idx][1])  # native-space pred
            
                
            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], dtype=torch.bool, device=device)
            assert nl > 0

            #tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords(batch_imgs[idx].shape[1:], tbox, shapes[idx][0], shapes[idx][1])  # native-space labels
                
            # for box_idx in range(pred_native.shape[0]):
            #     curr_bbox = pred_native[box_idx,:4]
            
            if len(pred_native) == 0:
                stats.append((correct,None,nl))
                continue
            
            best_ious, best_iou_indices= box_iou(pred_native[:,:4], tbox).max(1)  # best ious, indices
            
            scale_coords(shapes[idx][0], pred_native[:, :4],batch_imgs[idx].shape[1:])  #back to original space 
            
            #print("shapes[0]",shapes[idx][0])
            for i in range(len(best_ious)):     # processing one image at a time
                if best_ious[i] >= 0.50:
                    # apply classifier here !!!
                    #print("pred_native",pred_native.shape)
                    x1,y1,x2,y2 = pred_native[i][:4]
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    #print(x1,y1,x2,y2)
                    #print("batch imgs",batch_imgs.shape)
                    cropped = batch_imgs[idx,:,y1:y2,x1:x2]
                    #print("shape of crop",cropped.shape)
                    pcls = apply_external_classifier(cropped)
                    if pcls == tcls[best_iou_indices[i]] :
                        correct[i] = 1

            stats.append((correct,best_ious,nl))
            # Append statistics (correct, conf, pcls, tcls)
            #stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    # print(stats[0][0],'\n',stats[0][1])
    # print(stats[0][0].sum(),stats[0][0].shape,"actual no. of items",stats[0][2]) 

    AP = 0
    AR = 0
    for stat_img in stats:
        if len(stat_img[0]):
            AR += stat_img[0].sum()/stat_img[2]
            AP += stat_img[0].sum()/len(stat_img[0])
    
    print("len of stats",len(stats))
    AP = AP/len(stats)
    AR = AR/len(stats)

    Fscore = (2 * AP * AR) / (AP + AR)
    print("Upper bound of average precision @0.50:",AP.item())
    print("Upper bound of average recall @0.50: ",AR.item())
    print("Upper bound of average F score @0.50: ",Fscore.item())

       
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)
    
    test(opt.data,
        opt.weights,
        opt.batch_size,
        opt.img_size,
        opt.conf_thres,
        opt.iou_thres,
        opt.single_cls,
        opt.augment,
        opt.verbose,
        save_txt=opt.save_txt | opt.save_hybrid,
        save_hybrid=opt.save_hybrid,
        opt=opt
        )

    