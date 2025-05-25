import csv
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
import csv
import torchvision.transforms as transforms
import torch.nn.functional as F


# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    # in case latent output size is 26x26. For convnext with smaller strides.
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        # Since the outer latent patches have a smaller receptive field, skip size is set to 4 for the first and last patch. 8 for rest.
        h_coor_min = max(0, (h_idx - 1) * skip + 4)
        if h_idx < softmaxes_shape[-1] - 1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize
        w_coor_min = max(0, (w_idx - 1) * skip + 4)
        if w_idx < softmaxes_shape[-1] - 1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = h_idx * skip
        h_coor_max = min(img_size, h_idx * skip + patchsize)
        w_coor_min = w_idx * skip
        w_coor_max = min(img_size, w_idx * skip + patchsize)

    if h_idx == softmaxes_shape[1] - 1:
        h_coor_max = img_size
    if w_idx == softmaxes_shape[2] - 1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size - patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size - patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max


def get_patch_size(args):
    patchsize = 32
    # patchsize = 40
    skip = round((args.image_size - patchsize) / (args.wshape-1))
    return patchsize, skip


# Evaluates purity of CUB prototypes from csv file. General method that can be used for other part-prototype methods as well
# Assumes that coordinates in csv file apply to a 224x224 image!
def eval_prototypes_cub_parts_csv(csvfile, parts_loc_path, parts_name_path, imgs_id_path, epoch, args, log):
    patchsize, _ = get_patch_size(args)
    imgresize = float(args.image_size)
    path_to_id = dict()
    id_to_path = dict()
    with open(imgs_id_path) as f:
        for line in f:
            id, path = line.split('\n')[0].split(' ')
            path_to_id[path]=id
            id_to_path[id]=path

    img_to_part_xy_vis = dict()
    with open(parts_loc_path) as f:
        for line in f:
            img, partid, x, y, vis = line.split('\n')[0].split(' ')
            x =float(x)
            y =float(y)
            if img not in img_to_part_xy_vis.keys():
                img_to_part_xy_vis[img]=dict()
            if vis == '1':
                img_to_part_xy_vis[img][partid]=(x,y)

    parts_id_to_name = dict()
    parts_name_to_id = dict()
    with open (parts_name_path) as f:
        for line in f:
            id, name = line.split('\n')[0].split(' ',1)
            parts_id_to_name[id]=name
            parts_name_to_id[name]=id
    print(parts_id_to_name)

    # merge left and right cub parts
    duplicate_part_ids = []
    with open (parts_name_path) as f:
        for line in f:
            id, name = line.split('\n')[0].split(' ',1)
            if 'left' in name:
                new_name = name.replace('left', 'right')
                
                duplicate_part_ids.append((id, parts_name_to_id[new_name]))
           
    proto_parts_presences = dict()
    
    with open (csvfile, newline='') as f:
        filereader = csv.reader(f, delimiter=',')
        next(filereader) #skip header
        for (prototype, imgname, h_min_224, h_max_224, w_min_224, w_max_224) in filereader:
            
            if prototype not in proto_parts_presences.keys():
                proto_parts_presences[prototype]=dict()
            p = prototype
            img = Image.open(imgname)
            imgname = imgname.replace('\\', '/')
            imgnamec, imgnamef = imgname.split('/')[-2:]
            if 'normal_' in imgnamef:
                imgnamef = imgnamef.split('normal_')[-1]
            imgname = imgnamec+'/'+imgnamef
            img_id = path_to_id[imgname]
            img_orig_width, img_orig_height = img.size
            h_min_224, h_max_224, w_min_224, w_max_224 = float(h_min_224), float(h_max_224), float(w_min_224), float(w_max_224)
            
            diffh = h_max_224 - h_min_224
            diffw = w_max_224 - w_min_224
            if diffh > patchsize: #patch size too big, we take the center. otherwise the bigger the patch, the higher the purity.
                correction = diffh-patchsize
                h_min_224 = h_min_224 + correction//2.
                h_max_224 = h_max_224 - correction//2.
            if diffw > patchsize:
                correction = diffw-patchsize
                w_min_224 = w_min_224 + correction//2.
                w_max_224 = w_max_224 - correction//2.

            orig_img_location_h_min = (img_orig_height/imgresize) * h_min_224 
            orig_img_location_h_max = (img_orig_height/imgresize) * h_max_224 
            orig_img_location_w_min = (img_orig_width/imgresize) * w_min_224 
            orig_img_location_w_max = (img_orig_width/imgresize) * w_max_224 
                        
            part_dict_img = img_to_part_xy_vis[img_id]
            for part in part_dict_img.keys():
                x, y = part_dict_img[part]
                part_in_patch = 0 
                if y >= orig_img_location_h_min and y <= orig_img_location_h_max:
                    if x >= orig_img_location_w_min and x <= orig_img_location_w_max:
                        part_in_patch = 1
                if part not in proto_parts_presences[p].keys():
                    proto_parts_presences[p][part]=[]
                proto_parts_presences[p][part].append(part_in_patch)
            
            for pair in duplicate_part_ids:
                if pair[0] in part_dict_img.keys():
                    if pair[1] in part_dict_img.keys():
                        presence0 = proto_parts_presences[p][pair[0]][-1]
                        presence1 = proto_parts_presences[p][pair[1]][-1]
                        if presence0 > presence1: 
                            proto_parts_presences[p][pair[1]][-1] = presence0

                        del proto_parts_presences[p][pair[0]]
                    else:

                        if pair[1] not in proto_parts_presences[p].keys():
                            proto_parts_presences[p][pair[1]]=[]
                        proto_parts_presences[p][pair[1]].append(proto_parts_presences[p][pair[0]][-1])
                        del proto_parts_presences[p][pair[0]]
                        
    log("\n Eval CUB Parts - Epoch: \t{0}".format(epoch))
    log("Number of prototypes in parts_presences: \t{0}".format(len(proto_parts_presences.keys())))
    
    prototypes_part_related = 0
    max_presence_purity = dict()
    max_presence_purity_part = dict()
    max_presence_purity_sum = dict()

    most_often_present_purity = dict()
    part_most_present = dict()
    
    for proto in proto_parts_presences.keys():
        
        max_presence_purity[proto]= 0.
        part_most_present[proto] = ('0',0)
        most_often_present_purity[proto] = 0.

        # CUB parts 7,8 and 9 are  duplicate (right and left). additional check that these should not occur (already fixed earlier in this function)
        if ('7' in proto_parts_presences[proto].keys() or '8' in proto_parts_presences[proto].keys() or '9' in proto_parts_presences[proto].keys()):
            log("unused part in keys! \t{0}  \t{1}  \t{2}".format(proto, proto_parts_presences[proto].keys(), proto_parts_presences[proto]))
            raise ValueError()
        
        for part in proto_parts_presences[proto].keys():
            presence_purity = np.mean(proto_parts_presences[proto][part])
            sum_occurs = np.array(proto_parts_presences[proto][part]).sum()
        
            # evaluate whether the purity of this prototype for this part is higher than for other parts
            if presence_purity > max_presence_purity[proto]:
                max_presence_purity[proto]=presence_purity
                max_presence_purity_part[proto]=parts_id_to_name[part]
                max_presence_purity_sum[proto] = sum_occurs
            elif presence_purity == max_presence_purity[proto]:
                if presence_purity == 0.:
                    max_presence_purity[proto]=presence_purity
                    max_presence_purity_part[proto]=parts_id_to_name[part]
                    max_presence_purity_sum[proto] = sum_occurs
                elif sum_occurs > max_presence_purity_sum[proto]:
                    max_presence_purity[proto]=presence_purity
                    max_presence_purity_part[proto]=parts_id_to_name[part]
                    max_presence_purity_sum[proto] = sum_occurs
        
            if sum_occurs > part_most_present[proto][1]:
                part_most_present[proto] = (part, sum_occurs)
                most_often_present_purity[proto]=presence_purity         
        if max_presence_purity[proto] > 0.5:
            prototypes_part_related += 1 
            
    log("Number of part-related prototypes (purity>0.5): \t{0}".format(prototypes_part_related))

    log("Mean purity of prototypes (corresponding to purest part): \t{0}     std: \t{1}".format(np.mean(list(max_presence_purity.values())), np.std(list(max_presence_purity.values()))))
    log("Prototypes with highest-purity part (no contraints): \t{0}".format(max_presence_purity_part))
    log("Prototype with part that has most often overlap with prototype: \t{0}".format(part_most_present))

    print('log_epoch_overview', "p_cub_" + str(epoch),
                   "mean purity (averaged over all prototypes, corresponding to purest part)", "std purity",
                   "mean purity (averaged over all prototypes, corresponding to part with most often overlap)",
                   "std purity", "# prototypes in csv", "#part-related prototypes (purity > 0.5)", "", "")

    print('log_epoch_overview', "p_cub_" + str(epoch), np.mean(list(max_presence_purity.values())),
                   np.std(list(max_presence_purity.values())), np.mean(list(most_often_present_purity.values())),
                   np.std(list(most_often_present_purity.values())), len(list(proto_parts_presences.keys())),
                   prototypes_part_related, "", "")


# Writes coordinates of image patches per prototype to csv file (image resized to 224x224)
def get_proto_patches_cub(net, projectloader, epoch, device, args, threshold=0.5):    
    # Make sure the model is in evaluation mode
    net.eval()
    
    imgs = projectloader.dataset.imgs
    classification_weights = net.module.last_layer.weight
    patchsize, skip = get_patch_size(args)
    
    proto_img_coordinates = []
    
    csvfilepath = os.path.join(args.log_dir, str(epoch)+'_pipnet_prototypes_cub_all.csv')
    columns = ["prototype", "img name", "h_min_224", "h_max_224", "w_min_224", "w_max_224"]
    with open(csvfilepath, "w", newline='') as csvfile:
        print("Collecting Prototype Image Patches for Evaluating CUB part purity. Writing CSV file with image patche coordinates..", flush=True)
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(columns)
        # Iterate through the prototypes and projection set
        img_iter = tqdm(enumerate(range(len(imgs))), total=len(imgs),mininterval=50.,ncols=0,desc='Collecting patch coordinates CUB')
        for _, imgid in img_iter:
            imgname = imgs[imgid][0]
            imgtensor = projectloader.dataset[imgid][0].unsqueeze(0).to(device)
            with torch.no_grad():
                # Use the model to classify this input image
                _, proto_acts = net.module.push_forward(imgtensor)
                proto_acts = - proto_acts
                pooled = F.max_pool2d(proto_acts, kernel_size=(proto_acts.size()[2], proto_acts.size()[3])).squeeze()
                proto_acts = proto_acts.squeeze(0)

                for prototype in range(proto_acts.shape[0]):
                    c_weight = torch.max(classification_weights[:,prototype]) 
                    if c_weight > 1e-5:#ignore prototypes that are not relevant to any class
                        if pooled[prototype].item()>threshold: #similarity score > threshold
                            location_h, location_h_idx = torch.max(proto_acts[prototype,:,:], dim=0)
                            _, location_w_idx = torch.max(location_h, dim=0)
                            h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, proto_acts.shape, patchsize, skip, location_h_idx[location_w_idx].item(), location_w_idx.item())
                            proto_img_coordinates.append([prototype, imgname, h_coor_min, h_coor_max, w_coor_min, w_coor_max])

        writer.writerows(proto_img_coordinates) 
    return csvfilepath

# Writes coordinates of top-k image patches per prototype to csv file (image resized to 224x224)
def get_topk_cub(net, projectloader, k, epoch, device, args):    
    # Make sure the model is in evaluation mode
    net.eval()
    
    # Show progress on progress bar
    project_iter = tqdm(enumerate(projectloader),
                        total=len(projectloader),
                        desc='@@@@ Collecting top-k Prototypes CUB parts',
                        mininterval=50.,
                        ncols=0)
    imgs = projectloader.dataset.imgs
    classification_weights = net.module.last_layer.weight
    patchsize, skip = get_patch_size(args)
    scores_per_prototype = dict()
    
    # Iterate through the projection set
    for i, (xs, ys) in project_iter:
        xs, ys = xs.to(device), ys.to(device)

        with torch.no_grad():
            # Use the model to classify this batch of input data
            _, proto_acts = net.module.push_forward(xs)
            proto_acts = - proto_acts
            pooled = F.max_pool2d(proto_acts, kernel_size=(proto_acts.size()[2], proto_acts.size()[3])).squeeze()
            proto_acts = proto_acts.squeeze(0)
            # class_p = ys.item() * 10 + np.arange(0, 10)  #############################
            for p in range(proto_acts.shape[0]):

                # if p not in class_p:         #############################
                #     continue

                c_weight = torch.max(classification_weights[:, p])
                if c_weight > 1e-5:  # ignore prototypes that are not relevant to any class
                    if p not in scores_per_prototype:
                        scores_per_prototype[p] = []
                    scores_per_prototype[p].append((i, pooled[p].item()))

    # top_M = 2
    # pos_mask = torch.t(net.module.prototype_class_identity).cuda()
    # proto_prior = net.module.last_layer.weight[pos_mask == 1].view(net.module.num_classes, -1)
    # threshold_prune, keep_index = torch.topk(proto_prior, top_M, dim=1)  # [200, top_M]
    # for c in range(keep_index.shape[0]):
    #     keep_index[c, :] = keep_index[c, :] + c*10
    # keep_index = keep_index.view(-1).cpu().numpy().tolist()

    proto_img_coordinates = []
    csvfilepath = os.path.join(args.log_dir, str(epoch)+'_pipnet_prototypes_cub_topk.csv')
    too_small = set()
    protoype_iter = tqdm(enumerate(scores_per_prototype.keys()), total=len(list(scores_per_prototype.keys())), mininterval=5., ncols=0, desc='Collecting top-k patch coordinates CUB')
    # protoype_iter = tqdm(enumerate(keep_index), total=len(keep_index), mininterval=5., ncols=0, desc='Collecting top-k patch coordinates CUB')  ###############
    with open(csvfilepath, "w", newline='') as csvfile:
        print("Writing CSV file with top k image patches..", flush=True)
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["prototype", "img name", "h_min_224", "h_max_224", "w_min_224", "w_max_224"])
        for _, prototype in protoype_iter:   # [0-1999]
            df = pd.DataFrame(scores_per_prototype[prototype], columns=['img_id', 'scores'])
            topk = df.nlargest(k, 'scores')
            for index, row in topk.iterrows():
                imgid = int(row['img_id'])
                imgname = imgs[imgid][0]
                imgtensor = projectloader.dataset[imgid][0].unsqueeze(0).to(device)
                with torch.no_grad():
                    # Use the model to classify this batch of input data
                    _, proto_acts = net.module.push_forward(imgtensor)
                    proto_acts = - proto_acts
                    pooled = F.max_pool2d(proto_acts, kernel_size=(proto_acts.size()[2], proto_acts.size()[3])).squeeze()
                    proto_acts = proto_acts.squeeze(0)
                    if pooled[prototype].item() < 0.1:
                        too_small.add(prototype)
                    location_h, location_h_idx = torch.max(proto_acts[prototype, :, :], dim=0)
                    _, location_w_idx = torch.max(location_h, dim=0)
                    location = (location_h_idx[location_w_idx].item(), location_w_idx.item())
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, proto_acts.shape, patchsize, skip, location[0], location[1])
                    proto_img_coordinates.append([prototype, imgname, h_coor_min, h_coor_max, w_coor_min, w_coor_max])
            # write intermediate results in case of large dataset
            if len(proto_img_coordinates) > 10000:
                writer.writerows(proto_img_coordinates) 
                proto_img_coordinates = []
        print("Warning: image patches included in topk, but similarity < 0.1! This might unfairly reduce the purity metric because prototype has less than k similar image patches. You could consider reducing k for prototypes", too_small, flush=True)

        writer.writerows(proto_img_coordinates)

    return csvfilepath
