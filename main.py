import os
import shutil
import random, math
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import re
import wandb
from utils.helpers import makedir, MyImageFolder, datestr
import push, model, train_and_test as tnt
from utils import save, losses
from utils.log import create_logger
from utils.preprocess import mean, std, preprocess_input_function
import settings


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', type=str, default='0')
parser.add_argument('-dataset', type=str, default="CUB")
parser.add_argument('-arch', type=str, default='resnet34')
parser.add_argument('-aux_loss', type=str, default="Proxy_Anchor")
parser.add_argument('-aux_emb_sz', type=int, default=32)
parser.add_argument('-mem_sz', type=int, default=800, help="memory capacity")
parser.add_argument('-mine_level', type=int, default=20, help="number of mining levels")
args = parser.parse_args()


model_dir ='saved_models-R34/{}/'.format(datestr()) + args.arch + '/' + '/'
if os.path.exists(model_dir) is True:
    shutil.rmtree(model_dir)
makedir(model_dir)


log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'


# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic = True


# WandB – Initialize a new run
wandb.init(project='MGProto-CUB', mode='disabled')   # mode='disabled'
wandb.run.name = 'R34-' + wandb.run.id


#model param
num_classes = settings.num_classes
img_size = settings.img_size
add_on_layers_type = settings.add_on_layers_type
prototype_shape = settings.prototype_shape
prototype_activation_function = settings.prototype_activation_function
#datasets
train_dir = settings.train_dir
test_dir = settings.test_dir
train_push_dir = settings.train_push_dir

test_dir_ood1 = settings.test_dir_ood1
test_dir_ood2 = settings.test_dir_ood2

train_batch_size = settings.train_batch_size
test_batch_size = settings.test_batch_size
train_push_batch_size = settings.train_push_batch_size
#optimzer
joint_optimizer_lrs = settings.joint_optimizer_lrs
joint_lr_step_size = settings.joint_lr_step_size
warm_optimizer_lrs = settings.warm_optimizer_lrs

last_layer_optimizer_lr = settings.last_layer_optimizer_lr
# weighting of different training losses
coefs = settings.coefs
# number of training epochs, number of warm epochs, push start epoch, push epochs
num_train_epochs = settings.num_train_epochs
num_warm_epochs = settings.num_warm_epochs
push_start = settings.push_start
push_epochs = settings.push_epochs
mine_start = settings.mine_start
updateGMM_start = settings.updateGMM_start

log(train_dir)

# all datasets
# train set
num_workers = 0  # 16
normalize = transforms.Normalize(mean=mean, std=std)
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=25, shear=(-15, 15), translate=[0.05, 0.05]),
        transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.60, 1.0)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=False)
# push set
train_push_dataset = MyImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)
# test set
# test_dataset = datasets.ImageFolder(
#     test_dir,
#     transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#         normalize,
#     ]))
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize((img_size+32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)

# test set ood
test_dataset_ood1 = datasets.ImageFolder(
    test_dir_ood1,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader_ood1 = torch.utils.data.DataLoader(
    test_dataset_ood1, batch_size=test_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)


# test set ood
test_dataset_ood2 = datasets.ImageFolder(
    test_dir_ood2,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader_ood2 = torch.utils.data.DataLoader(
    test_dataset_ood2, batch_size=test_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)
    

log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))
log("basis concept size:{}".format(prototype_shape))
# construct the model
ppnet = model.construct_MGProto(base_architecture=args.arch,
                                pretrained=True,
                                img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type,
                                sz_embedding=args.aux_emb_sz,
                                mem_capacity=args.mem_sz,
                                mine_K=args.mine_level,
                                )
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

# DML Losses
if args.aux_loss == 'Proxy_Anchor':
    aux_criterion = losses.Proxy_Anchor(nb_classes=num_classes, sz_embed=args.aux_emb_sz, mrg=0.1, beta=32).cuda()
elif args.loss == 'Proxy_NCA':
    aux_criterion = losses.Proxy_NCA(nb_classes=num_classes, sz_embed=args.aux_emb_sz).cuda()
elif args.loss == 'MS':
    aux_criterion = losses.MultiSimilarityLoss().cuda()
elif args.loss == 'Contrastive':
    aux_criterion = losses.ContrastiveLoss().cuda()
elif args.loss == 'Triplet':
    aux_criterion = losses.TripletLoss().cuda()
elif args.loss == 'NPair':
    aux_criterion = losses.NPairLoss().cuda()

class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size

joint_optimizer_specs = \
[
 {'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-4},
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-4},
 {'params': aux_criterion.parameters(), 'lr': joint_optimizer_lrs['features'] * 100, 'weight_decay': 1e-4},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=1, gamma=0.4)   # 0.5

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[
 {'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-4},
 {'params': aux_criterion.parameters(), 'lr': joint_optimizer_lrs['features'] * 100, 'weight_decay': 1e-4},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)


GMM_optimizer_specs = \
[
 {'params': ppnet.prototype_means, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
prototype_optimizer = torch.optim.Adam(GMM_optimizer_specs)
ppnet.prototype_optimizer = prototype_optimizer
prototype_lr_scheduler = torch.optim.lr_scheduler.StepLR(ppnet.prototype_optimizer, step_size=1, gamma=0.4)


# train the model
log('start training')
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    use_mining = epoch >= mine_start
    update_GMM = (epoch >= updateGMM_start) and (ppnet.queue.mem_len == ppnet.capacity_pc).all()
    log('use mining: \t{0}'.format(use_mining))
    log('update GMM: \t{0}'.format(update_GMM))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _, train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer, use_mine=use_mining, aux_criterion=aux_criterion,
                                     update_GMM=update_GMM, class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        if epoch in [30, 45, 60, 75, 90]:  # R34
        # if epoch in [10, 15, 20, 25, 30]:  # R50
            joint_lr_scheduler.step()
        _, train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,  use_mine=use_mining, aux_criterion=aux_criterion,
                                     update_GMM=update_GMM, class_specific=class_specific, coefs=coefs, log=log)

    accu, test_results = tnt.test(model=ppnet_multi, dataloader=(test_loader, test_loader_ood1, test_loader_ood2), class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.00, log=log)
    wandb.log({
        'Epoch': epoch,
        "Optimizer LR": joint_optimizer.param_groups[0]['lr'],
        "Optimizer PLR": ppnet.prototype_optimizer.param_groups[0]['lr'],
        'update_GMM': update_GMM * 1.,
        'use_mining': use_mining * 1.,
    })

    # prototype projection
    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
             train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
             prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
             class_specific=class_specific,
             preprocess_input_function=preprocess_input_function,  # normalize if needed
             prototype_layer_stride=1,
             root_dir_for_saving_prototypes=img_dir,  # if not None, prototypes will be saved here
             epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
             prototype_img_filename_prefix=prototype_img_filename_prefix,
             prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
             proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
             save_prototype_class_identity=True,
             log=log)
        accu, test_results = tnt.test(model=ppnet_multi, dataloader=(test_loader, test_loader_ood1, test_loader_ood2), class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.00, log=log)

# pruning (optional)
ppnet.prune_prototypes_topM(top_M=8)
accu, test_results = tnt.test(model=ppnet_multi, dataloader=(test_loader, test_loader_ood1, test_loader_ood2), class_specific=class_specific, log=log)
save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'prune', accu=accu, target_accu=0.00, log=log)

logclose()

