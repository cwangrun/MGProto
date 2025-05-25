import os
import model
import torch
import argparse
from utils.interpretability import evaluate_purity
import push, model, train_and_test as tnt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import settings
from utils.preprocess import mean, std
from utils.cub_csv import eval_prototypes_cub_parts_csv, get_topk_cub, get_proto_patches_cub
from utils.log import create_logger
from utils.helpers import makedir


parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--data_set', default='CUB2011', type=str)
parser.add_argument('--data_path', type=str, default='/mnt/c/copy/data/CUB_200_2011/CUB_200_2011/')
parser.add_argument('--nb_classes', type=int, default=200)
parser.add_argument('--test_batch_size', type=int, default=80)  # 30

# Model
parser.add_argument('--base_architecture', type=str, default='resnet50')
parser.add_argument('--input_size', default=224, type=int, help='images input size')
parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
parser.add_argument('--prototype_activation_function', type=str, default='log')
parser.add_argument('--add_on_layers_type', type=str, default='regular')
parser.add_argument('--aux_emb_sz', type=int, default=32)
parser.add_argument('--mem_sz', type=int, default=800, help="memory capacity")
parser.add_argument('--mine_level', type=int, default=20, help="number of mining levels")
parser.add_argument('--resume', type=str)
args = parser.parse_args()


img_size = args.input_size
device = torch.device('cuda')


# Load the model
ppnet = model.construct_MGProto(base_architecture=args.base_architecture,
                                pretrained=True,
                                img_size=args.input_size,
                                prototype_shape=args.prototype_shape,
                                num_classes=args.nb_classes,
                                prototype_activation_function=args.prototype_activation_function,
                                add_on_layers_type=args.add_on_layers_type,
                                sz_embedding=args.aux_emb_sz,
                                mem_capacity=args.mem_sz,
                                mine_K=args.mine_level,
                                )
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

rets = ppnet.load_state_dict(torch.load("./R50_104nopush0.8224-28*28.pth"), strict=False)
print(rets)


# test set
train_dataset = datasets.ImageFolder(
    '/mnt/c/copy/a0619/CUB_200_2011/dataset/train/',
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=False)


# test set
test_dataset = datasets.ImageFolder(
    '/mnt/c/copy/a0619/CUB_200_2011/dataset/test_full/',
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=False)

class_specific = True
accu, test_results = tnt.test(model=ppnet_multi, dataloader=(test_loader, ), class_specific=class_specific, )



ppnet_multi.eval()
projectset_img0_path = train_loader.dataset.samples[0][0]
project_path = os.path.split(os.path.split(projectset_img0_path)[0])[0].split("dataset")[0]
parts_loc_path = os.path.join(project_path, "parts/part_locs.txt")
parts_name_path = os.path.join(project_path, "parts/parts.txt")
imgs_id_path = os.path.join(project_path, "images.txt")
cubthreshold = 0.5


epoch = 100
args.wshape = 28
args.image_size = 224
args.log_dir = './purity-R50/'
makedir(args.log_dir)


print("\n\nEvaluating cub prototypes for training set", flush=True)
log, logclose = create_logger(log_filename=os.path.join(args.log_dir, 'trainset.log'))
csvfile_topk = get_topk_cub(ppnet_multi, train_loader, 10, 'train_' + str(epoch), device, args)
eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path, 'train_topk_' + str(epoch),  args, log)

csvfile_all = get_proto_patches_cub(ppnet_multi, train_loader, 'train_all_' + str(epoch), device, args, threshold=cubthreshold)
eval_prototypes_cub_parts_csv(csvfile_all, parts_loc_path, parts_name_path, imgs_id_path, 'train_all_thres' + str(cubthreshold) + '_' + str(epoch), args, log)
logclose()


print("\n\nEvaluating cub prototypes for test set", flush=True)
log, logclose = create_logger(log_filename=os.path.join(args.log_dir, 'testset.log'))
csvfile_topk = get_topk_cub(ppnet_multi, test_loader, 10, 'test_' + str(epoch), device, args)
eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path, 'test_topk_' + str(epoch), args, log)

csvfile_all = get_proto_patches_cub(ppnet_multi, test_loader, 'test_' + str(epoch), device, args, threshold=cubthreshold)
eval_prototypes_cub_parts_csv(csvfile_all, parts_loc_path, parts_name_path, imgs_id_path, 'test_all_thres' + str(cubthreshold) + '_' + str(epoch), args, log)
logclose()


