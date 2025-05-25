import os
import model
import torch
import argparse
from utils.interpretability import evaluate_consistency
import push, model, train_and_test as tnt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import settings
from utils.preprocess import mean, std
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--data_set', default='CUB2011', type=str)
parser.add_argument('--data_path', type=str, default='/mnt/c/copy/data/CUB_200_2011/CUB_200_2011/')
parser.add_argument('--nb_classes', type=int, default=200)
parser.add_argument('--test_batch_size', type=int, default=80)  # 30

# Model
parser.add_argument('--base_architecture', type=str, default='vgg19')
parser.add_argument('--input_size', default=224, type=int, help='images input size')
parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
parser.add_argument('--prototype_activation_function', type=str, default='log')
parser.add_argument('--add_on_layers_type', type=str, default='regular')
parser.add_argument('--aux_emb_sz', type=int, default=32)
parser.add_argument('--mem_sz', type=int, default=800, help="memory capacity")
parser.add_argument('--mine_level', type=int, default=20, help="number of mining levels")
parser.add_argument('--resume', type=str)
args = parser.parse_args()


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


ckpt = './V19_180nopush0.7881.pth'
ppnet.load_state_dict(torch.load(ckpt), strict=False)

consistency_score = evaluate_consistency(ppnet, args)
print('Consistency Score: {:.2f}%'.format(consistency_score))

    
