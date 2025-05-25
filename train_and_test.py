import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import list_of_distances
import wandb
import numpy as np


def _training(model, dataloader, optimizer=None, aux_criterion=None, use_mine=False, update_GMM=False, class_specific=True, coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''

    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0

    total_cross_entropy = 0
    total_mine_cost = 0
    total_aux_cost = 0

    for i, (image, label) in enumerate(dataloader):

        image = image.cuda()
        target = label.cuda()

        grad_req = torch.enable_grad()
        with grad_req:

            output, x_auxiliary = model(image, target)

            # compute loss
            if use_mine:
                mine_loss = sum([F.cross_entropy(output[:, :, k], target) for k in range(1, output.shape[2])]) / (output.shape[2] - 1)
            else:
                mine_loss = torch.tensor(0.0).cuda()
            cross_entropy = F.cross_entropy(output[:, :, 0], target)
            aux_loss = aux_criterion(x_auxiliary, target)

            # evaluation statistics
            predicted = torch.argmax(output[:, :, 0].data, dim=1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_mine_cost += mine_loss.item()
            total_aux_cost += aux_loss.item()

            # compute gradient and do SGD step
            loss = coefs['crs_ent'] * cross_entropy + coefs['mine'] * mine_loss + coefs['aux'] * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EM update
            if update_GMM and (model.module.queue.mem_len.sum() > 0):
                if model.module.iteration_counter % model.module.update_interval == 0:
                    model.module.update_GMM()
            full_mem_ratio = (model.module.queue.mem_len == model.module.capacity_pc).sum().float() / model.module.num_classes

            if i % 20 == 0:
                print(
                    '{} {} \tLoss: {:.4f} \tL_ce: {:.4f} \tL_mine: {:.4f} \tL_aux: {:.4f} \tMem_ratio {:.2f} \tAcc: {:.4f}'.format(
                        i, len(dataloader), loss.item(), cross_entropy.item(), mine_loss.item(), aux_loss.item(), full_mem_ratio,
                        n_correct / (n_examples + 0.000001) * 100,
                    ))

                wandb.log({
                    "Train Total Loss": loss.item(),
                    "Train CE Loss": cross_entropy.item(),
                    "Train Mine Loss": mine_loss.item(),
                    "Train Aux Loss": aux_loss.item(),
                    "Train Acc": n_correct / (n_examples + 0.000001) * 100,
                    "Full Mem Ratio": full_mem_ratio,
                })

        del image
        del target
        del output
        del predicted

    end = time.time()

    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tmine: \t{0}'.format(total_mine_cost / n_batches))
    log('\taux: \t{0}'.format(total_aux_cost / n_batches))

    results_loss = {'cross_entropy': total_cross_entropy / n_batches,
                    'mine_loss': total_mine_cost / n_batches,
                    'aux_loss': total_aux_cost / n_batches,
                    'acc': n_correct/n_examples,
                    }
    return n_correct / n_examples, results_loss


def _testing(model, dataloader, optimizer=None, class_specific=True, coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''

    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0

    total_cross_entropy = 0

    for i, (image, label) in enumerate(dataloader[0]):    # in-distributions
        image = image.cuda()
        target = label.cuda()

        grad_req = torch.no_grad()
        with grad_req:

            output, _ = model(image, None)

            # compute loss
            cross_entropy = F.cross_entropy(output[:, :, 0], target)

            # evaluation statistics
            predicted = torch.argmax(output[:, :, 0].data, dim=1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()

        del image
        del target
        del output
        del predicted

    end = time.time()

    wandb.log({
        "Test Acc": n_correct / n_examples * 100,
    })

    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\ttest acc: \t\t{0}%'.format(n_correct / n_examples * 100))

    p = model.module.prototype_means.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    results_loss = {'cross_entropy': total_cross_entropy / n_batches,
                    'p_avg_pair_dist': p_avg_pair_dist,
                    'acc': n_correct/n_examples,
                    }
    return n_correct / n_examples, results_loss


def _testing_with_OoD(model, dataloader, optimizer=None, class_specific=True, coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''

    n_examples = 0
    n_correct = 0
    n_batches = 0

    output_prob_all = []
    for i, (image, label) in enumerate(dataloader[0]):   # in-distributions
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.no_grad()
        with grad_req:

            output, _ = model(input, None)

            output_prob = output[:, :, 0].exp()    # p(x|c)
            predicted = torch.argmax(output[:, :, 0].data, dim=1)
            output_prob_all.append(output_prob)

            # evaluation statistics
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            n_batches += 1

    log('\tTest Acc: \t{0}'.format(n_correct / n_examples * 100))
    wandb.log({
        "Test Acc": n_correct / n_examples * 100,
    })

    output_prob_all = torch.cat(output_prob_all)
    prob_sum_over_c = output_prob_all.sum(dim=1).cpu().numpy()
    ood_thresh = np.percentile(prob_sum_over_c, 5)

    pred_ood1 = []
    for j, (image, label) in enumerate(dataloader[1]):   # out-distributions
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.no_grad()
        with grad_req:

            output, _ = model(input, None)

            output_prob = output[:, :, 0].exp()    # p(x|c)
            pred_ood1.append(output_prob.mean(dim=1) > ood_thresh)
    pred_ood1 = torch.cat(pred_ood1)
    FPR95_1 = pred_ood1.sum() / len(pred_ood1)

    pred_ood2 = []
    for j, (image, label) in enumerate(dataloader[2]):  # out-distributions
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.no_grad()
        with grad_req:
            output, _ = model(input, None)

            output_prob = output[:, :, 0].exp()  # p(x|c)
            pred_ood2.append(output_prob.mean(dim=1) > ood_thresh)
    pred_ood2 = torch.cat(pred_ood2)
    FPR95_2 = pred_ood2.sum() / len(pred_ood2)

    log('\tFPR95_1: \t{0}'.format(FPR95_1))
    log('\tFPR95_2: \t{0}'.format(FPR95_2))

    results = {'FPR95_1': FPR95_1,
               'FPR95_2': FPR95_2,
              }

    wandb.log({'FPR95_1': FPR95_1,
               'FPR95_2': FPR95_2,
              })
        
    return n_correct / n_examples, results


def train(model, dataloader, optimizer, aux_criterion=None, use_mine=False, update_GMM=False, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    log('\ttrain')
    model.train()
    return _training(model=model, dataloader=dataloader, optimizer=optimizer, aux_criterion=aux_criterion, use_mine=use_mine, update_GMM=update_GMM,
                     class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _testing(model=model, dataloader=dataloader, optimizer=None, class_specific=class_specific, log=log)
    # return _testing_with_OoD(model=model, dataloader=dataloader, optimizer=None, class_specific=class_specific, log=log)


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    # model.module.prototype_means.requires_grad = True
    # for p in model.module.last_layer.parameters():
    #     p.requires_grad = False
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    # model.module.prototype_means.requires_grad = True
    # for p in model.module.last_layer.parameters():
    #     p.requires_grad = True
    log('\tjoint')