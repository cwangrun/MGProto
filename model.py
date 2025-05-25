import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, \
    resnet152_features
from models.densenet_features import densenet121_features, densenet161_features, densenet169_features, \
    densenet201_features
from models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, \
    vgg16_bn_features, \
    vgg19_features, vgg19_bn_features

from utils.receptive_field import compute_proto_layer_rf_info_v2
from utils.helpers import list_of_distances
from einops import rearrange, repeat
from utils.memory import MemoryBank
import numpy as np
import math


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}


def l2_normalize(x, dim):
    return F.normalize(x, p=2, dim=dim)


def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, prototype_class_identity=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prototype_class_identity = prototype_class_identity

        self.weight = nn.Parameter(torch.ones((out_features, in_features), **factory_kwargs), requires_grad=False)

    def forward(self, input, prototypes_to_keep_with_negative=None):

        negative_one_weights_locations = 1 - torch.t(self.prototype_class_identity)
        assert torch.sum(self.weight.data[negative_one_weights_locations == 1]) == 0

        if prototypes_to_keep_with_negative is not None:    # pruning
            assert torch.sum(self.weight.data[prototypes_to_keep_with_negative == 0]) == 0

        return F.linear(input, self.weight, bias=None)


class MGProto(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 sz_embedding=32,
                 mem_capacity=800,
                 mine_K=20):

        super(MGProto, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        self.prototype_activation_function = prototype_activation_function  # log

        assert (self.num_prototypes % self.num_classes == 0)
        # onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert (current_out_channels == self.prototype_shape[1]) 
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # for R50 pretrained on iNaturalist
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                # nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                # nn.Sigmoid()
            )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(first_add_on_layer_in_channels, sz_embedding)

        self.prototype_means = nn.Parameter(torch.rand(self.num_classes, self.num_prototypes_per_class, self.prototype_shape[1]), requires_grad=True)
        self.prototype_means.data.copy_(l2_normalize(self.prototype_means.data, dim=2))

        self.init_sigma = 1 / math.sqrt(2*math.pi)
        self.prototype_covs = nn.Parameter(torch.ones(self.num_classes, self.num_prototypes_per_class, self.prototype_shape[1]) * self.init_sigma, requires_grad=False)

        self.last_layer = NonNegLinear(self.num_prototypes, self.num_classes, prototype_class_identity=self.prototype_class_identity)

        if init_weights:
            self.initialize_weights()

        self.mine_T = mine_K
        self.capacity_pc = mem_capacity
        self.queue = MemoryBank(
            self.num_classes,
            self.prototype_shape[1],
            self.capacity_pc * self.num_classes,
            mode='all'
        )
        self.memory_updated_cls = torch.zeros((self.num_classes)).bool()
        self.iteration_counter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.prototype_optimizer = None

        self.update_interval = 1
        self.num_em_loop = 3
        self.alpha = 0.1
        self.tau = 0.990

    def conv_features(self, x):

        x = self.features(x)
        x_add = self.add_on_layers(x)
        x_avg = self.gap(x)

        x_avg = x_avg.view(x_avg.size(0), -1)
        x_embed = self.embedding(x_avg)
        x_embed = l2_normalize(x_embed, dim=1)

        return x_add, x_embed

    def global_max_pooling_gmm_topT(self, similarities, conv_features, mine_T=20):
        similarities_new = rearrange(similarities, 'b c p h w -> b (c p) (h w)')
        val_largest, ind_largest = torch.topk(similarities_new, mine_T, dim=2)   # [b, 2000, mine_T]
        max_similarities = val_largest

        max_index_ret = []
        select_ret = []
        feat_dim = conv_features.shape[1]
        conv_features = conv_features.view(conv_features.shape[0], conv_features.shape[1], -1)
        for iii in range(mine_T):
            max_index = ind_largest[:, :, iii].unsqueeze(-1)
            max_index_new = torch.cat([max_index] * feat_dim, dim=2).permute(0, 2, 1)   # [b, 64, 2000]
            select = torch.gather(conv_features, dim=2, index=max_index_new)
            select = rearrange(select, 'b d (c p) -> b c p d', b=similarities.shape[0], c=similarities.shape[1])
            max_index = rearrange(max_index.squeeze(2), 'b (c p) -> b c p', b=similarities.shape[0], c=similarities.shape[1])
            select_ret.append(select)
            max_index_ret.append(max_index)

        return max_similarities, torch.stack(select_ret).permute(1, 2, 3, 4, 0), torch.stack(max_index_ret).permute(1, 2, 3, 0)

    def forward(self, x, gt):
        base_feature, x_auxiliary = self.conv_features(x)
        base_feature = l2_normalize(base_feature, dim=1)
        _feat = rearrange(base_feature, 'b c h w -> (b h w) c')

        _log_prob = self.compute_log_prob(_feat)    # [b*h*w, 200, 10]
        _log_prob = rearrange(_log_prob, "(b h w) c k -> b c k h w", b=base_feature.shape[0], h=base_feature.shape[2])  # [b, 200, 10, h, w]
        _prob = _log_prob.exp()

        prototype_activations, max_feat, max_index = self.global_max_pooling_gmm_topT(_prob, base_feature, mine_T=self.mine_T)
        if gt is not None:
            prototypes_of_wrong_class = 1 - torch.t(self.prototype_class_identity[:, gt.cpu()])
            for k in range(1, prototype_activations.shape[2]):
                prototype_activations[:, :, k][prototypes_of_wrong_class == 1] = prototype_activations[:, :, 0][prototypes_of_wrong_class == 1]
        final_probs = torch.stack([self.last_layer(prototype_activations[:, :, k]) for k in range(prototype_activations.shape[2])], dim=2)    # [b, 200]

        # enqueue only top-1 features
        max_feat = max_feat[:, :, :, :, 0].unsqueeze(-1)
        max_index = max_index[:, :, :, 0].unsqueeze(-1)

        if gt is not None:
            # update memory
            unique_c_list = gt.unique().int()
            for c in unique_c_list:
                mask_c = gt == c
                temp_index = max_index[mask_c, c.item()]
                enq_feat = max_feat[mask_c, c.item()].detach()
                if len(enq_feat) == 0: continue
                temp_index = rearrange(temp_index, "b p k -> b (p k)")
                enq_feat = rearrange(enq_feat, "b p d k -> b (p k) d")
                # only enqueue unique feature vectors of a training sample
                unique_enq_feat = []
                for b in range(temp_index.shape[0]):
                    unique_ind = torch.unique(temp_index[b])
                    for v in unique_ind:
                        position = torch.where(temp_index[b] == v)[0][0]
                        f_vec = enq_feat[b, position, :].unsqueeze(0)
                        unique_enq_feat.append(f_vec)
                unique_enq_feat = torch.cat(unique_enq_feat)

                if len(unique_enq_feat) == 0: continue
                self.queue.push(unique_enq_feat, torch.cat([c.unsqueeze(0)]*unique_enq_feat.shape[0], dim=0).long())
                self.memory_updated_cls[c] = True

            self.iteration_counter += 1

        return torch.log(final_probs), x_auxiliary

    def compute_log_prob(self, _fea, n_block=4, c_block=1, eps=0e-10):
        _n_group = _fea.shape[0] // n_block
        _c_group = self.num_classes // c_block

        assert (_fea.shape[0] % n_block == 0) and (self.num_classes % c_block == 0)

        _probs = torch.zeros((_fea.shape[0], self.num_classes, self.num_prototypes_per_class), device=_fea.device)
        for _c in range(0, self.num_classes, _c_group):
            _c_means = self.prototype_means[_c:_c + _c_group].detach()
            _c_covariances = self.prototype_covs[_c:_c + _c_group].detach()
            _prob_c = torch.zeros((_fea.shape[0], _c_means.shape[0] * _c_means.shape[1]), device=_fea.device)

            for _n in range(0, _fea.shape[0], _n_group):
                scale_diag = _c_covariances.view(-1, self.prototype_shape[1])
                location = _c_means.view(-1, self.prototype_shape[1])
                diff = _fea[_n:_n + _n_group, None, ...] - location
                _prob_c[_n:_n + _n_group] = -0.5 * self.prototype_shape[1] * math.log(2 * math.pi) - scale_diag.log().sum(-1) - 0.5 * (diff / (scale_diag + eps)).pow(2).sum(-1)
            _c_probs = _prob_c.view(_prob_c.shape[0], -1, self.num_prototypes_per_class)
            _probs[:, _c:_c + _c_group, :] = _c_probs
        return _probs

    def update_GMM(self, ):
        last_layer = self.last_layer.weight.data.t().clone()     # [2000, 200]
        mem_feat, mem_label = self.queue.pull()

        for _c in torch.arange(0, self.num_classes):

            if not self.memory_updated_cls[_c]: continue
            pi_c = last_layer[:, _c][self.prototype_class_identity[:, _c] == 1]

            _c = _c if isinstance(_c, int) else _c.item()
            self.memory_updated_cls[_c] = False

            if (mem_label == _c).sum() < self.capacity_pc: continue

            _mem_feat_c = mem_feat[mem_label == _c]   # (N, d)
            pi_old = pi_c.unsqueeze(0).unsqueeze(2)
            for i in range(self.num_em_loop):
                log_likelihood_old, log_resp = self._e_step(_mem_feat_c, self.prototype_means[_c].unsqueeze(0).data.detach(), self.prototype_covs[_c].unsqueeze(0).data.detach(), pi_old.data.detach())
                pi, mean, var = self._m_step_diversified(_mem_feat_c, log_resp.detach(), self.prototype_means[_c].unsqueeze(0), self.prototype_covs[_c].unsqueeze(0), pi_old)
                # log_likelihood = self._score(_mem_feat_c, mean, var, pi)
                pi_old = momentum_update(old_value=pi_old, new_value=pi, momentum=self.tau, debug=False)
                last_layer[:, _c][self.prototype_class_identity[:, _c] == 1] = pi_old.squeeze()

        self.last_layer.weight = nn.Parameter(last_layer.t(), requires_grad=False)
        assert self.memory_updated_cls.sum() == 0

    def _e_step(self, x, mu, var, pi, eps=1e-10):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self._check_size(x)

        weighted_log_prob = self._estimate_log_prob(x, mu, var) + torch.log(pi + eps)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm   # log form of responsibility/posterior

        return torch.mean(log_prob_norm), log_resp

    def _estimate_log_prob(self, x, mu, var, eps=1e-10):   # log form of posterior
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        x = self._check_size(x)
        fea_dim = x.shape[-1]
        log_p = torch.sum(((x - mu) / (var + eps)).pow(2), dim=2, keepdim=True)
        log_sigma = torch.sum(torch.log(var + eps), dim=2, keepdim=True)

        return -0.5 * fea_dim * math.log(2 * math.pi) - log_sigma - 0.5 * log_p

    def _m_step(self, x, log_resp, eps=1e-10):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self._check_size(x)
        resp = torch.exp(log_resp)    # responsibility/posterior

        # additive smoothing to prevent assigning to a single Gaussian
        resp = (resp + self.alpha) / (resp + self.alpha).sum(1, keepdim=True)

        pi = torch.sum(resp, dim=0, keepdim=True) + eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi
        x2 = (resp * x * x).sum(0, keepdim=True) / pi
        mu2 = mu * mu
        xmu = (resp * mu * x).sum(0, keepdim=True) / pi
        var = x2 - 2 * xmu + mu2 + eps
        var = var.sqrt()

        pi = pi / x.shape[0]

        return pi, mu, var

    def _m_step_diversified(self, x, log_resp, mu_old, var_old, pi_old, eps=1e-10, lamda=1.0):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """

        x = self._check_size(x)
        resp = torch.exp(log_resp)    # responsibility/posterior

        # additive smoothing to prevent assigning to a single Gaussian
        resp = (resp + self.alpha) / (resp + self.alpha).sum(1, keepdim=True)

        pi = torch.sum(resp, dim=0, keepdim=True) + eps

        log_likelihood = self._estimate_log_prob(x, mu_old, var_old) + torch.log(pi_old + eps)    # [1000, 10, 1]
        weighted_log_likelihood = - (resp * log_likelihood).sum(1).mean(0).squeeze()    # [1000, 10, 1]

        pair_dist = list_of_distances(torch.squeeze(mu_old), torch.squeeze(mu_old))     # [10, 10]
        I_operator = 1 - torch.eye(mu_old.size(1), mu_old.size(1)).cuda()
        diversity_cost = (torch.exp(-pair_dist) * I_operator).sum() / I_operator.sum()
        gmm_loss = weighted_log_likelihood + lamda * diversity_cost

        self.prototype_optimizer.zero_grad()
        gmm_loss.backward()
        self.prototype_optimizer.step()

        pi = pi / x.shape[0]

        return pi, mu_old, var_old

    def _score(self, x, mu, var, pi, as_average=True, eps=1e-10):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              torch.Tensor (1)
            (or)
            per_sample_score:   torch.Tensor (n)

        """
        weighted_log_prob = self._estimate_log_prob(x, mu, var) + torch.log(pi + eps)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

        if as_average:
            return per_sample_score.mean()
        else:
            return torch.squeeze(per_sample_score)

    def _check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)
        return x

    def push_forward(self, x):
        base_feature, _ = self.conv_features(x)
        base_feature = l2_normalize(base_feature, dim=1)
        _feat = rearrange(base_feature, 'b c h w -> (b h w) c')
        _log_prob = self.compute_log_prob(_feat)
        _log_prob = rearrange(_log_prob, "(b h w) c k -> b c k h w", b=base_feature.shape[0], h=base_feature.shape[2])
        _prob = _log_prob.exp()
        _prob = rearrange(_prob, "b c k h w -> b (c k) h w")
        distances = -_prob
        return base_feature, distances

    def set_last_layer_incorrect_connection(self, incorrect_strength):

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1./self.num_prototypes_per_class
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(correct_class_connection * positive_one_weights_locations + incorrect_class_connection * negative_one_weights_locations)

    def initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.kaiming_normal_(self.embedding.weight, mode='fan_out')
        nn.init.constant_(self.embedding.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=0.0)

    def prune_prototypes_topM(self, top_M=1):
        '''
        prototypes_to_keep: True or False to indicates the prototypes to be kept or removed
        '''

        pos_mask = torch.t(self.prototype_class_identity).cuda()
        proto_prior = self.last_layer.weight[pos_mask == 1].view(self.num_classes, -1)
        threshold_prune, _ = torch.topk(proto_prior, top_M, dim=1)   # [200, top_M]

        self.prototypes_to_keep = torch.stack([proto_prior[c] >= threshold_prune[c][-1] for c in range(self.num_classes)])
        self.prototypes_to_keep_with_negative = torch.stack([self.last_layer.weight[c] >= threshold_prune[c][-1] for c in range(self.num_classes)])
        assert (self.prototypes_to_keep.sum(1) >= 1).all()  # at least 1 prototypes per class

        # changing self.last_layer in place
        for c in range(self.num_classes):
            self.last_layer.weight.data[c][self.prototypes_to_keep_with_negative[c] == 0] = 0.0


def construct_MGProto(base_architecture, pretrained=True, img_size=224,
                      prototype_shape=(2000, 128, 1, 1), num_classes=200,
                      prototype_activation_function='log',
                      add_on_layers_type='bottleneck',
                      sz_embedding=32,
                      mem_capacity=1000,
                      mine_K=10):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return MGProto(features=features,
                   img_size=img_size,
                   prototype_shape=prototype_shape,
                   proto_layer_rf_info=proto_layer_rf_info,
                   num_classes=num_classes,
                   init_weights=True,
                   prototype_activation_function=prototype_activation_function,
                   add_on_layers_type=add_on_layers_type,
                   sz_embedding=sz_embedding,
                   mem_capacity=mem_capacity,
                   mine_K=mine_K,
                   )

