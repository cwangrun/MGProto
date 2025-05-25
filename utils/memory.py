import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    def __init__(self, num_classes, dim_feature, capacity=1024, mode='fix_class', fix_length_mult=4):
        super(MemoryBank, self).__init__()
        self.num_classes = num_classes
        self.dim_feature = dim_feature
        self.capacity = capacity
        self.mode = mode
        self.pull = getattr(self, 'pull_' + mode)
        self.fix_length_mult = fix_length_mult

        assert self.capacity % self.num_classes == 0, (self.capacity, self.num_classes)
        cap_cls = self.capacity // self.num_classes
        for i in range(self.num_classes):
            self.register_buffer('cls%d' % i, torch.zeros((cap_cls, self.dim_feature), dtype=torch.float32))
        self.register_buffer('mem_len', torch.zeros((self.num_classes,), dtype=torch.int64))

        self.debug_counter = 0
        self.num_last_push = 0
        self.num_last_pull = 0

    def print_debug_info(self):
        self.debug_counter += 1
        if (self.debug_counter % 200) == 0:
            print('mem debug', self.cls0.device, self.num_last_push, self.num_last_pull)

    @torch.no_grad()
    def push(self, feature, label):
        assert feature.dim() == 2, feature.size()
        assert label.dim() == 1, label.size()
        assert feature.size(0) == label.size(0), (feature.size(), label.size())
        if feature.size(0) == 0: return

        feature = feature.detach()
        label = label.detach()
        num_last_push = 0
        last_push_items = 0
        cap_cls = self.capacity // self.num_classes

        with torch.no_grad():
            class_mask = F.one_hot(label, self.num_classes).T.bool()  # [15, b]
            class_exists = class_mask.max(1)[0]

            for i in range(self.num_classes):
                if not class_exists[i]: continue
                feat = feature[class_mask[i]]
                if feat.size(0) > cap_cls:
                    selected = torch.randperm(feat.size(0), device=feat.device)
                    feat = feat[selected[:cap_cls]]
                tgt_data = getattr(self, 'cls%d' % i)

                num_remove = self.mem_len[i] + feat.size(0) - cap_cls
                if num_remove <= 0:
                    tgt_data[self.mem_len[i]: self.mem_len[i] + feat.size(0)] = feat
                    self.mem_len[i] += feat.size(0)
                elif num_remove >= self.mem_len[i]:
                    tgt_data[:] = feat
                    self.mem_len[i] = feat.size(0)
                elif self.mem_len[i] >= cap_cls:
                    torch.cat([tgt_data[num_remove:].clone(), feat], 0, out=tgt_data)
                else:
                    tgt_data[:] = torch.cat([tgt_data[:self.mem_len[i]], feat], 0)[-cap_cls:]
                    self.mem_len[i] = cap_cls
                assert tgt_data.size(0) == cap_cls and self.mem_len[i] <= cap_cls, (tgt_data.size(), self.mem_len[i], num_remove)

                num_last_push += feat.size(0)
                last_push_items += 1

        self.num_last_push = float(num_last_push) / (last_push_items + 1e-10)

    @torch.no_grad()
    def pull_fix_class(self, label):   # one-hot label
        assert list(label.size()) == [self.num_classes], label.size()
        indices = torch.nonzero(label.detach(), as_tuple=False).flatten()

        out_label, out_data = [], []
        for i in indices:
            if self.mem_len[i] == 0: continue
            tgt_data = getattr(self, 'cls%d' % i)
            out_data.append(tgt_data[:self.mem_len[i]].clone())
            out_label.append(torch.full((self.mem_len[i],), i, dtype=torch.int64, device=tgt_data.device))

        if len(out_data) == 0:
            return None, None

        out_data = torch.cat(out_data, 0)
        out_label = torch.cat(out_label, 0)
        self.num_last_pull = out_data.size(0)
        return out_data, out_label

    def pull_fix_length(self, label):
        assert label.dim() == 2 and label.size(1) == self.num_classes, label.size()
        cap_cls = self.capacity // self.num_classes
        pull_num = cap_cls * self.fix_length_mult

        label = label.detach()
        indices = torch.nonzero(label.max(0)[0], as_tuple=False).flatten()
        cand_data, cand_label = [], []
        for i in indices:
            if self.mem_len[i] == 0: continue
            tgt_data = getattr(self, 'cls%d' % i)
            cand_data.append(tgt_data[:self.mem_len[i]].clone())
            cand_label.append(torch.full((self.mem_len[i],), i, dtype=torch.int64, device=tgt_data.device))
        if len(cand_data) == 0:
            return None, None

        cand_data = torch.cat(cand_data, 0)
        cand_label = torch.cat(cand_label, 0)
        #assert cand_data.size(0) == cand_label.size(0), (cand_data.size(), cand_labe.size())
        #assert cand_label.dim() == 1, cand_label.size()
        #assert cand_data.dim() == 2, cand_data.size()

        cand_label_oh = F.one_hot(cand_label, self.num_classes).float()
        cand_mask = torch.matmul(label.float(), cand_label_oh.T) > 0.999
        if cand_mask.sum(1).min() < 1:
            return None, None

        out_data = []
        for i in range(label.size(0)):
            data = cand_data[cand_mask[i]]
            while data.size(0) < pull_num:
                #data = torch.cat([data, data], 0)
                data = data.repeat(2, 1)
            if data.size(0) > pull_num:
                data = data[torch.randperm(data.size(0))[:pull_num]]
            out_data.append(data.unsqueeze(1))
        out_data = torch.cat(out_data, 1).contiguous()
        self.num_last_pull = pull_num
        return out_data, None

    @torch.no_grad()
    def pull_all(self):
        out_data = []
        out_label = []
        for i in range(self.num_classes):
            if self.mem_len[i] == 0:
                continue
            tgt_data = getattr(self, 'cls%d' % i)
            out_data.append(tgt_data[:self.mem_len[i]].clone())
            out_label.append(torch.full((self.mem_len[i],), i, dtype=torch.int64, device=tgt_data.device))

        if len(out_data) == 0:
            return None, None

        out_data = torch.cat(out_data, 0)
        out_label = torch.cat(out_label, 0)
        return out_data, out_label
