import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
import timm


class efficientnet_b3(nn.Module):
    def __init__(self, args):
        super(efficientnet_b3, self).__init__()
        self.choice = args.choice
        self.num_classes = args.num_classes  # number of output classes for model
        self.device = args.device
        self.mixup_alpha = args.mixup_alpha
        self.is_contrastive = args.is_contrastive
        self.projection_size = args.projection_size
        self.use_advprop = args.use_advprop
        self.in_planes = 64

        # Convert names like 'efficientnet_b0' to 'efficientnet-b0'
        if '_' in args.model_arch:
            architecture_name = args.model_arch.replace('_', '-')

        # if self.use_imagenet_initialization:
        if self.choice == 1:
            self.model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True)
        else:
            self.model = timm.create_model('tf_efficientnet_b3_ns', pretrained=False)

        self.model = nn.Sequential(*list(self.model.children())[:-1])  # throw away last FC layer and swish activation

        if self.is_contrastive:
            self.projection_head = nn.Linear(in_features=args.embedding_size, out_features=self.projection_size)

        # self.fc1 = nn.Linear(in_features=args.embedding_size, out_features=512)
        self.classifier = nn.Linear(in_features=args.embedding_size, out_features=self.num_classes)

    def perform_mixup(self, args, data, label):
        self.mixup_lambdas = torch.from_numpy(
            np.random.beta(args.mixup_alpha, args.mixup_alpha, size=(data.size()[0], 1))).to(self.device).float()
        self.shuffled_idx_mm = torch.randperm(data.size()[0])

        if label is not None:
            mix_label = (self.mixup_lambdas * label) + ((1 - self.mixup_lambdas) * label[self.shuffled_idx_mm, :].float())
        else:
            mix_label = None

        if len(data.size()) == 4:
            self.mixup_lambdas = self.mixup_lambdas.unsqueeze(2).unsqueeze(2)
        mix_data = (self.mixup_lambdas * data) + ((1 - self.mixup_lambdas) * data[self.shuffled_idx_mm, :])

        return mix_data, mix_label

    def rand_bbox(self, size, lam):
        if len(size) > 2:
            W = size[2]
            H = size[3]
            if 'torch' in str(lam.dtype):
                cut_rat = np.sqrt(1. - lam.cpu().numpy())
            else:
                cut_rat = np.sqrt(1. - lam)
            # cut_rat = lam.cpu().numpy()
            cut_w = (W * cut_rat).astype(np.int)
            cut_h = (H * cut_rat).astype(np.int)
            # uniform
            # if not self.use_passed_lambda:
            # self.cx = np.random.randint(W)
            # self.cy = np.random.randint(H)

            cut_w = np.reshape(np.array(cut_w), (-1,))
            cut_h = np.reshape(np.array(cut_h), (-1,))

            self.cx = []
            for i in range(len(cut_w)):
                self.cx.append(np.random.choice(np.arange(cut_w[i] // 2, W - cut_w[i] // 2)))
            self.cx = np.array(self.cx)

            self.cy = []
            for i in range(len(cut_h)):
                self.cy.append(np.random.choice(np.arange(cut_h[i] // 2, H - cut_h[i] // 2)))
            self.cy = np.array(self.cy)

            # print("self.use_passed_lambda: ", self.use_passed_lambda)
            bbx1 = np.clip(self.cx - cut_w // 2, 0, W)
            bby1 = np.clip(self.cy - cut_h // 2, 0, H)
            bbx2 = np.clip(self.cx + cut_w // 2, 0, W)
            bby2 = np.clip(self.cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def perform_cutmix(self, args, data, label):
        if not self.use_passed_lambda:
            self.shuffled_idx_mcm = torch.randperm(data.size()[0])
        else:
            self.shuffled_idx_mcm = self.shuffled_idx_mm
        if not self.use_passed_lambda:
            self.mixup_lambdas = np.random.beta(args.mixup_alpha, args.mixup_alpha, size=(data.size()[0], 1))

        if len(data.size()) > 2:
            mix_data = data
            # bbx1, bby1, bbx2, bby2 = self.rand_bbox(mix_data.size(), self.mixup_lambdas)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(mix_data.size(), self.mixup_lambdas)

            # adjust lambda to exactly match pixel ratio
            # self.mixup_lambdas = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
            bbx1 = torch.from_numpy(bbx1).to(self.device)
            bby1 = torch.from_numpy(bby1).to(self.device)
            bbx2 = torch.from_numpy(bbx2).to(self.device)
            bby2 = torch.from_numpy(bby2).to(self.device)
            for i in range(bbx1.size()[0]):
                mix_data[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = data[self.shuffled_idx_mcm[i], :, bbx1[i]:bbx2[i],
                                                                   bby1[i]:bby2[i]]
            # self.mixup_lambdas = torch.from_numpy(self.mixup_lambdas).to(self.device)
            self.mixup_lambdas = torch.from_numpy(self.mixup_lambdas).to(self.device)
        elif len(data.size()) == 2:
            self.mixup_lambdas = torch.from_numpy(self.mixup_lambdas).to(self.device).float()
            self.mixup_lambdas_orig = self.mixup_lambdas

            mix_data = data[self.shuffled_idx_mcm, :]
            # drop_features_count = torch.round(data.size()[1] * (self.prev_mixup_lambda)).squeeze(1).to(dtype=torch.long, device=self.device)
            drop_features_count = torch.round(data.size()[1] * (self.mixup_lambdas)).squeeze(1).to(dtype=torch.long,
                                                                                                   device=self.device)

            drop_init_index = torch.Tensor(data.size()[0]).random_(0, data.size()[1]).to(dtype=torch.long,
                                                                                         device=self.device)
            invalid_start_index = drop_init_index + drop_features_count > data.size()[0]
            drop_init_index[invalid_start_index] = drop_init_index[invalid_start_index] - drop_features_count[
                invalid_start_index]
            drop_features_mask_sample_1 = torch.ones_like(data).to(self.device)
            # drop_features_mask_sample_1[:, drop_init_index:(drop_init_index+drop_features_count)] = 0
            for i in range(drop_features_mask_sample_1.size()[0]):
                drop_features_mask_sample_1[i, drop_init_index[i]:drop_init_index[i] + drop_features_count[i]] = 0
                drop_features_mask_sample_1 = drop_features_mask_sample_1.to(torch.long)
            mix_data = (data * drop_features_mask_sample_1) + (mix_data * (1 - drop_features_mask_sample_1))
        if len(self.mixup_lambdas.size()) == 4:
            self.mixup_lambdas = self.mixup_lambdas.squeeze(2).squeeze(2)
        # mix_label = (self.mixup_lambdas * label) + ((1-self.mixup_lambdas) * label[self.shuffled_idx_mcm, :].double())
        mix_label = (self.mixup_lambdas * label) + ((1 - self.mixup_lambdas) * label[self.shuffled_idx_mcm, :].double())

        if 'torch' not in str(self.mixup_lambdas.dtype):
            self.mixup_lambdas = torch.from_numpy(self.mixup_lambdas)
        self.mixup_lambdas_orig = self.mixup_lambdas
        if len(self.mixup_lambdas_orig.size()) > 2:
            self.mixup_lambdas_orig = self.mixup_lambdas_orig.squeeze(2).squeeze(2)
        if 'torch' not in str(self.shuffled_idx_mcm.dtype):
            self.shuffled_idx_mcm = torch.from_numpy(self.shuffled_idx_mcm)

        if args.gt_plus_mix:
            return torch.cat([data, mix_data], dim=0), torch.cat([label, mix_label])
        else:
            del data, label
            return mix_data, mix_label

    def forward(self, args, x, label, enable_mixup=False):
        if enable_mixup:
            self.mixup_layer = np.random.choice(np.arange(0, 29))  # Choose one layer for mixup, randomly.
            # self.mixup_layer = np.random.choice([5, 6])  # Choose one layer for mixup, randomly.
        else:
            self.mixup_layer = None
            self.mixup_lambdas = None

        output = x

        output = self.model[2](self.model[1](self.model[0](output)))
        # -----------------
        # loop over all the blocks in efficientnet-b4
        for i in range(7):
            if enable_mixup and self.mixup_layer == i+1:  # offsetting to (i+1) for input mixup
                if args.mixup_method == 'manifold_mixup':
                    output, label = self.perform_mixup(args, output, label)
                if args.mixup_method == 'manifold_cutmix':
                    output, label = self.perform_cutmix(args, output, label)
            output = self.model[3][i](output)
        # -----------------
        output = self.model[4](output)  # conv_head
        output = self.model[5](output)  # Adaptive average pooling
        output = self.model[6](output)  # Dropout - Note, default drop prob. is 0.3
        output = self.model[7](output)  # Dropout - Note, default drop prob. is 0.3
        # print("output.size(): ", output.size())
        # output = F.dropout(output, p=0.1, inplace=True)
        # output = output.squeeze(2).squeeze(2)

        # -----------------
        if enable_mixup and self.mixup_layer == 27:
            if args.mixup_method == 'manifold_mixup':
                output, label = self.perform_mixup(args, output, label)
            if args.mixup_method == 'manifold_cutmix':
                output, label = self.perform_cutmix(args, output, label)

        embeddings = output

        output = self.classifier(output)

        return output, embeddings, label
