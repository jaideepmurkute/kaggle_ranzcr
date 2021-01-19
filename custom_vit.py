import timm
import torch.nn as nn
import torch
import numpy as np
from vit_pytorch import ViT

class CustomViT(nn.Module):
    def __init__(self, args):
        super(CustomViT, self).__init__()
        self.choice = args.choice
        self.num_classes = args.num_classes  # number of output classes for model
        self.device = args.device
        self.mixup_alpha = args.mixup_alpha
        self.is_contrastive = args.is_contrastive
        self.input_size = args.input_size

        self.model = ViT(image_size=self.input_size, patch_size=44, num_classes=11, dim=1024, depth=12, heads=32, mlp_dim=2048,
                         dropout=0.05, emb_dropout=0.05)

        # print(self.model)
        # exit(0)

    def forward(self, args, x, label, cat_label=None, enable_mixup=False, training=False):
        gammas = []
        if enable_mixup:
            self.mixup_layer = np.random.choice(np.arange(0, 1))
        else:
            self.mixup_layer = None
            self.mixup_lambdas = None
        output = x

        if enable_mixup and self.mixup_layer == 0:
            if args.mixup_method == 'manifold_mixup':
                output, label = self.perform_mixup(args, output, label)
            if args.mixup_method == 'manifold_cutmix':
                output, label = self.perform_cutmix(args, output, label)

        output = self.model(output)
        embeddings = output

        cat_label_output = None

        return output, embeddings, label, cat_label_output, cat_label, gammas