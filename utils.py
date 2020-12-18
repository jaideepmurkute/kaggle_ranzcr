import sys
import shutil
import random

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, f1_score
import torch
import albumentations.augmentations.transforms as album_aug_trans
import albumentations.pytorch.transforms as album_pt_trans
from albumentations import Compose
from dataset import *
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import seaborn as sns
import pandas as pd


def plot_tsne(args, all_embeddings_train, all_GTs_train, all_embeddings_val, all_GTs_val):
    from MulticoreTSNE import MulticoreTSNE as TSNE
    print("all_embeddings_train.shape: ", all_embeddings_train.shape)
    print("all_GTs_train.shape: ", all_GTs_train.shape)
    all_samples = all_embeddings_train
    all_labels = all_GTs_train

    num_classes = len(np.unique(all_labels))
    palette = sns.color_palette("deep", num_classes)

    for learning_rate in [10, 50, 100]:
        for perplexity in [50, 100, 150]:
            tsne = TSNE(n_components=2, verbose=1, learning_rate=learning_rate,
                        n_iter=5000, random_state=21,
                        n_iter_without_progress=1000, n_jobs=8, perplexity=perplexity)
            if all_samples.shape[1] == 2:
                tsne_results = all_samples
            else:
                tsne_results = tsne.fit_transform(all_samples)

            legend_name = 'class-id'
            df = pd.DataFrame(tsne_results, columns=['x-coord', 'y-coord'])
            df[legend_name] = all_labels
            sns.scatterplot(
                x="x-coord", y="y-coord",
                hue=legend_name,
                palette=sns.color_palette("Paired", num_classes),  # "hls"
                data=df,
                legend='full',
                alpha=0.5
            )
            plt.title("Penultimate Embeddings")
            plt.savefig(args.plot_store_dir + '/tsne/cc_class_id_lr_{}_perplex_{}.png'.format(learning_rate, perplexity), dpi=500)
            plt.close()


def print_progress(args, epoch, logger, dataset_type='validation'):
    print("Performance on {} dataset: ".format(dataset_type))
    if dataset_type == 'train':
        print("Epoch: {}/{} \t Loss: {:.8f} \t Accuracy: {:.8f} \t AUROC: {:.8f}".format(epoch, args.num_epochs,
            logger.per_epoch_train_loss[-1], logger.per_epoch_train_accuracy[-1], logger.per_epoch_train_auroc[-1]))
    else:
        print("Epoch: {}/{} \t Loss: {:.8f} \t Accuracy: {:.8f} \t AUROC: {:.8f}".format(epoch, args.num_epochs,
            logger.per_epoch_val_loss[-1], logger.per_epoch_val_accuracy[-1], logger.per_epoch_val_auroc[-1]))


def housekeeping(args):
    if torch.cuda.is_available():
        print("PyTorch is using GPU: ", torch.cuda.get_device_name(0))
    else:
        print("GPU not found ...using CPU ...")

    cuda_available = torch.cuda.is_available()
    args.device = torch.device("cuda" if cuda_available else "cpu")

    use_gpu_id = 0  # Set which GPU ID to use
    if cuda_available:
        if torch.cuda.device_count() == 1:
            use_gpu_id = 0

    args.gpu_id = use_gpu_id
    torch.cuda.set_device(args.gpu_id)

    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # -------------------------------------
    if args.execution_environment == 'local':
        model_store_home = './model_store'
        if not os.path.exists(model_store_home):
            os.mkdir(model_store_home)

        args.model_store_dir = model_store_home + '/' + args.model_name
        args.log_store_dir = args.model_store_dir + '/logs'
        args.plot_store_dir = args.model_store_dir + '/plots'

        if not os.path.exists(args.model_store_dir):
            os.mkdir(args.model_store_dir)
        if not os.path.exists(args.log_store_dir):
            os.mkdir(args.log_store_dir)
        if not os.path.exists(args.plot_store_dir):
            os.mkdir(args.plot_store_dir)

        # files_to_backup - absolute or relative path (with extension, if it's not a directory).
        if args.choice == 1:
            files_to_backup = []
            for file in os.listdir('./'):
                if os.path.isfile(os.path.join('./', file)) and '.py' in file:
                    files_to_backup.append('./' + file)

            for file_path in files_to_backup:
                try:
                    if os.path.isfile(file_path) or os.path.isdir(file_path):
                        file_name = file_path.split('/')[-1]
                        backup_path = args.model_store_dir + '/' + file_name
                        print("Copying {}   to   {}".format(file_path, backup_path))
                        shutil.copy2(file_path, backup_path)
                    else:
                        print("could not backup file {}, File not found at mentioned path ...".format(file_path))
                except Exception as e:
                    print("Exception occurred during creating backup code files... ")
                    print(e)
    elif args.execution_environment == 'kaggle':
        root_path = '../input/'+args.external_dataset_name+'/model_store'
        args.model_store_dir = '../input/'+args.external_dataset_name+'/model_store'
        args.log_store_dir = root_path + '/logs'
        args.plot_store_dir = root_path + '/plots'


def get_embedding_size(args):
    if args.model_arch == 'densenet169':
        args.embedding_size = 1664
    elif args.model_arch == 'densenet201':
        args.embedding_size = 1920
    elif args.model_arch == 'resnet18':
        args.embedding_size = 512
    elif args.model_arch == 'resnet34':
        args.embedding_size = 512
    elif args.model_arch == 'resnet50':
        args.embedding_size = 2048
    elif args.model_arch == 'wide_resnet50':
        args.embedding_size = 2048
    elif args.model_arch == 'resnet101':
        args.embedding_size = 2048
    elif args.model_arch == 'efficientnet_b0':
        args.embedding_size = 1280
    elif args.model_arch == 'efficientnet_b1':
        args.embedding_size = 1280
    elif args.model_arch == 'efficientnet_b2':
        args.embedding_size = 1408
    elif args.model_arch == 'efficientnet_b3':
        args.embedding_size = 1536
    elif args.model_arch == 'efficientnet_b4':
        args.embedding_size = 1792
    else:
        print("Architecture name not recognized.")
        exit()


def get_reconstruction_loss(recon, data):
    # (N, C, H, W)
    criterion = torch.nn.MSELoss(reduction='mean')
    return criterion(recon, data)


def get_kld(mu, logvar):
    kld = - 0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
    return kld


def get_elbo_loss(args, mu, logvar, reconstruction, data):
    KLD = get_kld(mu, logvar)
    NLL = get_reconstruction_loss(reconstruction, data)
    # ELBO = KLD + NLL
    return KLD, NLL


def compute_f1_score(args, all_predictions, all_GTs):
    all_pred_classes = np.argmax(all_predictions, axis=1)
    f1_score_val = f1_score(all_GTs, all_pred_classes, average='weighted')
    return f1_score_val


def get_confusion_matrix(args, all_predictions, all_GTs):
    return confusion_matrix(all_predictions, all_GTs)


def get_scheduler(args, optimizer):
    if args.scheduler_type == 'plateau':
        # if plateau found, new_lr = lr * factor
        if args.monitor == 'accuracy' or args.monitor == 'f1_score':
            mode = 'max'
        elif args.monitor == 'loss':
            mode = 'min'

        return ReduceLROnPlateau(optimizer, mode=mode, patience=3, verbose=True, factor=0.5)  # 0.7
    elif args.scheduler_type == 'cyclic':
        return CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=800, mode='triangular2')
    elif args.scheduler_type == 'CAWR':
        # return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)

        from cosine_annnealing_warmup import CosineAnnealingWarmUpRestarts
        return CosineAnnealingWarmUpRestarts(optimizer, T_0=300, T_mult=1, eta_max=3e-4, T_up=10, gamma=0.9)


def freeze_densenet(model, active_flag, freeze_key):
    if active_flag is False:
        print("Freezing densenet ...")
    else:
        print("Unfreezing densenet ...")
    '''
    i = 0
    for key in model.state_dict().keys():
        if freeze_key in key:
            try:
                model.state_dict()[key].requires_grad = active_flag
            except:
                pass  # Skip action for layers that do not require gradient; like BatchNorm, transition_blocks, pooling
        i += 1
        if i < 50:
            print("key: {} \t requires_grad: {}".format(key, model.state_dict()[key].requires_grad))
    print(model.densenet.features.denseblock1.denselayer1.conv2.weight.requires_grad)
    print(model.state_dict()['densenet.features.denseblock1.denselayer1.conv2.weight'].requires_grad)
    '''

    for param in model.parameters():
        param.requires_grad = active_flag
    model.fc1.weight.requires_grad = True
    model.fc1.bias.requires_grad = True
    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True
    
    return model


def is_bad_grad(grad_output):
    """
    Checks if gradients have exploded (NaN or greater than 1e6).
    Returns True, if exploded; False, otherwise.
    """
    grad_output = grad_output.data
    return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()


def activate_dropout(m):
    """
    To activate dropout during test time.
    Should be called with model.apply(activate_dropout) right after model.eval(), if dropout needed.
    Will recursively go through model layers and put dropout layers into train() mode.
    """
    if type(m) == nn.Dropout:
        m.train()


def gradient_clipper(args, model):
    try:
        if args.monitor_grad_explosion:
            grad_check_list = [model.fc1.weight.grad, model.fc1.bias.grad, model.compute_mean.weight.grad, model.compute_mean.bias.grad, 
                               model.compute_log_sigma.weight.grad, model.compute_log_sigma.bias.grad, model.classifier.weight.grad, 
                               model.classifier.bias.grad]
            for k, gradient_tensor in enumerate(grad_check_list):
                if gradient_tensor is not None:  # skips grad-check if layer mentioned in the list is not being used in current model configuraion or training stage
                    if is_bad_grad(gradient_tensor):
                        torch.nn.utils.clip_grad_norm(gradient_tensor)  # clip gradients to tensor's L2-norm, in-place
                        print("gradients exploded and clipped in {}-th tensor in grad_check_list ...")
    except Exception as e:
        print("Exception occurred in gradient monitor ...")
        print(e)
    
    return model


def get_optimizer(args, model, load_from_checkpoint=False):
    optimizer_type = args.optimizer_type.lower()
    supported_optimizers = ['adam', 'sgd']
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.l2_penalty)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(list(model.parameters()), lr=args.lr, weight_decay=args.l2_penalty, nesterov=True)
    else:
        print("Unsupported optimizer type requested: {}. Supported optimizers: {}".format(optimizer_type, supported_optimizers))
        return None
    
    if load_from_checkpoint:
        checkpoint = torch.load(args.model_store_dir + '/' + args.model_name + '_optimizer.pt')
        optimizer.load_state_dict(checkpoint)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value, torch.Tensor):
                    state[key] = value.to(args.device)
        
        print("Optimizer loaded from the checkpoint at: ", args.model_store_dir + '/' + args.model_name + '.pt')
    
    return optimizer
    

def get_model_reference(args, load_from_checkpoint=False):
    get_embedding_size(args)

    if args.model_arch == 'densenet169':
        from densenet import densenet
        model = densenet(args)
    elif args.model_arch == 'resnet18':
        from resnet18 import resnet
        model = resnet(args)
    elif args.model_arch == 'resnet34':
        from resnet34 import resnet
        model = resnet(args)
    elif args.model_arch == 'resnet50':
        from resnet import resnet
        model = resnet(args)
    elif args.model_arch == 'wide_resnet50':
        from wide_resnet_50 import resnet
        model = resnet(args)
    elif args.model_arch == 'resnet101':
        from resnet101 import resnet
        model = resnet(args)
    elif args.model_arch == 'densenet201':
        from densenet201 import densenet201
        model = densenet201(args)
    elif args.model_arch == 'efficientnet_b4':
        from efficientnet_b4 import efficientnet_b4
        model = efficientnet_b4(args)
    elif args.model_arch == 'efficientnet_b3':
        # from efficientnet_b3 import efficientnet_b3
        from efficientnet_b3_timm import efficientnet_b3
        model = efficientnet_b3(args)

    model.to(args.device)
    if load_from_checkpoint:
        if args.choice == 1 and args.is_contrastive:
            model_weights_path_to_load = args.model_store_dir + '/' + args.model_name + '_best_pretrained.pt'
        else:
            model_weights_path_to_load = args.model_store_dir + '/' + args.model_name + '_best.pt'
        model.load_state_dict(torch.load(model_weights_path_to_load, map_location='cuda:'+str(args.gpu_id)))
        print("Model loaded with weights from: ", model_weights_path_to_load)

    print("Model parameters stats: ")
    print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
    print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    return model


class k_transformer:
    def __init__(self, args, transform):
        self.transform = transform
        self.num_inference_transforms = args.num_inference_transforms

    def __call__(self, x):
        transformed_data = []
        for i in range(self.num_inference_transforms):
            # transformed_data.append(self.transform(x))  # .float()
            transformed_data.append(self.transform(image=x)['image'].unsqueeze(0))  # .float()
        # print("len(transformed_data): ", len(transformed_data))
        # transformed_data = torch.cat(transformed_data, dim=0)
        # print("catted transformed_data.size(): ", transformed_data.size())
        return transformed_data


def get_data_loader(args, dataset_type=None, training=False):
    assert dataset_type is not None
    
    dataset_type = dataset_type.lower()
    if training:
        shuffle_flag = {'train': True, 'validation': False, 'test': False}
    else:
        shuffle_flag = {'train': False, 'validation': False, 'test': False}
    
    if training:
        # transforms_list = [album_aug_trans.LongestMaxSize(400),
        #                    album_aug_trans.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.1, rotate_limit=45, p=0.5),
        #                    album_aug_trans.Flip(p=0.5),
        #                    album_aug_trans.RandomCrop(224, 224),
        #                    album_aug_trans.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        #                    album_aug_trans.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        #                    album_aug_trans.Normalize(),
        #                    album_pt_trans.ToTensor()]
        #
        transforms_list = [album_aug_trans.LongestMaxSize(512),
                           album_aug_trans.Resize(256, 256),
                        album_aug_trans.Transpose(p=0.5),
                        album_aug_trans.HorizontalFlip(p=0.5),
                        album_aug_trans.VerticalFlip(p=0.5),
                        album_aug_trans.ShiftScaleRotate(p=0.5),
                        album_aug_trans.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.5),
                        album_aug_trans.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                        album_aug_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                        album_aug_trans.CoarseDropout(p=0.5),
                        album_aug_trans.Cutout(p=0.5),

                        album_pt_trans.ToTensorV2(p=1.0)]
    else:
        # transforms_list = [album_aug_trans.LongestMaxSize(400),
        #                    album_aug_trans.CenterCrop(224, 224),
        #                    album_aug_trans.Normalize(),
        #                    album_pt_trans.ToTensor()]
        #
        transforms_list = [album_aug_trans.LongestMaxSize(512),
                           album_aug_trans.Resize(256, 256),
                            album_aug_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                            album_pt_trans.ToTensorV2(p=1.0),]

    if args.choice == 3 or args.choice == 5:
        # transforms_list = [album_aug_trans.LongestMaxSize(400),
        #                    album_aug_trans.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.1, rotate_limit=45, p=0.5),
        #                    album_aug_trans.Flip(p=0.5),
        #                    album_aug_trans.RandomCrop(224, 224),
        #                    album_aug_trans.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        #                    album_aug_trans.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        #                    album_aug_trans.Normalize(),
        #                    album_pt_trans.ToTensor()]

        transforms_list = [album_aug_trans.LongestMaxSize(512),
                           album_aug_trans.Resize(256, 256),
                           album_aug_trans.Transpose(p=0.5),
                           album_aug_trans.HorizontalFlip(p=0.5),
                           album_aug_trans.VerticalFlip(p=0.5),
                           album_aug_trans.ShiftScaleRotate(p=0.5),
                           album_aug_trans.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                           album_aug_trans.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                           album_aug_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                           album_aug_trans.CoarseDropout(p=0.5),
                           album_aug_trans.Cutout(p=0.5),
                           album_pt_trans.ToTensorV2(p=1.0)]

    # if ('efficientnet' in args.model_arch) and args.use_advprop:
    #     # transforms_list.extend([transforms.Lambda(lambda img: img * 2.0 - 1.0)])
    #     transforms_list.extend([album_aug_trans.Lambda(lambda img: img * 2.0 - 1.0)])
    # else:
    #     # transforms_list.extend([transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    #     transforms_list.extend([album_aug_trans.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    # transform = transforms.Compose(transforms_list)
    transform = Compose(transforms_list)

    if args.choice == 3 or args.choice == 5:
        transform = k_transformer(args, transform)

    dataset = cassava_dataset(args, dataset_type, transform)

    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                              shuffle=shuffle_flag[dataset_type], num_workers=0, pin_memory=True)
    
    return data_loader



class Logger():
    def __init__(self, args):
        self.log_store_dir = args.log_store_dir
        self.plot_store_dir = args.plot_store_dir
        self.min_loss = sys.maxsize
        self.max_accuracy = -1
        self.max_f1_score = -1
        self.max_auroc_score = -1
        self.best_epoch_number = 0

        self.per_epoch_train_loss = []
        self.per_epoch_train_accuracy = []
        self.per_epoch_train_auroc = []

        self.per_epoch_val_loss = []
        self.per_epoch_val_accuracy = []
        self.per_epoch_val_auroc = []

        self.per_epoch_lr = []

    def save_log(self):
        np.save(file=self.log_store_dir + '/per_epoch_train_loss.npy', arr=np.array(self.per_epoch_train_loss))
        np.save(file=self.log_store_dir + '/per_epoch_train_accuracy.npy', arr=np.array(self.per_epoch_train_accuracy))
        np.save(file=self.log_store_dir + '/per_epoch_train_auroc.npy', arr=np.array(self.per_epoch_train_auroc))

        np.save(file=self.log_store_dir + '/per_epoch_val_loss.npy', arr=np.array(self.per_epoch_val_loss))
        np.save(file=self.log_store_dir + '/per_epoch_val_accuracy.npy', arr=np.array(self.per_epoch_val_accuracy))
        np.save(file=self.log_store_dir + '/per_epoch_train_auroc.npy', arr=np.array(self.per_epoch_train_auroc))

        np.save(file=self.log_store_dir + '/per_epoch_lr.npy', arr=np.array(self.per_epoch_lr))

    def load_log(self):
        self.per_epoch_train_loss = list(np.load(self.log_store_dir + '/per_epoch_train_loss.npy', allow_pickle=True))
        self.per_epoch_train_accuracy = list(np.load(self.log_store_dir + '/per_epoch_train_accuracy.npy', allow_pickle=True))
        self.per_epoch_train_auroc = list(np.load(self.log_store_dir + '/per_epoch_train_auroc.npy', allow_pickle=True))

        self.per_epoch_val_loss = list(np.load(self.log_store_dir + '/per_epoch_val_loss.npy', allow_pickle=True))
        self.per_epoch_val_accuracy = list(np.load(self.log_store_dir + '/per_epoch_val_accuracy.npy', allow_pickle=True))
        self.per_epoch_val_auroc = list(np.load(self.log_store_dir + '/per_epoch_val_auroc.npy', allow_pickle=True))

        try:
            self.per_epoch_lr = list(np.load(self.log_store_dir + '/per_epoch_lr.npy', allow_pickle=True))
        except:
            self.per_epoch_lr = []

    def plot_log(self):
        try:
            print("max(self.per_epoch_lr): ", max(self.per_epoch_lr))
            print("min(self.per_epoch_lr): ", min(self.per_epoch_lr))

            plt.figure()
            plt.plot(self.per_epoch_lr, color='green')
            plt.title("Learning rate log")
            plt.savefig(self.plot_store_dir + '/lr_log.png')
            plt.close()
        except:
            pass

        plt.figure()
        plt.plot(self.per_epoch_train_loss, color='green', label='train')
        plt.plot(self.per_epoch_val_loss, color='red', label='validation')
        plt.title("BCE Loss")
        plt.legend()
        plt.savefig(self.plot_store_dir + '/loss_log.png')
        plt.close()

        plt.figure()
        plt.plot(self.per_epoch_train_accuracy, color='green', label='train')
        plt.plot(self.per_epoch_val_accuracy, color='red', label='validation')
        plt.title("Accuracy")
        plt.legend()
        plt.savefig(self.plot_store_dir + '/accuracy_log.png')
        plt.close()

        plt.figure()
        plt.plot(self.per_epoch_train_auroc, color='green', label='train')
        plt.plot(self.per_epoch_val_auroc, color='red', label='validation')
        plt.title("AU_ROC")
        plt.legend()
        plt.savefig(self.plot_store_dir + '/AU_ROC.png')
        plt.close()

