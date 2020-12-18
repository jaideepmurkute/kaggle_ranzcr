from PIL import Image
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Force PIL to load such cases - avoids exception in CenterCrop
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2

class cassava_dataset():
    def __init__(self, args, dataset_type, transform):
        self.num_classes = args.num_classes
        self.choice = args.choice
        self.transform = transform
        self.dataset_type = dataset_type.lower()
        self.execution_environment = args.execution_environment.lower()

        if self.execution_environment == 'local':
            self.root_path = './data'
        elif self.execution_environment == 'kaggle':
            self.root_path = '../input/cassava-leaf-disease-classification'

        self.num_folds = args.num_folds
        self.model_store_dir = args.model_store_dir

        self.img_paths = []
        self.labels = []

        self.target_cols = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal',
                       'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
                       'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

        if self.dataset_type == 'train' or self.dataset_type == 'validation':
            train_imgs = os.listdir(self.root_path + "/train/")
            train_label = pd.read_csv(self.root_path + "/train.csv")
            pids = train_label['StudyInstanceUID'].to_numpy()
            train_label = train_label[self.target_cols].to_numpy()
            train_label_dict = dict()
            for i in range(pids.shape[0]):
                train_label_dict[pids[i]] = train_label[i, :]
            # ----------------------------
            val_idx_dict = dict()
            train_idx_dict = dict()
            skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=args.seed)
            train_label_int = np.array(train_label[:, 1]).astype(np.int)
            for fold_id, (trn_idx, val_idx) in enumerate(skf.split(np.arange(len(train_imgs)), train_label_int)):
                train_idx_dict[fold_id] = trn_idx
                val_idx_dict[fold_id] = val_idx

            train_idx = train_idx_dict[args.fold_id]
            val_idx = val_idx_dict[args.fold_id]
            # ----------------------------
            # all_idx = np.arange(train_label.shape[0])
            # np.random.shuffle(all_idx)
            # train_sample_counts = int(0.8 * all_idx.shape[0])
            # val_sample_counts = all_idx.shape[0] - train_sample_counts
            # assert train_sample_counts + val_sample_counts == all_idx.shape[0]
            # train_idx = all_idx[:train_sample_counts]
            # val_idx = all_idx[train_sample_counts:]
            # assert len(train_idx) + len(val_idx) == all_idx.shape[0]
            # ----------------------------

            if self.dataset_type == 'train':
                for i in range(len(train_idx)):
                    self.img_paths.append(self.root_path + '/train/' + train_imgs[train_idx[i]])
                    self.labels.append(train_label_dict[train_imgs[train_idx[i]].split('.jpg')[0]])
                # if self.choice == 1:
                #     class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(self.num_classes), y=self.labels)
                #     if self.execution_environment == 'local':
                #         np.save(file=self.model_store_dir + '/train_class_weights.npy', arr=class_weights)

            elif self.dataset_type == 'validation':
                img_names = []
                for i in range(len(val_idx)):
                    self.img_paths.append(self.root_path + '/train/' + train_imgs[val_idx[i]])
                    img_names.append(train_imgs[val_idx[i]])
                    self.labels.append(train_label_dict[train_imgs[val_idx[i]].split('.jpg')[0]])
                if self.execution_environment == 'local':
                    np.save(file=args.model_store_dir+'/valid_images.npy', arr=img_names)

        elif self.dataset_type == 'test':
            test_data_path = self.root_path + '/test'
            test_imgs = os.listdir(test_data_path)
            self.img_paths = []
            for i in range(len(test_imgs)):
                self.img_paths.append(test_data_path + '/' + test_imgs[i])
            self.labels = [0]

        print("len(self.img_paths): ", len(self.img_paths))
        if self.dataset_type != 'test':
            print("len(self.labels): ", len(self.labels))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.array(image)
        if self.choice == 3 or self.choice == 5:
            image = self.transform(image)  # k-transformer function will handle this
        else:
            image = self.transform(image=image)['image']

        if self.dataset_type == 'test':
            label = 0
        else:
            label = self.labels[index]

        return image, label
