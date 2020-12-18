import argparse
import random
import shutil
from utils import *
from tqdm import tqdm
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_argparser():
    parser = argparse.ArgumentParser(description='Skin Lesion Classifier')
    parser.add_argument("--seed", nargs='?', type=int, default=0,
                        help="Random seed for init.")
    parser.add_argument("--model_name", nargs='?', type=str, default='vdx_vanilla_aux_1',
                        help="The name model to be saved with; Default: 'model_name'")
    parser.add_argument("--choice", nargs='?', type=int, default=-1, help="If -1, will ask to enter input; otherwise pass value through command line 1] Train  2] Test")

    parser.add_argument("--use_mixup", nargs='?', type=str2bool, default=False, help="Whether to use mixup.")
    parser.add_argument("--mixup_alpha", nargs='?', type=float, default=2.0, help="Mixup alpha")
    parser.add_argument("--mixup_method", nargs='?', type=str, default='manifold_mixup', help="manifold_mixup cutmix")

    parser.add_argument("--fold_id", nargs='?', type=int, default=0, help="Which fold ID to train/test.")
    parser.add_argument("--num_folds", nargs='?', type=int, default=5, help="How many folds")

    parser.add_argument("--model_arch", nargs='?', type=str, default='densenet169', help="Architecture name in torchvision - densenet169 or resnet50 etc.")
    parser.add_argument("--use_advprop", nargs='?', type=str2bool, default=False, help="Whether to use adversarial propagated weights for efficientnets")

    parser.add_argument("--use_class_weighted_loss", nargs='?', type=str2bool, default=False, help="Whether to use Class frequency weighted loss")

    parser.add_argument("--num_inference_transforms", nargs='?', type=int, default=1, help="How many inputs transforms to perform to generate prediction."
                                                                                           "valid for choice=3.")

    parser.add_argument("--bce_loss_scalar", nargs='?', type=float, default=1.0, help="Loss scalar for 5 class classification loss")
    parser.add_argument("--num_classes", nargs='?', type=int, default=11,
                        help="Num classes; Default: 80")

    parser.add_argument("--lr", type=float, nargs='?', default=1e-4, help="learning rate; Default:5e-4")
    parser.add_argument("--l2_penalty", nargs='?', type=float, default=1e-6, help="l2 penalty; Default=0.0")
    parser.add_argument("--optimizer_type", nargs='?', type=str, default='adam', help="Choose optimizer: 'adam' or 'SGD'; Default: 'adam'.")
    parser.add_argument("--num_epochs", nargs='?', type=int, default=3, help="Number of epochs for training. Default: 50")

    parser.add_argument("--batch_size", nargs='?', type=int, default=32,
                        help="batch size; Default: 32")

    parser.add_argument("--use_lr_scheduler", nargs='?', type=str2bool, default=False,
                        help="Whether to use learning rate schedule with ReduceLROnPlateau; Default: False")

    parser.add_argument("--scheduler_type", nargs='?', type=str, default='plateau',
                        help="Select from plateau(ReduceLROnPlateau) or cyclic or CAWR(cosine annealing with warm restart); Default: plateau")

    parser.add_argument("--validate_during_training", nargs='?', type=str2bool, default=True,
                        help="To validate performance during training; Default: True")
    parser.add_argument("--monitor_grad_explosion", nargs='?', type=str2bool, default=False,
                        help="Monitor and clip very large gradients in FC layers, for stable training; Default: True")

    parser.add_argument("--monitor", nargs='?', type=str, default='accuracy',
                        help="Which metric to monitor during training to select best model: loss, accuracy, f1_score; Default: accuracy")

    parser.add_argument("--no_progress_patience", nargs='?', type=int, default=15,
                        help="Number of epochs to tolerate, without any progress on metric; Default: 5")
    parser.add_argument("--generate_submission_file", nargs='?', type=str2bool, default=False,
                        help="Whether to write preds in the submission file; Default: False")

    parser.add_argument("--execution_environment", nargs='?', type=str, default='local',
                        help="Local or Kaggle - To adjust input/output paths; Default: local")

    parser.add_argument("--warm_startup_epochs", nargs='?', type=int, default=0,
                        help="How many epochs to freeze conv layers and just train FC layer?; Default: 0")
    parser.add_argument("--is_contrastive", nargs='?', type=str2bool, default=False,
                        help="Is this model trained with contrastive pretrainined - to load projection head; Default: False")

    parser.add_argument("--projection_size", nargs='?', type=int, default=128,
                        help="projection_size for contrastive pretraining; Default: 128")

    return parser.parse_args()


args = get_argparser()

# -------------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score
def get_auroc_score(y_true, y_pred):
    return roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))

def update_conv_layers(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True


def run_epoch(args, logger, data_loader, model, optimizer=None, scheduler=None, training=False, dataset_type='', epoch_num=0):
    all_predictions = None
    all_embeddings = None
    all_GTs = None

    epoch_num_correct = 0
    epoch_loss_total = 0

    pbar = tqdm(data_loader)
    for batch_idx, (data, label) in enumerate(pbar):
        # save_image(data[-1].cpu(), 'im2.png')
        # exit()
        if args.choice == 3 or args.choice == 5:
            data = torch.cat(data, dim=0).squeeze(1)
            label = label.repeat(args.num_inference_transforms)

        # ------------------
        data = data.to(args.device)
        label = label.to(args.device).float()
        # ----
        if args.choice == 1 and args.use_mixup and dataset_type == 'train':
            enable_mixup = True
        else:
            enable_mixup = False

        with autocast():
            output, embedding, label = model(args, data, label, enable_mixup=enable_mixup)

            if args.choice == 3 or args.choice == 5:
                k_unlabeled_preds = torch.chunk(output, args.num_inference_transforms, dim=0)
                for i in range(args.num_inference_transforms):
                    if i == 0:
                        sum_k_unlabeled_preds = k_unlabeled_preds[i]
                    else:
                        sum_k_unlabeled_preds += k_unlabeled_preds[i]
                output = sum_k_unlabeled_preds / args.num_inference_transforms

                if args.choice == 3:
                    label = torch.chunk(label, args.num_inference_transforms, dim=0)[0]


            if args.choice != 5:
                loss = cls_criterion(output, label)
                # print("unscaled loss: ", loss.item())
                loss *= args.bce_loss_scalar
                # print("scaled loss: ", loss.item())
                preds = output.clone().sigmoid()
                preds[preds > 0.5] = 1
                preds[preds <= 0.5] = 0
                batch_correct = (preds == label).float().sum()
                epoch_num_correct += (preds == label).float().sum()
                epoch_loss_total += loss.item()
            else:
                batch_correct = 0
                loss = torch.Tensor([0])
                epoch_loss_total = 0
                epoch_num_correct = 0

        if training:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # loss.backward()
            if args.monitor_grad_explosion:
                model = gradient_clipper(args, model)

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

        # if args.choice != 1:
        if all_predictions is None:
            all_predictions = output.sigmoid().detach().cpu().numpy()
            all_embeddings = embedding.detach().cpu().numpy()
            all_GTs = label[:output.size()[0]].cpu().numpy()
        else:
            all_predictions = np.concatenate((all_predictions, output.sigmoid().detach().cpu().numpy()), axis=0)
            all_embeddings = np.concatenate((all_embeddings, embedding.detach().cpu().numpy()), axis=0)
            all_GTs = np.concatenate((all_GTs, label[:output.size()[0]].cpu().numpy()), axis=0)

        if args.use_lr_scheduler and args.choice == 1 and args.scheduler_type == 'CAWR' and training:
            logger.per_epoch_lr.append(optimizer.param_groups[0]['lr'])
            # scheduler.step(epoch_num + batch_idx / len(data_loader.dataset))
            scheduler.step()

        batch_acc = batch_correct/(data.size()[0] * args.num_classes)
        pbar.set_postfix({'acc': batch_acc.item()})

    au_roc = get_auroc_score(all_GTs, all_predictions)

    if training and dataset_type == 'train':
        logger.per_epoch_train_loss.append(epoch_loss_total / len(data_loader.dataset))
        logger.per_epoch_train_accuracy.append(epoch_num_correct / (len(data_loader.dataset)*args.num_classes) )
        logger.per_epoch_train_auroc.append(au_roc)

        if args.scheduler_type != 'CAWR' or not args.use_lr_scheduler:
            logger.per_epoch_lr.append(optimizer.param_groups[0]['lr'])

    else:
        logger.per_epoch_val_loss.append(epoch_loss_total / len(data_loader.dataset))
        logger.per_epoch_val_accuracy.append(epoch_num_correct / (len(data_loader.dataset)*args.num_classes)  )
        logger.per_epoch_val_auroc.append(au_roc)

    if args.use_lr_scheduler and args.choice == 1 and dataset_type == 'validation' and args.scheduler_type != 'CAWR':
        if args.monitor == 'accuracy':
            epoch_acc = epoch_num_correct / len(data_loader.dataset)
            scheduler.step(metrics=epoch_acc)
        if args.monitor == 'loss':
            scheduler.step(metrics=loss)

    return all_predictions, all_GTs, all_embeddings


def train_model(args, logger):
    model.train()
    if args.validate_during_training:
        print("Loading validation set ...")
        val_loader = get_data_loader(args, dataset_type='validation', training=False)
    
    for epoch in range(1, args.num_epochs+1):
        if epoch <= args.warm_startup_epochs:
            print("Freezing Conv Layers...")
            update_conv_layers(model, requires_grad=False)
        if epoch == args.warm_startup_epochs + 1:
            print("Unfreezing Conv Layers...")
            update_conv_layers(model, requires_grad=True)

        print("args.model_name: ", args.model_name)
        print("learning rate: ", optimizer.param_groups[0]['lr'])
        all_predictions, all_GTs, all_embeddings = run_epoch(args, logger, train_loader, model, optimizer, scheduler=scheduler,
                                                             training=True, dataset_type='train', epoch_num=epoch)
        print_progress(args, epoch, logger, dataset_type='train')

        if args.validate_during_training:
            model.eval()
            test_model(args, model, val_loader, dataset_type='validation')
            model.train()

        logger.save_log()

        is_best_epoch = False
        if args.validate_during_training:
            if args.monitor == 'loss':
                is_best_epoch = logger.per_epoch_val_loss[-1] < logger.min_loss
            elif args.monitor == 'au_roc':
                is_best_epoch = logger.per_epoch_val_auroc[-1] >= logger.max_auroc_score
            elif args.monitor == 'accuracy':
                is_best_epoch = logger.per_epoch_val_accuracy[-1] >= logger.max_accuracy

            if is_best_epoch:
                logger.min_loss = logger.per_epoch_val_loss[-1]
                logger.max_accuracy = logger.per_epoch_val_accuracy[-1]
                torch.save(model.state_dict(), args.model_store_dir + '/' + args.model_name + '_best.pt')
        else:
            if args.monitor == 'loss':
                is_best_epoch = logger.per_epoch_train_loss[-1] <= logger.min_loss
            elif args.monitor == 'au_roc':
                is_best_epoch = logger.per_epoch_train_auroc[-1] >= logger.max_auroc_score
            elif args.monitor == 'accuracy':
                is_best_epoch = logger.train_f1_score[-1] >= logger.max_accuracy

            if is_best_epoch:
                logger.min_loss = logger.per_epoch_train_loss[-1]
                logger.max_accuracy = logger.per_epoch_train_accuracy[-1]
                # logger.max_f1_score = logger.train_f1_score[-1]
                torch.save(model.state_dict(), args.model_store_dir + '/' + args.model_name + '_best.pt')

        torch.save(model.state_dict(), args.model_store_dir + '/' + args.model_name + '_last.pt')
        torch.save(optimizer.state_dict(), args.model_store_dir + '/' + args.model_name + '_optimizer_last.pt')
        
        if is_best_epoch:
            logger.best_epoch_number = epoch
            print("New best epoch found on validation set... Model and optimizer has been saved ...")
        else:
            if args.validate_during_training:
                print("This was not the best epoch....Last best epoch: {}  Last best loss: {:.6f}  Last best accuracy: {:.6f}".format(logger.best_epoch_number,
                                                                                                            logger.min_loss, logger.max_accuracy))
                print("Last best au_roc: {:.6f}".format(logger.max_auroc_score))
            else:
                print("This was not the best epoch....Last best epoch: {}  Last best loss: {:.6f}  Last best accuracy: {:.6f}".format(logger.best_epoch_number,
                                                                                                            logger.min_loss, logger.max_accuracy))
                print("Last best au_roc: {:.6f}".format(logger.max_auroc_score))

        if epoch - logger.best_epoch_number == args.no_progress_patience:
            print("Best epoch not found since epoch {}. Patience {} reached. Stopping training.".format(logger.best_epoch_number, args.no_progress_patience))
            break

        logger.save_log()
        
        print("-" * 100)


def test_model(args, model, data_loader, dataset_type, return_preds=False):
    model.eval()

    with torch.no_grad():
        all_predictions, all_GTs, all_embeddings = run_epoch(args, logger, data_loader, model, optimizer, scheduler=scheduler, training=False,
                                                             dataset_type='validation')
        print("all_GTs.shape: ", all_GTs.shape)
        print("all_predictions.shape: ", all_predictions.shape)
        au_roc = get_auroc_score(all_GTs, all_predictions)
        print("AU-ROC: {:.5f}".format(au_roc))

        if args.choice != 5:
            print_progress(args, 0, logger, dataset_type='validation')

    # if args.choice == 2:
    #     f1_score_val = compute_f1_score(args, all_predictions, all_GTs)
    #     print("Weighted F1 Score: {:.6f}".format(f1_score_val))
    #
    #     preds = np.argmax(all_predictions, axis=1)
    #     conf_mat = get_confusion_matrix(args, preds, all_GTs)
    #     print("conf_mat: ")
    #     print(conf_mat)
    #
    #     conf_mat_norm = conf_mat / np.sum(conf_mat, axis=1)
    #     df_cm = pd.DataFrame(conf_mat_norm, range(args.num_classes), range(args.num_classes))
    #     plt.figure(figsize=(10, 7))
    #     sns.set(font_scale=1.4)  # for label size
    #     sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')  # font size
    #     plt.savefig(args.plot_store_dir + '/conf_mat.png')
    #     plt.close()

    if return_preds:
        return all_predictions, all_GTs, all_embeddings


if __name__ == "__main__":
    if args.choice == -1:
        choice = int(input("Enter choice: 1] Train \t 2] Test \n 3] Prediction with TTA \t 4] Plot tsne \t 5] Test inference"))
    else:
        choice = args.choice

    args.model_name_orig = args.model_name

    args.model_name += str(args.fold_id)
    housekeeping(args)
    logger = Logger(args)

    if choice == 1:
        if args.is_contrastive:
            try:
                file_paths_to_backup = ['./contrastive/model_store/' + args.model_name + '/' + args.model_name + '_best.pt',
                                        './contrastive/model_store/' + args.model_name + '/' + args.model_name + '_last.pt']
                for file_path in file_paths_to_backup:
                    if os.path.isfile(file_path) or os.path.isdir(file_path):
                        file_name = file_path.split('/')[-1]
                        backup_file_name = file_name.split('.')[0] + '_pretrained.pt'
                        backup_path = args.model_store_dir + '/' + backup_file_name
                        print("Copying {}   to   {}".format(file_path, backup_path))
                        shutil.copy2(file_path, backup_path)
                    else:
                        print("could not backup file {}, File not found at mentioned path ...".format(file_path))
            except Exception as e:
                print("Exception occurred during creating backup code files... ")
                print(e)

        if args.is_contrastive:
            model = get_model_reference(args, load_from_checkpoint=True)
        else:
            model = get_model_reference(args)

        model.train()
        scaler = GradScaler()
        train_loader = get_data_loader(args, dataset_type='train', training=True)

        if args.use_class_weighted_loss:
            class_weights = np.load(args.model_store_dir + '/train_class_weights.npy', allow_pickle=True)
            print("class_weights: ", class_weights)
            train_class_probs = torch.from_numpy(class_weights).float().to(args.device)
        else:
            train_class_probs = None

        cls_criterion = nn.BCEWithLogitsLoss(reduction='sum', weight=train_class_probs)
        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)

        print("Training ........")
        train_model(args, logger)

    elif choice == 2:
        # housekeeping(args)
        logger = Logger(args)
        logger.load_log()
        logger.plot_log()
        cls_criterion = nn.BCEWithLogitsLoss(reduction='mean', weight=None)
        args.load_model_from_checkpoint = True
        args.batch_size = 32
        model = get_model_reference(args, load_from_checkpoint=True)
        scaler = GradScaler()
        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)
        model.eval()
        logger.load_log()
        dataset_type = 'validation'
        print("Evaluating on {} dataset.".format(dataset_type))
        data_loader = get_data_loader(args, dataset_type=dataset_type, training=False)
        all_predictions, all_GTs, all_embeddings = test_model(args, model, data_loader, dataset_type=dataset_type, return_preds=True)
        print("all_predictions.shape: ", all_predictions.shape)

    elif choice == 3:
        # inference with k-input-augmentations
        cls_criterion = nn.BCEWithLogitsLoss(reduction='mean', weight=None)
        print("Evaluating on {} dataset.")
        args.load_model_from_checkpoint = True
        args.batch_size = 8
        model = get_model_reference(args, load_from_checkpoint=True)
        scaler = GradScaler()
        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)
        model.eval()
        logger.load_log()
        logger.plot_log()

        if not os.path.exists(args.plot_store_dir + '/tsne'):
            os.mkdir(args.plot_store_dir + '/tsne')

        dataset_type = 'validation'
        print("Prediction {} dataset.".format(dataset_type))
        data_loader = get_data_loader(args, dataset_type=dataset_type, training=False)
        all_predictions_val, all_GTs_val, all_embeddings_val = test_model(args, model, data_loader, dataset_type=dataset_type, return_preds=True)

    elif choice == 4:
        cls_criterion = nn.BCEWithLogitsLoss(reduction='mean', weight=None)
        print("Evaluating on {} dataset.")
        args.load_model_from_checkpoint = True
        args.batch_size = 32
        model = get_model_reference(args, load_from_checkpoint=True)
        scaler = GradScaler()
        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)
        model.eval()
        logger.load_log()
        logger.plot_log()

        if not os.path.exists(args.plot_store_dir + '/tsne'):
            os.mkdir(args.plot_store_dir + '/tsne')

        dataset_type = 'validation'
        print("Prediction {} dataset.".format(dataset_type))
        data_loader = get_data_loader(args, dataset_type=dataset_type, training=False)
        all_predictions_val, all_GTs_val, all_embeddings_val = test_model(args, model, data_loader, dataset_type=dataset_type, return_preds=True)

        print("all_GTs_val.shape: ", all_GTs_val.shape)
        print("np.unique(all_GTs_val): ", np.unique(all_GTs_val))
        print("np.bincount(all_GTs_val): ", np.bincount(all_GTs_val.astype(np.int)))

        dataset_type = 'train'
        print("Prediction {} dataset.".format(dataset_type))
        data_loader = get_data_loader(args, dataset_type=dataset_type, training=False)
        all_predictions_train, all_GTs_train, all_embeddings_train = test_model(args, model, data_loader, dataset_type=dataset_type, return_preds=True)

        plot_tsne(args, all_embeddings_train, all_GTs_train, all_embeddings_val, all_GTs_val)
    elif choice == 5:
        cls_criterion = nn.BCEWithLogitsLoss(reduction='mean', weight=None)
        print("Evaluating on {} dataset.")
        args.batch_size = 32
        scaler = GradScaler()

        all_predictions_comb = None
        all_GTs_comb = None
        for fold_id in range(args.num_folds):
            print("*"*30)
            args.fold_id = fold_id
            args.model_name = args.model_name_orig
            print('args.model_name: ', args.model_name)
            args.model_name += str(args.fold_id)
            housekeeping(args)
            logger = Logger(args)

            args.load_model_from_checkpoint = True
            model = get_model_reference(args, load_from_checkpoint=True)
            model.eval()
            optimizer = get_optimizer(args, model)
            scheduler = get_scheduler(args, optimizer)

            dataset_type = 'validation'
            print("Prediction {} dataset.".format(dataset_type))
            data_loader = get_data_loader(args, dataset_type=dataset_type, training=False)
            all_predictions, all_GTs, all_embeddings = test_model(args, model, data_loader, dataset_type=dataset_type, return_preds=True)
            print("all_GTs: ", all_GTs)
            if all_predictions_comb is None:
                all_predictions_comb = all_predictions
                all_GTs_comb = all_GTs
            else:
                all_predictions_comb = np.concatenate((all_predictions_comb, all_predictions), axis=0)
                all_GTs_comb = np.concatenate((all_GTs_comb, all_GTs), axis=0)

            print("fold_id: {}    all_predictions.shape: {}".format(fold_id, all_predictions.shape))

        print("all_predictions_comb.shape: ", all_predictions_comb.shape)
        print("all_GTs_comb.shape: ", all_GTs_comb.shape)
        all_preds_comb = np.argmax(all_predictions_comb, axis=1)
        all_preds_comb = np.reshape(all_preds_comb, newshape=(all_preds_comb.shape[0], ))

        accuracy = np.sum(all_preds_comb == all_GTs_comb) / all_GTs_comb.shape[0]

        print("OOF accuracy: {:.6f}".format(accuracy))

        # print("all_predictions_comb: ", all_predictions_comb)
        # all_predictions_comb /= args.num_folds
        # print("avg. all_predictions_comb: ", all_predictions_comb)
        #
        # print("all_predictions_comb.shape: ", all_predictions_comb.shape)
        #
        # if args.execution_environment == 'kaggle':
        #     root_path = '../input/cassava-leaf-disease-classification'
        # elif args.execution_environment == 'local':
        #     root_path = './data'
        # test = pd.DataFrame()
        #
        # # test['image_id'] = list(os.listdir(root_path + '/test_images'))
        # test['image_id'] = np.load(args.model_store_dir + '/valid_images.npy')
        #
        # print("all_predictions_comb.shape: ", all_predictions_comb.shape)
        # print("test.shape: ", test.shape)
        # test['label'] = np.argmax(all_predictions_comb, axis=1)
        # test.to_csv('submission.csv', index=False)

    else:
        print("Enter valid choice...exiting...")
        # exit()
