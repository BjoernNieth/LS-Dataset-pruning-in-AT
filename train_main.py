import os.path
import pandas as pd
import mlflow
from core.utils.utils import seed
from core.utils.logging import log_args, save_args_json
import numpy as np
from torch.utils.data import DataLoader
from core.gowal21uncovering.utils import WATrainer
from core.data import DATASETS
from core.data import CustomDataset
from core.utils.parser import train_parser
from core.utils.logger import Logger
import time
from core.data import load_data, get_data_info
import pickle

print("Start")

parse = train_parser()
args = parse.parse_args()
seed(args.seed)
log_directory = os.path.join("./results", args.run_name)
os.makedirs(log_directory, exist_ok=True)
logger = Logger(os.path.join(log_directory, 'log-train.log'))

# Store meta information about training
meta_dict = {}
model_string = args.dataset + "_" + args.model + "_" + args.scheduler + "_" + str(args.lr) + "_adversarial_" + str(args.adversarial)
if args.adversarial:
    model_string = model_string + "_" + str(args.attack)
if args.use_normal_predictions:
    normal_model_string = args.dataset + "_" + args.model + "_" + args.scheduler + "_" + str(args.lr) + "_adversarial_False"
    predictions_path = os.path.join("./data_files", "dynamic_uncertainty", normal_model_string)
else:
    predictions_path = os.path.join("./data_files", "dynamic_uncertainty", model_string)

train_dataset, test_dataset, _, train_loader, test_loader, val_loader = load_data(args.dataset, predictions_path,
              args.deduplication_threshold, args.subsampling, args.prune_percentage, args.batch_size,
              args.batch_size_validation, args.generated_fraction, args.prune_base, args.keep_class_distribution,
              args.use_normal_predictions, args.use_adversarial_predictions, args.only_generated, args.adversarial,
              args.no_overfitted_epochs, args.fundamental_frequency_pruning)

info = get_data_info(args.dataset)
meta_dict["batches_per_epoch"] = len(train_loader)
meta_dict["model_string"] = model_string
meta_dict["predictions_path"] = predictions_path

isAugmented, _ = DATASETS[args.dataset]
predictions_path = os.path.join("./data_files", "dynamic_uncertainty", model_string)

# Check if predictions for dynamic uncertainty are already available for this model + dataset + isadverserial
record_predictions = not os.path.isfile(os.path.join(predictions_path, "predictions.npy")) and \
                     args.deduplication_threshold == None and args.prune_percentage == None and \
                     (not isAugmented or args.only_generated)



record_predictions_adversarial = not os.path.isfile(os.path.join(predictions_path, "predictions.npy")) and\
                                 args.deduplication_threshold == None and args.prune_percentage == None and args.adversarial and \
                                 (not isAugmented or args.only_generated)

trainer = WATrainer(info, args, len(train_dataset))
if args.resume_training:
    start_epoch, run_id, best_acc, record_predictions, record_predictions_adversarial = trainer.load_model_resume(os.path.join(log_directory, 'state-last.pt'))
    start_epoch = start_epoch + 1
    logger.log(f'Resuming at epoch {start_epoch}')
    if record_predictions:
        predictions = np.load(os.path.join(predictions_path, "predictions.npy"))
        predictions = predictions[0:start_epoch]
        if args.adversarial:
            predictions_adversarial = np.load(os.path.join(predictions_path, "predictions_adversarial.npy"))
            predictions_adversarial = predictions_adversarial[0:start_epoch]

        if isAugmented:

            predictions_dataset = CustomDataset(train_dataset.data,
                                                train_dataset.targets,
                                                transform=test_dataset.transform)
            train_predictions_loader = DataLoader(predictions_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            predictions_dataset = CustomDataset(train_dataset.data,
                                                train_dataset.targets,
                                                transform=test_dataset.transform)
            train_predictions_loader = DataLoader(predictions_dataset, batch_size=args.batch_size, shuffle=False)

    metrics_df = pd.read_csv(os.path.join(log_directory, 'metrics.csv'))
    metrics_df = metrics_df.iloc[:start_epoch]
    with open(os.path.join(log_directory, "meta_dict.pkl"), 'rb') as f:
        meta_dict = pickle.load(f)
else:
    start_epoch = 1
    run_id = None
    metrics_df = None
    best_acc = np.zeros(1)
    save_args_json(args, os.path.join(log_directory, "args.json"))

    # Only record the predictions, if dataset hasn't been pruned
    if record_predictions or record_predictions_adversarial :
        # For augmented data just use the augmentation dataset
        if isAugmented:
            assert args.only_generated is True
            if args.adversarial:
                predictions_adversarial = np.array([], dtype=np.float).reshape(0, len(train_dataset))
            predictions = np.array([], dtype=np.float).reshape(0,len(train_dataset))
            predictions_dataset = CustomDataset(train_dataset.data,
                                                train_dataset.targets,
                                                transform=test_dataset.transform)
            train_predictions_loader = DataLoader(predictions_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            if args.adversarial:
                predictions_adversarial = np.array([], dtype=np.float).reshape(0, train_dataset.data.shape[0])
            predictions = np.array([], dtype=np.float).reshape(0, train_dataset.data.shape[0])
            # Train dataset without data augmentations
            predictions_dataset = CustomDataset(train_dataset.data,
                                                train_dataset.targets,
                                                transform=test_dataset.transform)
            train_predictions_loader = DataLoader(predictions_dataset, batch_size=args.batch_size, shuffle=False)

if args.adversarial:
    log_dataframe = pd.DataFrame(columns=["Epoch", "Learning Rate", "" ])

mlflow.end_run()
with open(os.path.join(log_directory, "meta_dict.pkl"), 'wb') as f:
    pickle.dump(meta_dict, f)

with mlflow.start_run(run_name=args.run_name, run_id=run_id) as run:
    log_args(args)
    for epoch in range(start_epoch, args.epochs + 1):
        metrics = {"Epoch" : epoch}
        logger.log("--------- Epoch {} ---------".format(epoch))

        if args.scheduler:
            mlflow.log_metric("Learning Rate", trainer.scheduler.get_last_lr()[0], step=epoch)
            metrics["Learning Rate"] = trainer.scheduler.get_last_lr()[0]

        start = time.time()
        res = trainer.train(train_loader, epoch=epoch, adversarial=args.adversarial)
        end = time.time()
        logger.log("Time taken training {}".format(end - start))
        for metric in res.keys():
            mlflow.log_metric(metric, res[metric], step=epoch)
            metrics[metric] = res[metric]


        start = time.time()
        if args.adversarial:
            val_acc_adv = trainer.eval(val_loader, adversarial=args.adversarial)
            mlflow.log_metric("Validation Accuracy Adversarial", val_acc_adv, step=epoch)
            metrics["Validation Accuracy Adversarial"] = val_acc_adv

        val_acc = trainer.eval(val_loader, adversarial=False)
        mlflow.log_metric("Validation Accuracy Clean", val_acc, step=epoch)
        metrics["Validation Accuracy Clean"] = val_acc
        end = time.time()
        logger.log("Time taken validation {}".format(end - start))

        if args.adversarial:
            if val_acc_adv > best_acc[0]:
                best_acc[0] = val_acc_adv
                trainer.save_model(os.path.join(log_directory, 'best_weights.pt'))
                logger.log("New best epoch with {}".format(val_acc_adv))
                meta_dict["best epoch"] = epoch
                with open(os.path.join(log_directory, "meta_dict.pkl"), 'wb') as f:
                    pickle.dump(meta_dict, f)
        else:
            if val_acc > best_acc[0]:
                best_acc[0] = val_acc
                trainer.save_model(os.path.join(log_directory, 'best_weights.pt'))
                logger.log("New best epoch with {}".format(val_acc))
                meta_dict["best epoch"] = epoch
                with open(os.path.join(log_directory, "meta_dict.pkl"), 'wb') as f:
                    pickle.dump(meta_dict, f)

        if record_predictions:
            start = time.time()
            predictions = np.vstack([predictions, trainer.predictions(train_predictions_loader,
                                                                      adversarial=False)])
            if not os.path.isdir(predictions_path):
                os.mkdir(predictions_path)
            np.save(os.path.join(predictions_path, "predictions.npy"), predictions)
            end = time.time()
            logger.log("Time taken record predictions {}".format(end - start))

        if record_predictions_adversarial:
            start = time.time()
            predictions_adversarial = np.vstack([predictions_adversarial, trainer.predictions(train_predictions_loader,
                                                                      adversarial=True)])
            if not os.path.isdir(predictions_path):
                os.mkdir(predictions_path)
            np.save(os.path.join(predictions_path, "predictions_adversarial.npy"), predictions_adversarial)
            end = time.time()
            logger.log("Time taken record predictions {}".format(end - start))

        if epoch % 25 == 0:
            # Every 25 epochs make a save of the predictions so far
            if record_predictions:
                if not os.path.isdir(predictions_path):
                    os.mkdir(predictions_path)
                np.save(os.path.join(predictions_path, "predictions.npy"), predictions)

        df = pd.DataFrame([metrics]).set_index("Epoch")
        if metrics_df is not None:
            metrics_df = pd.concat([metrics_df, df])
        else:
            metrics_df = df
        if epoch % 10 == 0:
            trainer.save_model_resume(os.path.join(log_directory, 'state-last.pt'), epoch, run.info.run_id, best_acc, record_predictions, record_predictions_adversarial)
            metrics_df.to_csv(os.path.join(log_directory, 'metrics.csv'))


        if args.early_stop_diff > 0:
            if epoch - meta_dict["best epoch"] > args.early_stop_diff:
                break

    trainer.save_model(os.path.join(log_directory, 'last_weights.pt'))
    if record_predictions:
        if not os.path.isdir(predictions_path):
            os.mkdir(predictions_path)
        np.save(os.path.join(predictions_path, "predictions.npy"), predictions)

    if record_predictions_adversarial:
        if not os.path.isdir(predictions_path):
            os.mkdir(predictions_path)
        np.save(os.path.join(predictions_path, "predictions_adversarial.npy"), predictions_adversarial)
    metrics_df.to_csv(os.path.join(log_directory, 'metrics'))


    trainer.load_model(os.path.join(log_directory, 'best_weights.pt'))
    trainer.model.eval()

    start = time.time()
    trainer.autoattack_loss(test_loader, os.path.join(log_directory, 'log-autoattack.log'))

    end = time.time()
    logger.log("Time taken autoattack {}".format(end - start))

    start = time.time()
    if args.adversarial:
        test_acc_adv = trainer.eval(test_loader, adversarial=args.adversarial)
        mlflow.log_metric("Test Accuracy Adversarial", test_acc_adv)
        logger.log("Test Accuracy Robust: " + str(test_acc_adv))
    test_acc = trainer.eval(test_loader, adversarial=False)
    end = time.time()
    logger.log("Test Accuracy Clean: " + str(test_acc))
    logger.log("Time taken testing {}".format(end - start))

    if record_predictions_adversarial or record_predictions:
        with open(os.path.join(predictions_path, "meta_dict.pkl"), 'wb') as f:
            pickle.dump(meta_dict, f)

    mlflow.log_metric("Test Accuracy Clean", test_acc)
