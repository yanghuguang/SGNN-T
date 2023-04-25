#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""
import csv
import os
import sys
import time
from jarvis.core.atoms import Atoms
from alignn.data import get_train_val_loaders
from alignn.train import train_dgl

from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse

parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network"
)
parser.add_argument(
    "--root_dir",
    default="./",
    help="Folder with id_props.csv, structure files",
)
parser.add_argument(
    "--config_name",
    default="examples/sample_data/config_example.json",
    # default="data/config_example.json",
    help="Name of the config file",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--keep_data_order",
    default=False,
    help="Whether to randomly shuffle samples, True/False",
)

parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)

parser.add_argument(
    "--batch_size", default=None, help="Batch size, generally 64"
)

parser.add_argument(
    "--epochs", default=None, help="Number of epochs, generally 300"
)

parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)


def train_for_folder(
    root_dir="examples/sample_data",
    config_name="config.json",
    keep_data_order=False,
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    file_format="poscar",
    output_dir=None,
    label_test = "",
):
    """Train for a folder."""
    # config_dat=os.path.join(root_dir,config_name)
    # id_prop_dat = os.path.join(root_dir, "id_prop.csv")
    # id_prop_dat = "examples/sample_data/id_prop.csv"
    # id_prop_dat = "data/qm9_std_jctc.json"
    # id_prop_dat = "data/pdbbind_2015.json"
    id_prop_dat = "data/megnet.json"

    # id_prop_dat = "data/megnet.json"
    # print(config_name)
    config = loadjson(config_name)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    config.keep_data_order = keep_data_order
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    with open(id_prop_dat, "r") as f:
        import json
        reader = json.load(f)
        data = [row for row in reader]

    dataset = []
    n_outputs = []
    multioutput = False
    lists_length_equal = True


    # data = data[:30000]
    data = data[:50000]
    print(len(data))
    print(config.batch_size)
    for i in data:
        info = {}
        file_name = i["id"]
        info["atoms"] = i["atoms"]
        info["jid"] = file_name
        # tmp = i["U0"]     # 0.0077
        # tmp = i["R2"]     # 0.0081
        # tmp = i["H"]      # 0.0083
        # tmp = i["LUMO"]   # 0.0330
        # tmp = i["G"]      # 0.0086
        # tmp = i["Cv"]     # 0.0153
        # tmp = i["mu"]     # 0.0475
        # tmp = i["ZPVE"]   # 0.0070

        # tmp = i["e_hull"]
        # tmp = i["gap pbe"]
        # tmp = i["HOMO"]   # 0.0739
        # tmp = i["gap"]    # 0.0574
        # tmp = i["alpha"]  # 0.0184
        # tmp = i["U"]      # 0.0084
        # tmp = i["omega1"] # 0.0125
        tmp = i[label_test]
        info["target"] = tmp


        n_outputs.append(info["target"])
        dataset.append(info)




    # print ('n_outputs',n_outputs[0])
    if multioutput and classification_threshold is not None:
        raise ValueError("Classification for multi-output not implemented.")
    if multioutput and lists_length_equal:
        config.model.output_features = len(n_outputs[0])
    else:
        # TODO: Pad with NaN
        if not lists_length_equal:
            raise ValueError("Make sure the outputs are of same size.")
        else:
            config.model.output_features = 1
    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        target=config.target,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
    )
    t1 = time.time()
    train_dgl(
        config,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
    )
    t2 = time.time()
    print("Time taken (s):", t2 - t1)

    # train_data = get_torch_dataset(


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    # label_set = ["ZPVE", "HOMO", "gap", "alpha", "U", "omega1"]
    label_set = ["e_form"]  #0.0383    0.0341  0.0294
    # label_set = ["gap pbe"] #0.2716
    for curLabel in label_set:
        print(curLabel)
        print("aaaa      ")
        train_for_folder(
            root_dir=args.root_dir,
            config_name=args.config_name,
            keep_data_order=args.keep_data_order,
            classification_threshold=args.classification_threshold,
            output_dir=args.output_dir,
            batch_size=(args.batch_size),
            epochs=(args.epochs),
            file_format=(args.file_format),
            label_test = curLabel,
        )


