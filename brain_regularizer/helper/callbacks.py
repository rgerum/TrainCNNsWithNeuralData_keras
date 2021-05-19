import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import pandas as pd


def getCallbacks(args):
    # generate the output path
    output_path = getOutputPath(args)

    # callbacks
    if args.no_log is False:
        callbacks = [
            keras.callbacks.ModelCheckpoint(str(output_path / "weights.hdf5"), save_best_only=True,
                                            save_weights_only=True),
            TrainingHistory(output_path),
        ]
    else:
        callbacks = []

    return callbacks

class TrainingHistory(keras.callbacks.Callback):
    def __init__(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.output = output / "data.csv"
        self.data = []

    #def on_train_begin(self, logs={}):
    #    self.data = []

    def on_epoch_end(self, epoch, logs={}):
        logs["epoch"] = epoch
        self.data.append(logs)
        pd.DataFrame(self.data).to_csv(self.output, index=False)


def getGitHash():
    import subprocess
    try:
        short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        short_hash = str(short_hash, "utf-8").strip()
        return short_hash
    except subprocess.CalledProcessError:
        return ""

def getGitLongHash():
    import subprocess
    try:
        short_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        short_hash = str(short_hash, "utf-8").strip()
        return short_hash
    except subprocess.CalledProcessError:
        return ""


def getOutputPath(args):
    from datetime import datetime
    parts = [
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        getGitHash(),
    ]
    parts.extend([str(k) + "=" + str(v) for k, v in args._get_kwargs() if k != "output"])

    output = Path(args.output) / (" ".join(parts))
    import yaml
    output.mkdir(parents=True, exist_ok=True)
    arguments = dict(datetime=parts[0], commit=parts[1], commitLong=getGitLongHash(), run_dir=os.getcwd())
    arguments.update(args._get_kwargs())
    with open(output / "arguments.yaml", "w") as fp:
        yaml.dump(arguments, fp)
    print("OUTPUT_PATH=\""+str(output)+"\"")
    return output