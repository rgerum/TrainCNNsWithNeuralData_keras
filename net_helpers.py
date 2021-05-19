import glob
import sys

import numpy as np
import pandas as pd
from pathlib import Path

def processPaths(name, filter=None, file_name=None):
    results = []
    # if the function is called with a list or tuple, iterate over those and join the results
    if isinstance(name, (tuple, list)):
        for n in name:
            results.extend(processPaths(n, filter=filter, file_name=file_name))
    else:
        # add the file_name pattern if there is one and if it is not already there
        if file_name is not None and Path(name).suffix != Path(file_name).suffix:
            name = Path(name) / "**" / file_name
        # if it is a glob pattern, add all matching elements
        if "*" in str(name):
            results = glob.glob(str(name), recursive=True)
        # or add the name directly
        elif Path(name).exists():
            results = [name]
        # filter results if a filter is provided
        if filter is not None:
            results = [n for n in results if filter(n)]

        # if nothing was found, try to give a meaningful error message
        if len(results) == 0:
            # get a list of all parent folders
            name = Path(name).absolute()
            hierarchy = []
            while name.parent != name:
                hierarchy.append(name)
                name = name.parent
            # iterate over the parent folders, starting from the lowest
            for path in hierarchy[::-1]:
                # check if the path exists (or is a glob pattern with matches)
                if "*" in str(path):
                    exists = len(glob.glob(str(path)))
                else:
                    exists = path.exists()
                # if it does not exist, we have found our problem
                if not exists:
                    target = f"No file/folder \"{path.name}\""
                    if "*" in str(path.name):
                        target = f"Pattern \"{path.name}\" not found"
                    source = f"in folder \"{path.parent}\""
                    if "*" in str(path.parent):
                        source = f"in any folder matching the pattern \"{path.parent}\""
                    print(f"WARNING: {target} {source}", file=sys.stderr)
                    break
    return results


def read_data(name, load=pd.read_csv, filter=None, file_name=None, do_exclude=True):
    datas = []
    # iterate over all paths
    for name in processPaths(name, filter=filter, file_name=file_name):
        # load data as pandas dataframe
        data = load(name)
        # add filename
        data["filename"] = name
        # add metadata
        meta = getMeta(name)
        # add the metadata to the data frame
        for key, value in meta.items():
            data[key] = value
        # if a flag exclude is present in the data, do exclude it
        if "exclude" in meta and do_exclude is True:
            if meta["exclude"] is True:
                print("excluding", name)
                continue
        # add the data to a list
        datas.append(data)
    # concat the dataframes
    return pd.concat(datas)


def getMeta(filename, cached_meta_files={}):
    import yaml
    filename = Path(filename)
    # if the data is not cached yet
    if filename not in cached_meta_files:
        # get meta data from parent folder
        cached_meta_files[filename] = {}
        if filename.parent != filename:
            cached_meta_files[filename] = getMeta(filename.parent).copy()

        # find meta data filename
        if str(filename).endswith(".tif"):
            yaml_file = Path(str(filename).replace(".tif", "_meta.yaml"))
        elif str(filename).endswith("_evaluated_new.csv"):
            yaml_file = Path(str(filename).replace("_evaluated_new.csv", "_meta.yaml"))
        else:
            yaml_file = filename / "meta.yaml"

        # load data from file and join with parent meta data
        if yaml_file.exists():
            with yaml_file.open() as fp:
                data = yaml.load(fp, Loader=yaml.SafeLoader)
            if data is not None:
                cached_meta_files[filename].update(data)

    # return the metadata
    return cached_meta_files[filename]


def load_data(names, do_exclude=True):
    def load(name):
        data = pd.read_csv(name)
        if "accuracy" not in data:
            data["accuracy"] = np.nan
            data["val_accuracy"] = np.nan

        if "class_accuracy" in data:
            accuracy = np.array(data.accuracy)
            val_accuracy = np.array(data.val_accuracy)
            accuracy[np.isnan(data.accuracy)] = data.class_accuracy[np.isnan(data.accuracy)]
            val_accuracy[np.isnan(data.val_accuracy)] = data.val_class_accuracy[np.isnan(data.val_accuracy)]
            data["accuracy"] = accuracy
            data["val_accuracy"] = val_accuracy
        if "epoch" not in data:
            data["epoch"] = np.array(data.index)
        return data

    return read_data(names, load, filter=None, file_name="*.csv", do_exclude=do_exclude)
