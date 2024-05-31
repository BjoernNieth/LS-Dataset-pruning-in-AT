import mlflow
import json

def log_args(args):
    for name in iter(args.__dict__.keys()):
        if name == "resume_training":
            continue
        elif name == "early_stop_diff":
            continue

        mlflow.log_param(name, args.__dict__[name])

def save_args_json(args, path):
    args_dict = vars(args)
    with open(path, "w") as f:
        json.dump(args_dict, f, indent=4)