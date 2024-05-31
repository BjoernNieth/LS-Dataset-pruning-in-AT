import mlflow

def log_args(args):
    for name in iter(args.__dict__.keys()):
        mlflow.log_param(name, args.__dict__[name])
