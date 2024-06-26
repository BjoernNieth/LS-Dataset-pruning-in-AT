## Reused code from:
## Wang, Z., Pang, T., Du, C., Lin, M., Liu, W., and Yan, S.
## Better diffusion models further improve adversarial training, June 2023
## Code available at https://github.com/wzekai99/DM-Improves-AT

from contextlib import contextmanager


class ctx_noparamgrad(object):
    def __init__(self, module):
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        set_param_grad_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_param_grad_state(self.module, self.prev_grad_state)
        return False


class ctx_eval(object):
    def __init__(self, module):
        self.prev_training_state = get_module_training_state(module)
        self.module = module
        set_module_training_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_module_training_state(self.module, self.prev_training_state)
        return False


@contextmanager
def ctx_noparamgrad_and_eval(module):
    with ctx_noparamgrad(module) as a, ctx_eval(module) as b:
        yield (a, b)


def get_module_training_state(module):
    return {mod: mod.training for mod in module.modules()}


def set_module_training_state(module, training_state):
    for mod in module.modules():
        mod.training = training_state[mod]


def set_module_training_off(module):
    for mod in module.modules():
        mod.training = False


def get_param_grad_state(module):
    return {param: param.requires_grad for param in module.parameters()}


def set_param_grad_state(module, grad_state):
    for param in module.parameters():
        param.requires_grad = grad_state[param]


def set_param_grad_off(module):
    for param in module.parameters():
        param.requires_grad = False