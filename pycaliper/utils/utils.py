import hashlib
import numpy as np
import torch

from rich import print
from ..tables.table_view import TableView


def get_leaf_modules(model):
    children = list(model.children())
    leaf_modules = []

    if not children:
        return [model]
    else:
        for child in children:
            leaf_modules.extend(get_leaf_modules(child))

    return leaf_modules


def get_module_name(module):
    klass = module.__class__
    module_name = klass.__module__

    if module_name == 'builtins':
        return klass.__qualname__
    return module_name + '.' + klass.__qualname__


def get_module_attrs(module):
    return sorted([attr for attr in vars(module) if attr[0] != "_" and attr != "training"])


def get_module_attrs_string(module):
    attrs = get_module_attrs(module)
    attr_string = "type=" + get_module_name(module) + "--"

    for attr in attrs:
        attr_string += attr + "=" + str(getattr(module, attr)) + "--"

    return attr_string[:-2]


def get_int_hash(module):
    attr_string = get_module_attrs_string(module)
    return int(hashlib.sha1(attr_string.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def init_weights(model):
    modules = get_leaf_modules(model)

    for module in modules:
        seed = get_int_hash(module)
        torch.random.manual_seed(seed)

        for p in module.parameters():
            torch.nn.init.normal_(p)


def _is_equal(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape == b.shape:
            return np.isclose(a, b).all()
        else:
            return False
    else:
        return a == b


def find_mismatches(a, b):
    already_matched = [False] * len(b)

    for doc_a in a:
        for idx, doc_b in enumerate(b):
            if already_matched[idx]:
                continue

            if _is_equal(doc_a, doc_b):
                already_matched[idx] = True

    mismatches = []
    for idx, match in enumerate(already_matched):
        if not match:
            mismatches.append(b[idx])

    return mismatches


def print_table(table, as_table=True):
    if as_table:
        table.print()
    else:
        print(table.data)


def show_comparison(missing_modules, as_table=True):
    models = []

    for model_name, missing_modules_list in missing_modules.items():
        models.append((model_name, missing_modules_list))

    if models[0][1] != []:
        model_name = models[0][0]
        missing_modules = models[0][1]
        missing_modules = sorted(missing_modules, key=lambda x: x["type"])
        table = TableView(missing_modules, model_name)
        print(f"\n[bold]Leaf modules present in [green]{model_name}[/green] but missing in [red]{models[1][0]}[/red][/bold]")
        print_table(table, as_table)

    if models[1][1] != []:
        model_name = models[1][0]
        missing_modules = models[1][1]
        missing_modules = sorted(missing_modules, key=lambda x: x["type"])
        table = TableView(missing_modules, model_name)
        print(f"\n[bold]Leaf modules present in [green]{model_name}[/green] but missing in [red]{models[0][0]}[/red][/bold]")
        print_table(table, as_table)
