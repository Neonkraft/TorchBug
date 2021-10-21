import hashlib
import numpy as np
import torch

from rich import print
from ..tables.table_view import TableView


def get_leaf_modules(model):
    """Gets all the leaf modules present inside a module as a flat list.
       'Leaf module' here refers to all the torch modules which have no children, such as
       torch.nn.modules.linear.Linear or torch.nn.modules.conv.Conv2d, but not torch.nn.Sequential
       or other custom modules with other modules inside them.

    Args:
        model       : Instance of torch.nn.Module

    Returns:
        List of all leaf modules in
    """
    children = list(model.children())
    leaf_modules = []

    if not children:
        return [model]
    else:
        for child in children:
            leaf_modules.extend(get_leaf_modules(child))

    return leaf_modules


def get_module_name(module):
    """Gets the fully qualified name of a given module."""
    klass = module.__class__
    module_name = klass.__module__

    if module_name == 'builtins':
        return klass.__qualname__
    return module_name + '.' + klass.__qualname__


def get_module_attrs(module):
    """Gets a list of all the non-private module attributes."""
    return sorted([attr for attr in vars(module) if attr[0] != "_" and attr != "training"])


def get_module_attrs_string(module):
    """Gets the string representation of a module based on its non-private attributes."""
    attrs = get_module_attrs(module)
    attr_string = "type=" + get_module_name(module) + "--"

    for attr in attrs:
        attr_string += attr + "=" + str(getattr(module, attr)) + "--"

    return attr_string[:-2]


def get_int_hash(module):
    """Get an integer hash of a module based on its non-private attrbutes."""
    attr_string = get_module_attrs_string(module)
    return int(hashlib.sha1(attr_string.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def init_weights(model):
    """Initializes the weights of the modules of the given model using a seed derived from the non-private module attributes."""
    modules = get_leaf_modules(model)

    for module in modules:
        seed = get_int_hash(module)
        torch.random.manual_seed(seed)

        for p in module.parameters():
            torch.nn.init.normal_(p)


def _is_equal(a, b, rtol=10e-5, atol=10e-8):
    """Compares two objects and returns the result."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape == b.shape:
            return np.isclose(a, b, rtol=rtol, atol=atol).all()
        else:
            return False
    else:
        return a == b


def find_mismatches(a, b, rtol=10e-5, atol=10e-8):
    already_matched = [False] * len(b)

    for doc_a in a:
        for idx, doc_b in enumerate(b):
            if already_matched[idx]:
                continue

            if _is_equal(doc_a, doc_b, rtol, atol):
                already_matched[idx] = True

    mismatches = []
    matches = []
    for idx, match in enumerate(already_matched):
        if not match:
            mismatches.append(b[idx])
        else:
            matches.append(b[idx])

    return mismatches, matches


def print_table(table, as_table=True):
    """Prints the given table as a table or as json, depending on the as_table argument."""
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
