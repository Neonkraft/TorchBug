from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

from rich import print
from rich.console import Console

from ..tables.table_view import TableView
from ..statistics.model_statistics import ModelStatistics
from ..utils import (
    find_mismatches,
    show_comparison,
    init_weights,
    get_leaf_modules,
    get_module_attrs_string,
    get_module_name
)


def mark_module_for_comparison(module, name):
    assert not list(module.children()), "The given module is not a leaf module"
    module.__pycaliper_attributes = {}
    module.__pycaliper_attributes["name"] = name


def mark_all_modules_for_comparison(model):
    for module in get_leaf_modules(model):
        mark_module_for_comparison(module, "")


def unmark_module_for_comparison(module):
    if hasattr(module, "__pycaliper_attributes"):
        del(module.__pycaliper_attributes)


def unmark_all_modules_for_comparison(model):
    for module in get_leaf_modules(model):
        unmark_module_for_comparison(module)


def compare_modules_in_forward_pass(target_model_stats, model_stats, input_shape, as_table=True):
    x = torch.randn(*input_shape)

    console = Console()
    with console.status("[bold green]Passing input data through models...") as status:
        target_stats, target_inputs = forward_with_hooks(target_model_stats.model, x)
        stats, model_inputs = forward_with_hooks(model_stats.model, x)

    target_rows = [row for row in target_stats.get_db().all()]
    model_rows = [row for row in stats.get_db().all()]

    mismatches_target, _ = find_mismatches(target_rows, model_rows)
    mismatches_model, _ = find_mismatches(model_rows, target_rows)

    if mismatches_model or mismatches_target:
        print("\n[red][bold]Number of leaf modules in forward pass do not match! See below:[/bold][/red]")
    else:
        print("\n[green][bold]Number of leaf modules in forward pass match![/bold][/green]")

    show_comparison({model_stats.name: mismatches_target, target_model_stats.name: mismatches_model}, as_table=as_table)


def count_matches(target_tensors, model_tensors, rtol=10e-5, atol=10e-8):
    module_matches = {}

    for module in target_tensors.keys():
        if module in model_tensors.keys():
            mismatches, matches = find_mismatches(target_tensors[module], model_tensors[module], rtol, atol)
            n_inputs = len(target_tensors[module])
            n_matches = len(matches)
            module_matches[module] = (n_inputs, n_matches)

    return module_matches


def find_matches(target_tensors, model_tensors, rtol=10e-5, atol=10e-8):
    module_matches = []
    module_no_matches = []

    for target_module in target_tensors.keys():
        for module in model_tensors.keys():
            _, matches = find_mismatches(target_tensors[target_module], model_tensors[module], rtol, atol)
            if len(matches) > 0:
                module_matches.append((target_module, module, len(matches)))

    matched_modules = {m[1] for m in module_matches}
    modules_without_match = sorted(list(set(model_tensors.keys()).difference(matched_modules)))

    return module_matches, modules_without_match


def compare_module_outputs_in_forward_pass(target_model_stats, model_stats, input_shape, as_table=True,
                                           show_matches=True, modules=None, marked_modules_only=False,
                                           rtol=10e-5, atol=10e-8):
    x = torch.randn(*input_shape)

    init_weights(target_model_stats.model)
    init_weights(model_stats.model)

    console = Console()

    with console.status("[bold green]Passing input data through models...") as status:
        _, target_outputs = forward_with_hooks(target_model_stats.model, x, modules, marked_modules_only)
        _, model_outputs = forward_with_hooks(model_stats.model, x, modules, marked_modules_only)

    if marked_modules_only:
        with console.status("[bold green]Matching marked module inputs... This might take some time...") as status:
            matches, no_matches_modules = find_matches(target_outputs, model_outputs, rtol, atol)

        for target_module, module, n_matches in matches:
            print(f"Output of [magenta][italic]{module}[/italic] in {model_stats.name}[/magenta] " +
                  f"[green]matches with[/green] output of [magenta][italic]{target_module}[/italic] in {target_model_stats.name} [/magenta]")

        if no_matches_modules:
            print(f"[red][bold]No matches found for following modules:")
            for module in no_matches_modules:
                print(f"[magenta]{module} in {model_stats.name}")

        return

    with console.status("[bold green]Matching module inputs... This might take some time...") as status:
        matches = count_matches(target_outputs, model_outputs, rtol, atol)

    rows = []
    for key in matches.keys():
        attributes = key.split("--")
        row = {}

        for attr in attributes:
            k, v = attr.split("=")
            row[k] = v

        row["matches"] = matches[key][1]
        row["mismatches"] = matches[key][0] - matches[key][1]

        rows.append(row)

    print(f"\n[bold][magenta]Module-wise output comparison [/magenta][/bold]")

    if not show_matches:
        rows = [row for row in rows if row["mismatches"] > 0]

    rows = sorted(rows, key=lambda x: str(x))

    row_types = defaultdict(lambda: [])
    for row in rows:
        row_types[row["type"]].append(row)

    for rows in row_types.values():
        table = TableView(rows, "", last_columns=["mismatches", "matches"])

        if as_table:
            table.print()
        else:
            print(table.data)

    if not [row for row in rows if row["mismatches"] > 0]:
        print(
            f"[green]Outputs of all modules present in {model_stats.name} match with the corresponding {target_model_stats.name} module outputs!\n")


def compare_final_outputs_in_forward_pass(target_model, model, input_shape, rtol=10e-5, atol=10e-8):
    init_weights(target_model)
    init_weights(model)

    x = torch.randn(*input_shape)

    out_target = target_model(x).detach().numpy()
    out_model = model(x).detach().numpy()

    if out_target.shape != out_model.shape:
        return False

    is_equal = np.allclose(out_target, out_model, rtol=rtol, atol=atol)

    if is_equal:
        print("\n[green][bold]Model outputs match![/bold][/green]")
    else:
        print("\n[red][bold]Model outputs do not match![/bold][/red]")

    return is_equal


def forward_with_hooks(model, x, modules=None, marked_modules_only=False):
    """Registers forward hook in the leaf modules of the model, based on the other arguments.

    Args:
        model               : Pytorch model
        x                   : Data to forward pass through the model
        modules             : List of names of modules. If specified, the forward hooks are only added to modules
                              of the kind specified in the list. Otherwise, the hooks are added to all the models
        marked_modules_only : If true, only leaf modules which have been marked will have the forward hooks registered
                              to them. Overrides the modules argument.
    """
    leaf_modules = get_leaf_modules(model)

    module_outputs = defaultdict(lambda: [])
    stats = ModelStatistics(nn.Module(), "model")

    def hook_fn(m, i, o):
        if hasattr(m, "__pycaliper_attributes") and m.__pycaliper_attributes["name"] != "":
            key = m.__pycaliper_attributes["name"]
        else:
            key = get_module_attrs_string(m)
        stats._add_module_to_db(m)
        module_outputs[key].append(o[0].detach().numpy())

    for module in leaf_modules:
        if marked_modules_only:
            if hasattr(module, "__pycaliper_attributes"):
                module.register_forward_hook(hook_fn)
        elif modules is None or get_module_name(module) in modules:
            module.register_forward_hook(hook_fn)

    model(x)

    return stats, module_outputs
