from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

from rich import print

from ..tables.table_view import TableView
from ..statistics.model_statistics import ModelStatistics
from ..utils import (
    find_mismatches,
    show_comparison,
    init_weights,
    get_leaf_modules,
    get_module_attrs_string
)


def compare_modules_in_forward_pass(target_model_stats, model_stats, input_shape, as_table=True):
    x = torch.randn(*input_shape)
    target_stats, target_inputs = forward_with_hooks(target_model_stats.model, x)
    stats, model_inputs = forward_with_hooks(model_stats.model, x)

    target_rows = [row for row in target_stats.get_db().all()]
    model_rows = [row for row in stats.get_db().all()]

    mismatches_target = find_mismatches(target_rows, model_rows)
    mismatches_model = find_mismatches(model_rows, target_rows)

    if mismatches_model or mismatches_target:
        print("\n[red][bold]Number of leaf modules in forward pass do not match! See below:[/bold][/red]")
    else:
        print("\n[green][bold]Number of leaf modules in forward pass match![/bold][/green]")

    show_comparison({target_model_stats.name: mismatches_model, model_stats.name: mismatches_target}, as_table=as_table)


def count_matches(target_inputs, model_inputs):
    module_matches = {}

    for module in target_inputs.keys():
        if module in model_inputs.keys():
            mismatches = find_mismatches(target_inputs[module], model_inputs[module])
            n_inputs = len(target_inputs[module])
            n_matches = len(target_inputs[module]) - len(mismatches)
            n_matches = 0 if n_matches < 0 else n_matches
            module_matches[module] = (n_inputs, n_matches)

    return module_matches


def compare_module_inputs_in_forward_pass(target_model_stats, model_stats, input_shape, as_table=True, show_matches=True):
    x = torch.randn(*input_shape)

    init_weights(target_model_stats.model)
    init_weights(model_stats.model)

    _, target_inputs = forward_with_hooks(target_model_stats.model, x)
    _, model_inputs = forward_with_hooks(model_stats.model, x)

    matches = count_matches(target_inputs, model_inputs)

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

    if not show_matches:
        rows = [row for row in rows if row["mismatches"] > 0]

    print(f"\n[bold][magenta]Module-wise input comparison [/magenta][/bold]")

    rows = sorted(rows, key=lambda x: str(x))

    table = TableView(rows, "", last_columns=["mismatches", "matches"])

    if as_table:
        table.print()
    else:
        print(table.data)


def compare_outputs_forward_pass(target_model, model, input_shape):
    init_weights(target_model)
    init_weights(model)

    x = torch.randn(*input_shape)

    out_target = target_model(x).detach().numpy()
    out_model = model(x).detach().numpy()

    if out_target.shape != out_model.shape:
        return False

    is_equal = np.allclose(out_target, out_model)

    if is_equal:
        print("\n[green][bold]Model outputs match![/bold][/green]")
    else:
        print("\n[red][bold]Model outputs do not match![/bold][/red]")

    return is_equal


def forward_with_hooks(model, x):
    modules = get_leaf_modules(model)

    module_inputs = defaultdict(lambda: [])
    stats = ModelStatistics(nn.Module(), "model")

    def hook_fn(m, i, o):
        key = get_module_attrs_string(m)
        stats._add_module_to_db(m)
        module_inputs[key].append(i[0].detach().numpy())

    for module in modules:
        module.register_forward_hook(hook_fn)

    model(x)

    return stats, module_inputs
