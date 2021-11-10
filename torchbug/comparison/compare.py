from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

from rich import print
from rich.console import Console

from ..tables.table_view import TableView
from ..summary.model_summary import ModelSummary
from ..utils import (
    find_mismatches,
    show_comparison,
    init_weights,
    get_leaf_modules,
    get_module_attrs_string,
    get_module_name
)


def mark_module_for_comparison(module, name):
    """Marks a given leaf module for comparison.

    Args:
        module      : Instance of torch.nn.Module (such and Conv2d or BatchNorm2d)
        name        : Name to given the output tensor from the given module, for comparison later

    Returns:
        None.
    """
    assert not list(module.children()), "The given module is not a leaf module"
    module.__torchbug_attributes = {}
    module.__torchbug_attributes["name"] = name


def mark_all_modules_for_comparison(model):
    """Marks all the leaf modules of a module for comparison. The modules will be marked with an empty string for its name.

    Args:
        model      : Any instance of torch.nn.Module, which can have other sub modules.

    Returns:
        None.
    """
    for module in get_leaf_modules(model):
        mark_module_for_comparison(module, "")


def unmark_module_for_comparison(module):
    """Unmarks a given leaf module for comparison.

    Args:
        module      : Instance of torch.nn.Module (such and Conv2d or BatchNorm2d).

    Returns:
        None.
    """
    if hasattr(module, "__torchbug_attributes"):
        del(module.__torchbug_attributes)


def unmark_all_modules_for_comparison(model):
    """Unmarks all the leaf modules of a module for comparison.

    Args:
        model      : Any instance of torch.nn.Module, which can have other sub modules.

    Returns:
        None.
    """
    for module in get_leaf_modules(model):
        unmark_module_for_comparison(module)


def compare_modules_in_forward_pass(target_model_stats, model_stats, input_shape, as_table=True):
    """Compares the modules called during the forward pass in both models.

       This is done to ensure that the registered sub-modules are actually consumed in the forwad pass.
       For example, comparing the registered modules of the following two models will indicate they are
       equivalent, but their outputs during forward pass will not match:

        class ModelA(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(10, 10, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                return x

        class ModelB(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(10, 10, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                return x

    Args:
        target_model_stats      : Target model wrapped in a ModelSummary object.
        model_stats             : New model wrapped in a ModelSummary object.
        input_shape             : Shape of the input tensor to both the models.
        as_table                : If True, shows the results in tabular form.
                                  Else, prints it as json.

    Returns:
        None.
    """
    x = torch.randn(*input_shape)

    console = Console()
    with console.status("[bold green]Passing input data through models...") as status:
        target_stats, _ = _forward_with_hooks(target_model_stats.model, x)
        stats, _ = _forward_with_hooks(model_stats.model, x)

    target_rows = [row for row in target_stats.get_db().all()]
    model_rows = [row for row in stats.get_db().all()]

    mismatches_target, _ = find_mismatches(target_rows, model_rows)
    mismatches_model, _ = find_mismatches(model_rows, target_rows)

    if mismatches_model or mismatches_target:
        print("\n[red][bold]Number of leaf modules in forward pass do not match! See below:[/bold][/red]")
    else:
        print("\n[green][bold]Number of leaf modules in forward pass match![/bold][/green]")

    show_comparison({model_stats.name: mismatches_target, target_model_stats.name: mismatches_model}, as_table=as_table)


def _count_matches(target_tensors, model_tensors, rtol=10e-5, atol=10e-8):
    """Counts the matches between the two given dictionary of tensors

    Args:
        target_tensors      : Dictionary with module name (with attributes) from the target model as key, and a list of all output
                              tensors from modules with said attributes as value.
                              Eg : {"type=torch.nn.modules.batchnorm.BatchNorm2d--eps=1e-05--num_features=16": [tensor1, tensor2, ... tensorN]}
                              This dict, for example, indicates that there were N BatchNorm2d instances in the model, with the given attributes.
                              The tensors in the list are the outputs from each of those batchnorms.
        model_tensors       : Same dictionary as target_tensors, but for the modules from the new model.
        rtol                : Relative tolerance for comparison of tensors. See https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
        atol                : Absolute tolerance for comparison of tensors. See https://numpy.org/doc/stable/reference/generated/numpy.isclose.html

    Returns:
        Dictionary mapping from module name with attributes to tuple (total number of tensors, number of tensors which match).
    """
    module_matches = {}

    for module in target_tensors.keys():
        if module in model_tensors.keys():
            mismatches, matches = find_mismatches(target_tensors[module], model_tensors[module], rtol, atol)
            n_inputs = len(target_tensors[module])
            n_matches = len(matches)
            module_matches[module] = (n_inputs, n_matches)

    return module_matches


def _find_matches(target_tensors, model_tensors, rtol=10e-5, atol=10e-8):
    """Finds the matches between the two given dictionary of tensors.

    Args:
        target_tensors      : Dictionary with module name (with attributes) from the target model as key, and a list of all output
                              tensors from modules with said attributes as value.
                              Eg : {"type=torch.nn.modules.batchnorm.BatchNorm2d--eps=1e-05--num_features=16": [tensor1, tensor2, ... tensorN]}
                              This dict, for example, indicates that there were N BatchNorm2d instances in the model, with the given attributes.
                              The tensors in the list are the outputs from each of those batchnorms.
        model_tensors       : Same dictionary as target_tensors, but for the modules from the new model.
        rtol                : Relative tolerance for comparison of tensors. See https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
        atol                : Absolute tolerance for comparison of tensors. See https://numpy.org/doc/stable/reference/generated/numpy.isclose.html

    Returns:
        (module_matches, modules_without_match)
        modules_matches         : List of tuples (target model module, new model module which matches target module, number of matches).
        modules_without_match   : Sorted list of new model modules which do not have a match.
    """
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
    """Initializes all the leaf modules in both models using a seed which is generated based on their attributes, and
       then passes the same data through both models. Forward hooks are registered in leaf modules based on the other
       arguments, and the outputs from similar modules are compared with each other. Displays the results in tabular
       or json form.

    Args:
        target_model_stats      : Target model wrapped in a ModelSummary object.
        model_stats             : New model wrapped in a ModelSummary object.
        input_shape             : Shape of the input tensor to both the models.
        as_table                : If True, shows the results in tabular form.
                                  Else, prints it as json.
        show_matches            : If True, shows the modules which match completely also in the results.
        modules                 : List of modules names (such as "torch.nn.modules.activation.ReLU"). The forward hooks
                                  are only added to modules of the given types. Makes the brute force comparison of outputs
                                  much faster, since there will be fewer tensors to compare.
        marked_modules_only     : If True, adds the forward hooks to only modules which have been marked for comparison.
                                  (see function mark_module_for_comparison)
                                  Overrides modules and show_matches.
        rtol                    : Relative tolerance for comparison of tensors. See https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
        atol                    : Absolute tolerance for comparison of tensors. See https://numpy.org/doc/stable/reference/generated/numpy.isclose.html

    Returns:
        None.
    """
    x = torch.randn(*input_shape)

    init_weights(target_model_stats.model)
    init_weights(model_stats.model)

    console = Console()

    with console.status("[bold green]Passing input data through models...") as status:
        _, target_outputs = _forward_with_hooks(target_model_stats.model, x, modules, marked_modules_only)
        _, model_outputs = _forward_with_hooks(model_stats.model, x, modules, marked_modules_only)

    if marked_modules_only:
        with console.status("[bold green]Matching marked module inputs... This might take some time...") as status:
            matches, no_matches_modules = _find_matches(target_outputs, model_outputs, rtol, atol)

        for target_module, module, n_matches in matches:
            print(f"Output of [magenta][italic]{module}[/italic] in {model_stats.name}[/magenta] " +
                  f"[green]matches with[/green] output of [magenta][italic]{target_module}[/italic] in {target_model_stats.name} [/magenta]")

        if no_matches_modules:
            print(f"\n[red][bold]No matches found for following modules:")
            for module in no_matches_modules:
                print(f"[magenta]{module} in {model_stats.name}")

        return

    with console.status("[bold green]Matching module inputs... This might take some time...") as status:
        matches = _count_matches(target_outputs, model_outputs, rtol, atol)

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


def compare_final_outputs_in_forward_pass(target_model_stats, model_stats, input_shape, rtol=10e-5, atol=10e-8, target_output_fn=None, model_output_fn=None):
    """Initializes all the leaf modules in both models using a seed which is generated based on their attributes,
       passes the same data through both models, and then compares the outputs of both the models.

    Args:
        target_model_stats      : Target model wrapped in a ModelSummary object.
        model_stats             : New model wrapped in a ModelSummary object.
        input_shape             : Shape of the input tensor to both the models.
        rtol                    : Relative tolerance for comparison of tensors. See https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
        atol                    : Absolute tolerance for comparison of tensors. See https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
        target_output_fn        : Function to extract the tensor for comparison from the target model output.
                                  The model might output a tuple of tensors, for example, consisting of the main output and the output of an auxilliary head.
        model_output_fn         : Same as target_output_fn, but for the new model.

    Returns:
        True if the outputs are equal, else False.
    """
    target_model = target_model_stats.model
    model = model_stats.model

    init_weights(target_model)
    init_weights(model)

    x = torch.randn(*input_shape)

    if target_output_fn:
        out_target = target_output_fn(target_model(x)).detach().numpy()
    else:
        out_target = target_model(x).detach().numpy()

    if model_output_fn:
        out_model = model_output_fn(model(x)).detach().numpy()
    else:
        out_model = model(x).detach().numpy()

    if out_target.shape != out_model.shape:
        return False

    is_equal = np.allclose(out_target, out_model, rtol=rtol, atol=atol)

    if is_equal:
        print("\n[green][bold]Model outputs match![/bold][/green]")
    else:
        print("\n[red][bold]Model outputs do not match![/bold][/red]")

    return is_equal


def _forward_with_hooks(model, x, modules=None, marked_modules_only=False):
    """Registers forward hook in the leaf modules of the model, based on the other arguments.

    Args:
        model               : Pytorch model.
        x                   : Data to forward pass through the model.
        modules             : List of names of modules. If specified, the forward hooks are only added to modules
                              of the kind specified in the list. Otherwise, the hooks are added to all the models.
        marked_modules_only : If True, only leaf modules which have been marked will have the forward hooks registered
                              in them. Overrides the modules argument.

    Returns:
        Tuple of (stats, module_outputs)
        stats               : ModelSummary object with a list of all the modules invoked during forward pass in its db
        module_outputs      : Dictionary with module name with attributes as key and list of output tensors of these modules as values.
                              Eg : {"type=torch.nn.modules.batchnorm.BatchNorm2d--eps=1e-05--num_features=16": [tensor1, tensor2, ... tensorN]}
                              This dict, for example, indicates that there were N BatchNorm2d instances in the model, with the given attributes.
    """
    leaf_modules = get_leaf_modules(model)

    module_outputs = defaultdict(lambda: [])
    stats = ModelSummary(nn.Module(), "model")

    def hook_fn(m, i, o):
        if hasattr(m, "__torchbug_attributes") and m.__torchbug_attributes["name"] != "":
            key = m.__torchbug_attributes["name"]
        else:
            key = get_module_attrs_string(m)
        stats._add_module_to_db(m)
        module_outputs[key].append(o[0].detach().numpy())

    for module in leaf_modules:
        if marked_modules_only:
            if hasattr(module, "__torchbug_attributes"):
                module.register_forward_hook(hook_fn)
        elif modules is None or get_module_name(module) in modules:
            module.register_forward_hook(hook_fn)

    model(x)

    return stats, module_outputs
