import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbug.summary import ModelSummary
from torchbug.comparison import (
    compare_module_outputs_in_forward_pass,
    compare_modules_in_forward_pass,
    compare_final_outputs_in_forward_pass,
    mark_module_for_comparison,
    mark_all_modules_for_comparison
)


class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutions
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 6, 3)

        # Pool and activation
        self.prelu = nn.PReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # Linear Layers
        self.fc1 = nn.Linear(24, 10)
        self.fc2 = nn.Linear(10, 2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = self.prelu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutions
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 6, 3)

        # Pool and activation
        self.relu = nn.ReLU() # Using nn.ReLU instead of F.relu, functionally the same.
        self.pool = nn.MaxPool2d(2, 2)

        # Linear Layers
        self.fc1 = nn.Linear(24, 10)
        self.fc2 = nn.Linear(10, 2)


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x)) # Using relu instead of prelu
        x = self.relu(self.fc2(x))
        return x


if __name__ == "__main__":
    target_model = TargetModel() # Original model which you're trying to emulate
    new_model = NewModel() #New implementation of the target model

    # Wrap the models in ModelSummary objects first
    target_model_stats = ModelSummary(target_model, "Target Model")
    new_model_stats = ModelSummary(new_model, "New Model")

    show_results_as_table = True # Results can be printed as tables or json
    input_shape = (1, 3, 32, 32) # Shape of input data to the models

    # See the counts of various leaf modules in the models
    target_model_stats.print(as_table=show_results_as_table)
    new_model_stats.print(as_table=show_results_as_table)

    # Compare the modules of the two models
    new_model_stats.compare(target_model_stats, as_table=show_results_as_table)

    # Compare the modules which are called during the forward pass through the models
    compare_modules_in_forward_pass(target_model_stats, new_model_stats, input_shape=input_shape, as_table=show_results_as_table)

    # Compare the outputs from both the models, when they are initialized with the same weights and passed the same input
    compare_final_outputs_in_forward_pass(target_model_stats, new_model_stats, input_shape=input_shape, rtol=10e-6, atol=10e-6)

    # Compare the outputs of every module of both models during forward pass
    compare_module_outputs_in_forward_pass(target_model_stats,
                                           new_model_stats,
                                           input_shape=input_shape,
                                           show_matches=True,
                                           as_table=show_results_as_table)

    # This next part gives you more flexibility in comparing the models. By marking specific modules of the target and new models
    # with specific names, compare_module_outputs_in_forward_pass(...) lets you compare the outputs of only those modules.
    # This can be helpful if you want to check if the output of a given module in the new model matches, say, with any of the modules
    # in the target model, or if you want to compare the outputs of two modules which are *supposed* to match.

    # Mark the modules you want to compare in both models
    mark_all_modules_for_comparison(target_model_stats.model)  # Marking all the leaf modules in the target model
    mark_module_for_comparison(target_model_stats.model.conv2, "Second Convolution")  # Marking a specific convolution with a name

    # Marking only specific modules of the new model for comparison
    mark_module_for_comparison(new_model_stats.model.conv2, "Second Convolution")
    mark_module_for_comparison(new_model_stats.model.fc1, "First Linear Layer")
    mark_module_for_comparison(new_model_stats.model.fc2, "Second Linear Layer")

    # Compare the outputs of only the marked module of the models
    compare_module_outputs_in_forward_pass(target_model_stats,
                                           new_model_stats,
                                           input_shape=input_shape,
                                           show_matches=True,
                                           as_table=show_results_as_table,
                                           marked_modules_only=True,  # Compare only marked modules - this is much faster as fewer hooks have to be registered this way
                                           rtol=10e-6,
                                           atol=10e-6
                                           )
