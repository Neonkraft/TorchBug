import torch
import torch.nn as nn
import torch.nn.functional as F

from pycaliper.statistics import ModelStatistics
from pycaliper.comparison import (
    compare_module_outputs_in_forward_pass,
    compare_modules_in_forward_pass,
    compare_outputs_forward_pass,
    mark_module_for_comparison,
    mark_all_modules_for_comparison
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    target_model = Net()
    new_model = Net2()

    # Wrap the models in ModelStatistics objects first
    target_model_stats = ModelStatistics(target_model, "Target Model")
    new_model_stats = ModelStatistics(new_model, "New Model")

    show_results_as_table = True

    # See the statistics of the models
    target_model_stats.print(as_table=show_results_as_table)
    new_model_stats.print(as_table=show_results_as_table)

    # Compare the modules of the two models
    new_model_stats.compare(target_model_stats, as_table=show_results_as_table)

    # Compare the outputs from both the models, when they are initialized with the same weights and passed the same input
    compare_outputs_forward_pass(target_model, new_model, input_shape=(1, 3, 32, 32))

    # Compare the modules which are called during the forward pass through the models
    compare_modules_in_forward_pass(target_model_stats, new_model_stats, input_shape=(1, 3, 32, 32), as_table=show_results_as_table)

    # Compare the outputs of each module of the models during forward pass
    compare_module_outputs_in_forward_pass(target_model_stats,
                                           new_model_stats,
                                           input_shape=(1, 3, 32, 32),
                                           show_matches=True,
                                           as_table=show_results_as_table)

    # Mark the modules you want to compare in both models
    mark_all_modules_for_comparison(target_model_stats.model)  # Marking all the leaf modules in the target model
    mark_module_for_comparison(target_model_stats.model.conv2, "Second Convolution")  # Marking a specific convolution with a name

    # Marking only a specific convolution of the new model
    mark_module_for_comparison(new_model_stats.model.conv2, "Second Convolution ahoy!")
    mark_module_for_comparison(new_model_stats.model.fc2, "FC2")  # Marking only a specific linear layer of the new model

    # Compare the outputs of only the marked module of the models
    compare_module_outputs_in_forward_pass(target_model_stats,
                                           new_model_stats,
                                           input_shape=(1, 3, 32, 32),
                                           show_matches=False,
                                           as_table=show_results_as_table,
                                           marked_modules_only=True)  # Compares only marked modules
