from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


# Metamapping

# NOTE: end-to-end optimization is used (phew)
# Work based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7777120/
import json


def load_grid_to_tensor(grid):
    """
    Convert a 2D grid (list of lists) to a PyTorch tensor.
    """
    tensor = torch.tensor(grid, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


class GridEmbedder(nn.Module):
    """
    A conv2d network that takes an input grid and converts it into
    the embedding space
    """

    def __init__(self, input_channels=1):
        """
        Constructs the input embedder, a CNN connected to a 2-layer perceptron

        :param input_channels: The number of channels in the grid image, default 1
        :param input_dim: The dimension of the input to determine the initial kernel
                The smallest possible grid size is 1x1 and the largest is 30x30.
        :param output_dim: The dimension of the output
        """
        super(GridEmbedder, self).__init__()
        # conv net they used for visual tasks, will use as baseline
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2)

        self.max_pool = nn.MaxPool2d(kernel_size=3)

        # Flatten the grid:
        self.flatten = nn.Flatten()

        # 2-layer perceptron like specified in the paper
        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)

        self.leaky_relu = nn.LeakyReLU()

    def pad_to_30x30(self, x):
        """Pad the input tensor to 30x30, the maximum grid size for an image"""
        h, w = x.size(2), x.size(3)
        pad_h = max(0, 30 - h)
        pad_w = max(0, 30 - w)

        # Calculate padding for each side
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x):
        # pad the image to 30x30
        x = self.pad_to_30x30(x)

        # Apply convolutions
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))

        # x = self.max_pool(x)
        x = self.flatten(x)

        # pass through convolutional layers
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)

        return x

class OutputDecoder(nn.Module):
    """
    Converts a 512-dimensional tuple in the embedding space
    into an output in the solution space

    Maps from Z to output

    The output consists of two parts:
        N: The dimension of the output grid
        V: The value space, the value of each 'pixel' in the grid
    """
    def __init__(self, N_max=30):
        super(OutputDecoder, self).__init__()
        self.N_max = N_max

        # layer to predict N
        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=30)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()

        # network to predict what values to put in N
        self.fc_values = nn.Linear(in_features=512, out_features=1024)
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, stride=1)
        self.softmax2d = nn.Softmax2d()


    def forward(self, x):
        batch_size = x.size(0)

        # Predict the size of the grid
        vector_to_hidden_layer = self.fc1(x)
        hidden_layer = self.leaky_relu(vector_to_hidden_layer)
        hidden_layer_to_logits = self.fc2(hidden_layer)
        N_logits = self.softmax(hidden_layer_to_logits)
        N_pred = N_logits.argmax(dim=1).item()

        # use the predicted size to construct a prediction grid
        vector_to_values_hidden_layer = self.fc_values(x)
        values_hidden_layer = self.leaky_relu(vector_to_values_hidden_layer)

        self.fc_grid = nn.Linear(in_features=1024, out_features=N_pred * N_pred * 10)
        vector_to_grid_shape = self.fc_grid(values_hidden_layer)
        vector_as_grid = vector_to_grid_shape.view(batch_size, N_pred, N_pred, 10)
        vector_to_image = vector_as_grid.permute(0, 3, 1, 2) # massage data into format expected by conv
        conv_layer = self.leaky_relu(self.conv1(vector_to_image))
        softmax = self.softmax2d(conv_layer)
        results = softmax.argmax(dim=1)
        return results



class TaskRepresentation(nn.Module):
    """
    A basic task representation constructed from
    a support set of (input, target output) tuples

    Which is how the arc-agi data is formatted

    Architecture taken from
    """
    def __init__(self):
        super(TaskRepresentation, self).__init__()
        self.input_embedder = GridEmbedder()
        self.output_encoder = GridEmbedder()
        self.output_decoder = OutputDecoder()

class ExampleNetwork(nn.Module):
    """
    The example network that maps from the set of encoded
    training examples to a single vector, z_task
    """
    def __init__(self):
        super(ExampleNetwork, self).__init__()

        # Map each tuple to the output vector
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.leaky_relu = nn.LeakyReLU()

        self.softmax = nn.Softmax()

    def apply_attention(self, z):
        """
        Applies attention mechanism to condense tuples into a single vector.
        :param z: Tensor of shape (num_tuples, 512)
        :return: Tensor of shape (desired_output_size)
        """
        # Define attention layer if not already defined
        if not hasattr(self, 'attention'):
            self.attention = nn.Linear(512, 1)  # Maps each tuple's features to a single scalar

        # Compute attention scores: (num_tuples, 1)
        attention_scores = self.attention(z)  # Shape: (num_tuples, 1)

        # Apply softmax to get attention weights: (num_tuples, 1)
        attention_weights = torch.softmax(attention_scores, dim=0)  # Softmax over num_tuples

        # Weighted sum of tuples: (512)
        weighted_sum = torch.sum(attention_weights * z, dim=1)  # Shape: (512)

        # Final condensation layer
        condensed = self.fc2(weighted_sum)  # Shape: (desired_output_size)
        return condensed

    def forward(self, z):
        """
        :param z: a set of encoded training examples Set[tuple(
        :return: a single mapped vector
        """

        # NOTE:

        # mathematically, we're trying to find the vector that best maps
        # from the input layer to the output layer, aggregated over each tuple
        # or at least trying to approximate it
        # TODO: make this a more efficient neural net that directly finds that vector
        batch_size, tuples_in_batch, input_or_output, vector_size = z.size()
        num_tuples = batch_size * tuples_in_batch
        # stack all the sets in the batch
        z = z.view(num_tuples, input_or_output * vector_size)
        z = torch.reshape(z, (1, num_tuples, 1024))
        z = self.fc1(z)
        z = self.leaky_relu(z)
        # softmax to avoid gradient loss
        z = self.softmax(z)
        z = self.apply_attention(z)
        return z

class HyperNetwork(nn.Module):
    """
    HyperNetwork H that maps from
    Z (the embedding space) to Theta (the parameter space of the task network T)
    """
    def __init__(self):
        super(HyperNetwork, self).__init__()

    def forward(self, x):
        parameter_space = dict()
        return parameter_space


class TaskNetwork(nn.Module):
    """
    Task network T that maps from Theta X Z to Z

    Once the parameters Theta have been specified by the Hyper Network,
    it serves as a mapping from Z to Z
    """
    def __init__(self, theta: dict):
        super(TaskNetwork, self).__init__()
        self.parameter_space: dict = theta

    def forward(self, x):
        z_task = None
        return z_task




if __name__ == "__main__":
    embedder = GridEmbedder()
    decoder = OutputDecoder()
    with open("data/arc-agi_training_challenges.json", "r") as f:
        data = json.load(f)

    example_network = ExampleNetwork()
    for key in data.keys():
        train_and_test_data = data[key]
        batch = []
        for input_output_pair in train_and_test_data["train"]:
            input = input_output_pair['input']
            output = input_output_pair['output']
            z_input = embedder(load_grid_to_tensor(input))
            z_output = embedder(load_grid_to_tensor(output))

            pair = torch.stack(tensors=(z_input, z_output), dim=1)
            batch.append(pair)

        batch = torch.stack(batch, dim=1)
        z_task = example_network(batch)
        breakpoint()
