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


def pad_to_30x30(x):
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
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=64, kernel_size=5, stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=2, stride=2
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3)

        # Flatten the grid:
        self.flatten = nn.Flatten()

        # 2-layer perceptron like specified in the paper
        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
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
        """
        Returns both the size of a grid and the prediction grid

        Note to self: may need to tweak for the same neural network
        layer to process both N_pred and values_pred, so that they're not
        trying to independently learn the weights
        :param x:
        :return:
        """
        batch_size = x.size(0)

        # Predict the size of the grid
        vector_to_hidden_layer = self.fc1(x)
        hidden_layer = self.leaky_relu(vector_to_hidden_layer)
        hidden_layer_to_logits = self.fc2(hidden_layer)
        N_logits = self.softmax(hidden_layer_to_logits)
        N_pred = N_logits.argmax(dim=1)

        # construct a prediction grid
        vector_to_values_hidden_layer = self.fc_values(x)
        values_hidden_layer = self.leaky_relu(vector_to_values_hidden_layer)

        self.fc_grid = nn.Linear(in_features=1024, out_features=30 * 30 * 10)
        vector_to_grid_shape = self.fc_grid(values_hidden_layer)
        vector_as_grid = vector_to_grid_shape.view(batch_size, 30, 30, 10)
        vector_to_image = vector_as_grid.permute(
            0, 3, 1, 2
        )  # massage data into format expected by conv
        conv_layer = self.leaky_relu(self.conv1(vector_to_image))
        softmax = self.softmax2d(conv_layer)
        values_pred = softmax.argmax(dim=1)
        return N_pred, values_pred


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
        if not hasattr(self, "attention"):
            self.attention = nn.Linear(
                512, 1
            )  # Maps each tuple's features to a single scalar

        # Compute attention scores: (num_tuples, 1)
        attention_scores = self.attention(z)  # Shape: (num_tuples, 1)

        # Apply softmax to get attention weights: (num_tuples, 1)
        attention_weights = torch.softmax(
            attention_scores, dim=0
        )  # Softmax over num_tuples

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
        input_or_output, batch_size, vector_size = z.size()
        # stack all the sets in the batch
        z = z.view(batch_size, input_or_output * vector_size)
        z = torch.reshape(z, (1, batch_size, 1024))
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

    Original paper used 4 layers, 512 hidden units
    H output init. scale is 30 in visual tasks
    """

    def __init__(self):
        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512 * 3 * 2)

        self.leaky_relu = nn.LeakyReLU()

        self.batch_norm = nn.BatchNorm1d

    def forward(self, x):
        """
        Maps a vector x in the embedding space to a vector theta in the parameter space
        :param x:
        :return:
        """
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TaskNetwork(nn.Module):
    """
    Task network T that maps from Theta X Z to Z

    Once the parameters Theta have been specified by the Hyper Network,
    it serves as a mapping from Z to Z
    """

    def __init__(self, hidden_dims: list[int]):
        super(TaskNetwork, self).__init__()
        self.input_dim = 512
        self.hidden_dims = hidden_dims
        self.output_dim = 512

        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.num_layers = len(layer_dims)
        self.total_params = 2 * len(layer_dims) * 512

    def _extract_weights_and_biases(self, theta, num_layers):
        """
        Splits theta into weights and biases

        :param theta:
        :return:
        """
        params = theta.view(num_layers, 2, 512)
        return params

    def forward(self, batch, theta):
        """

        :param x: Input tensor to TaskNetwork, vector in embedding space
        :param theta: Tensor output from the HyperNetwork, used to define parameters
        :return:
        """
        x, y = torch.split(batch, 1, dim=0)

        assert (
            theta.numel() == self.total_params
        ), f"Size of theta ({theta.numel()}) does not match total parameters ({self.total_params})"

        params = self._extract_weights_and_biases(theta, self.num_layers)
        for i, (weight, bias) in enumerate(params):
            x = x * weight + bias
            if i < len(params) - 1:
                x = F.relu(x)  # Activation for intermediary layers
        return x.squeeze()


class MetamappingModel(nn.Module):
    """
    TaskManager that coordinates the metamapping learning

    Which is how the arc-agi data is formatted

    Architecture taken from
    """

    def __init__(self):
        super(MetamappingModel, self).__init__()
        self.input_embedder = GridEmbedder()
        self.output_encoder = GridEmbedder()
        self.output_decoder = OutputDecoder()
        self.example_network = ExampleNetwork()
        self.hyper_network = HyperNetwork()
        self.task_network = TaskNetwork([512])

    def encode_batch(self, input, output):
        input = self.input_embedder(input)
        output = self.output_encoder(output)
        batch = torch.stack((input, output))
        return batch


    def forward(self, batch):
        input, output = torch.split(batch, 1, dim=1)

        batch = self.encode_batch(input, output)
        z_task = self.example_network(batch)
        task_params = self.hyper_network(z_task)

        # TODO: figure out implementation of the two training flows

        if np.random.random() > 0.5:
            # task training flow
            z_out_batch = self.task_network(batch, task_params)
            N_pred, values_pred = self.output_decoder(z_out_batch)

            return N_pred, values_pred
        else:
            # metamapping training flow
            z_transformed_task = self.task_network(batch, task_params)



class TaskBatcher(Dataset):
    """
    loads the dataset given a challenges and solutions filepath
    """

    def __init__(self, challenges_file, solutions_file):
        self.challenges_file = challenges_file
        self.solutions_file = solutions_file

        with open(f"./data/{self.challenges_file}", "r") as f:
            self.data = list(json.load(f).items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        task_batch = self.data[idx]
        task_name = task_batch[0]
        task_train_test = task_batch[1]
        return (task_name, task_train_test)

class TrainingDataset(Dataset):
    """
    Loads training data into a batch of tuples of
    (input, output), which have been padded to 30x30 shape with -1
    """
    def __init__(self, train_data):
        self.data = []

        for data_dict in train_data:
            input = pad_to_30x30(load_grid_to_tensor(data_dict['input']))
            output = pad_to_30x30(load_grid_to_tensor(data_dict['output']))
            pair = torch.stack(tensors=(input, output), dim=1)
            self.data.append(pair)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class TaskDataset(Dataset):
    """
    Loads a dataset for a specific task
    """
    def __init__(self, task_name, task_train_test):
        self.task_name: str = task_name
        self.train_data: list[dict] = task_train_test['train']
        self.test_case: list[dict] = task_train_test['test']

        self.train_dataset = TrainingDataset(self.train_data)

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        return {"task_name": self.task_name, "task_tensor": self.train_dataset[idx]}



if __name__ == "__main__":
    # task_manager = TaskManager()
    # with open("data/arc-agi_training_challenges.json", "r") as f:
    #     data = json.load(f)
    #
    # for key in data.keys():
    #     train_and_test_data = data[key]
    #     batch = []
    #     for input_output_pair in train_and_test_data["train"]:
    #         input = input_output_pair['input']
    #         output = input_output_pair['output']
    #         input = load_grid_to_tensor(input)
    #         output = load_grid_to_tensor(output)
    #         input = pad_to_30x30(input)
    #         output = pad_to_30x30(output)
    #         pair = torch.stack(tensors=(input, output), dim=1)
    #         batch.append(pair)
    #
    #     batch = torch.stack(batch, dim=1)
    #
    #     batch = task_manager(batch)
    metamapping_model = MetamappingModel()
    metamapping_loss = nn.MSELoss()
    task_data = TaskBatcher("arc-agi_training_challenges.json", "arc-agi_training_solutions.json")
    for i in range(len(task_data)):
        task_name, task_train_test = task_data[i]
        task_dataset = TaskDataset(task_name, task_train_test)
        # returns a single TaskTensor of shape [1, 2, 1, 30, 30]
        task_name_batch = None # TODO: use language encoders on the task names if necessary
        task_batch = torch.stack([example['task_tensor'] for example in task_dataset], dim=0)
        task_batch = torch.squeeze(task_batch)
        N_pred, task_values_pred = metamapping_model(task_batch)
        breakpoint()




