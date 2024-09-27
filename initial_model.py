from torch import nn
from torch.utils.data import Dataset, DataLoader


class MemoryCluster(nn.Module):
    """
    A neural network cluster that can solve a particular task.

    The weights for the task cluster are preserved if it can solve the task well

    If existing clusters can be made, then no new cluster is made
    """

    def __init__(self):
        super().__init__()
        # TODO: allow the cluster to grow and change, or pre-define it?
        # iterate over the training data for each pair until it reaches optimum
        # growing cluster may be too complicated


class InputCNN(nn.Module):
    """
    CNN input network that extracts information from the
    grid being observed and passes it to each of the memory clusters
    """
    def __init__(self):
        super().__init__()

class OutputLayer(nn.Module):
    """
    Aggregates the results from the memory clusters and makes a prediction
    """
    def __init__(self):
        super().__init__()


class BabyBrain(nn.Module):
    """
    Experimental BabyBrain architecture

    inspired by neuroplasticity - grow new neurons in response to stimuli

    However, actually creating a network of interconnected neurons is hard,
    and I am dumb. So instead I'm going with:
    - start with one cluster of nodes, with a preset size and shape (TBD)
    - for each task, iterate over the training pair to fine-tune the cluster
    - Once the cluster is set, move on to the next task
    - Initialize a new cluster
    - First see if the combined output from the existing clusters works
    - If not, initialize a new cluster and fine-tune it

    process needs some refining
    """
    def __init__(self):
        super().__init__()
        self._input = InputCNN()
        self._brain = []
        self._output = OutputLayer()

    def add_cluster(self):
        new_cluster = MemoryCluster()
        self._brain.append(new_cluster)

    def train_clusters(self):
        pass

    def receive_input(self):
        pass


    def output(self):
        pass


class TaskDataset(Dataset):
    """
    Loads the entire task dataset
    """
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass


class Supervisor(nn.Module):
    """
    Supervisor runs the training process
    """
    def __init__(self):
        super().__init__()
        self.training_data = TaskDataset()
        self.train_dataloader = DataLoader(self.training_data, batch_size=1, shuffle=True)



