## The information is stored in two files:

### arc-agi_training-challenges.json
    contains input/output pairs that demonstrate reasoning pattern to be applied to the "test" input for each task. This file and the corresponding solutions file can be used as training for your models.
### arc-agi_training-solutions.json
    contains the corresponding task "test" outputs (ground truth).
### arc-agi_evaluation-challenges.json
    contains input/output pairs that demonstrate reasoning pattern to be applied to the "test" input for each task. This file and the corresponding solutions file can be used as validation data for your models.
### arc-agi_evaluation-solutions.json
    contains the corresponding task "test" outputs (ground truth).
### arc-agi_test-challenges.json
    this file contains the tasks that will be used for the leaderboard evaluation, and contains "train" input/output pairs as well as the "test" input for each task. Your task is to predict the "test" output. Note: The file shown on this page is a placeholder using tasks from arc-agi_evaluation-challenges.json. When you submit your notebook to be rerun, this file is swapped with the actual test challenges.
### sample_submission.json
    a submission file in the correct format

## Each task contains a dictionary with two fields:

### "train": demonstration input/output pairs. It is a list of "pairs" (typically 3 pairs).
### "test": test input - your model should predict the output.
### A "pair" is a dictionary with two fields:

#### "input": the input "grid" for the pair.
#### "output": the output "grid" for the pair.
#### A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid size is 1x1 and the largest is 30x30.

#### The data on this page should be used to develop and evaluate your models. When notebooks are submitted for rerun, they are scored using 100 unseen tasks found in the rerun file named arc-agi_test_challenges.json. The rerun tasks will contain train pairs of inputs and outputs as well as the tasks test input. Your algorithm must predict the test output. The majority of the 100 tasks used for leaderboard score only have one test input that will require a corresponding output prediction, although for a small number of tasks, you will be asked to make predictions for two test inputs.


Copied from https://www.kaggle.com/competitions/arc-prize-2024/data
