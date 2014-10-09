import numpy as np

CLASS_INDICES = {
    "math" : 0,
    "stats" : 1,
    "physics" : 2,
    "cs" : 3
}

CLASS_LABELS = [ "math", "stats", "physics", "cs" ]

ALPHA = 1
    

class MultinomialFeature(object):
  """Stores the occurences for a single feature."""

  def __init__(self):
    """Initialize the counts to 0."""

    # number of times each class occured
    self.class_counts = np.array([0, 0, 0, 0])

    # number of time each class occured for a given value
    # a map from value -> np.array([x1, x2, x3, x4])
    # where x1 is the number of times "math" was the class
    # when this feature was value
    self.value_counts = {}

  def update(self, value, label):
    """Update the counts with the current training example."""

    # if we haven't seen the value before, initialize counts to 0
    if value not in self.value_counts:
      self.value_counts[value] = np.array([0, 0, 0, 0])

    # increment the count of that class for that value
    label_index = CLASS_INDICES[label]
    self.value_counts[value][label_index] += 1

  def get_probs(self, value):
    """Returns the probabilities of each class given the value.
    ALPHA is used for smoothing

    Args:
      value: the query value to return the probabilities for.
    Returns:
      np.array([p1, p2, p3, p4]), where p1 is the probability that
      the training example is "math"
    """
    counts = self.value_counts[value]
    return (counts + ALPHA) / (self.class_counts + ALPHA)


class NaiveBayes(object):
  """Stores a list of features and total frequency counts of each class."""

  def __init__(self):

    # A list of features
    self.features = []
    self.total = np.array([0, 0, 0, 0])

  def learn(self, training_example, label):
    """Iterates through the features of a single training example
    updates the feature with the new data.
    """
    for i, feature in enumerate(training_example):
      self.features[i].update(feature, label)
    self.total[CLASS_INDICES[label]] += 1

  def train(self, training_examples, labels):
    """Iterates through the training examples and calls learn on them."""

    # instantiate the features
    for feature in training_examples[0]:
      self.features.append(MultinomialFeature())
    # go through the training examples and count the occurences
    for training_example, label in zip(training_examples, labels):
      self.learn(training_example, label)

  def classify(self, test_example):
    """Call after training to predict the class of a test example."""

    probs = self.features[0].get_probs(test_example[0])
    for i, feature in enumerate(test_example):
      probs *= self.features[i].get_probs(feature)
    total_examples = sum(self.total)
    probs *= self.total
    return CLASS_LABELS[np.argmax(probs)]
