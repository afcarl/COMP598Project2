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
  def __init__(self):
    # number of time each value occured for each class
    self.value_counts = {}
    self.class_counts = np.array([0, 0, 0, 0])

  def update(self, value, label):
    if value not in self.value_counts:
      self.value_counts[value] = np.array([0, 0, 0, 0])
    label_index = CLASS_INDICES[label]
    self.value_counts[value][label_index] += 1

  def get_probs(self, value):
    result = []
    counts = self.value_counts[value]
    return (counts + ALPHA) / (self.class_counts + ALPHA)


class NaiveBayes(object):

  def __init__(self):
    # A list of features
    self.features = []
    self.total = np.array([0, 0, 0, 0])

  def learn(self, training_example, label):
    for i, feature in enumerate(training_example):
      self.features[i].update(feature, label)
    self.total[CLASS_INDICES[label]] += 1

  def train(self, training_examples, labels):
    # instantiate the features
    for feature in training_examples[0]:
      self.features.append(MultinomialFeature())
    # go through the training examples and count the occurences
    for training_example, label in zip(training_examples, labels):
      self.learn(training_example, label)

  def classify(self, test_example):
    probs = self.features[0].get_probs(test_example[0])
    for i, feature in enumerate(test_example):
      probs *= self.features[i].get_probs(feature)
    total_examples = sum(self.total)
    probs *= self.total
    return CLASS_LABELS[np.argmax(probs)]
