import numpy as np

CLASS_INDICES = {
    "math" : 0,
    "stat" : 1,
    "physics" : 2,
    "cs" : 3
}

CLASS_LABELS = [ "math", "stat", "physics", "cs" ]

ALPHA = 1
    

class MultinomialFeature(object):
  """Stores the word counts for a single feature."""

  def __init__(self, word):
    """Initialize the counts to 0."""

    # number of times each class occured
    self.class_counts = np.array([0, 0, 0, 0])

    # number of time each class occured for a given value
    # a map from value -> np.array([x1, x2, x3, x4])
    # where x1 is the number of times "math" was the class
    # when this feature was value
    self.value_counts = {}
    
    self.word = word

  def update(self, value, label):
    """Update the counts with the current training example."""

    # if we haven't seen the value before, initialize counts to 0
    if value not in self.value_counts:
      self.value_counts[value] = np.array([0, 0, 0, 0])

    # increment the count of that class for that value
    label_index = CLASS_INDICES[label.strip()]
    self.value_counts[value][label_index] += 1
    self.class_counts[label_index] += 1

  def get_probs(self, value):
    """Returns the probabilities of each class given the value.
    ALPHA is used for smoothing

    Args:
      value: the query value to return the probabilities for.
    Returns:
      np.array([p1, p2, p3, p4]), where p1 is the probability that
      the training example is "math"
    """
    try:
        counts = self.value_counts[int(value)]
    except KeyError:
        print("never had %d occurences of %s" % (value, self.word))
        counts = [0, 0, 0, 0]
    num = [ float(x + ALPHA) for x in counts ]
    denom = self.class_counts + ALPHA
    return num / denom


class NaiveBayes(object):
  """Stores a list of features and total frequency counts of each class."""

  def __init__(self, dictionary):

    # A list of features
    self.total = np.array([0, 0, 0, 0])
    self.features = [MultinomialFeature(word) for word in dictionary]

  def learn(self, training_example, label):
    """Iterates through the features of a single training example
    updates the feature with the new data.
    """
    for i, feature in enumerate(training_example):
      self.features[i].update(feature, label)
    self.total[CLASS_INDICES[label.strip()]] += 1

  def train(self, training_examples, labels):
    """Iterates through the training examples and calls learn on them."""

    # go through the training examples and count the occurences
    for training_example, label in zip(training_examples, labels):
      self.learn(training_example, label)

  def predict(self, test_example):
    """Call after training to predict the class of a test example."""

    probs = self.features[0].get_probs(test_example[0])
    for i, feature in enumerate(test_example):
      probs *= self.features[i].get_probs(feature)
    total_examples = sum(self.total)
    probs *= self.total
    return CLASS_LABELS[np.argmax(probs)]


def predict_on_test_data():
  train_examples = np.loadtxt("train_examples.csv", delimiter=",")
  print("read feature vectors")
  labels = open("labels", "r").readlines()
  dictionary = open("header.csv").readline().split(",")

  clf = NaiveBayes(dictionary)
  clf.train(train_examples, labels)
  print("trained nb")

  test_data = np.loadtxt("test_examples.csv", delimiter=",")
  with open("test_output_nb.csv", "w") as out_file:
    out_file.write('"id","category"\n')
    for i, test_datum in enumerate(test_data):
      out_file.write('"{}","{}"\n'.format(i,clf.predict(test_datum)))

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def make_confusion_matrix():
  examples = np.loadtxt("train_examples.csv", delimiter=",")
  train_examples = examples[500:]
  test_examples = examples[:500]
  labels = [x.strip() for x in open("labels", "r")]
  train_labels = labels[500:]
  test_labels = labels[:500]
  dictionary = open("header.csv").readline().split(",")
  clf = NaiveBayes(dictionary)
  clf.train(train_examples, train_labels)

  preds = [clf.predict(x) for x in test_examples]
  cm = confusion_matrix(test_labels, preds)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(cm)
  fig.colorbar(cax)
  ax.set_xticklabels([''] + CLASS_LABELS)
  ax.set_yticklabels([''] + CLASS_LABELS)
  plt.title('Naive Bayes Confusion Matrix')
  plt.ylabel('True')
  plt.xlabel('Predicted')
  plt.show()
  print(cm)

  correct = 0.0
  for true_label, pred in zip(test_labels, preds):
      if true_label == pred:
          correct += 1
  print("%f" % (correct / len(test_labels)))




if __name__ == "__main__":
  make_confusion_matrix()
  # predict_on_test_data()
