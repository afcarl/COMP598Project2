import unittest
import naive_bayes as nb


class TestNaiveBayes(unittest.TestCase):

  def test_train(self):
    learner = nb.NaiveBayes()
    examples = [[0,2,3],[1,4,3],[0,2,3]]
    labels = ["math", "stats", "math"]
    learner.train(examples, labels)
    self.assertEqual(3, len(learner.features))
    self.assertEqual(2, len(learner.features[0].value_counts))
    self.assertEqual(2, len(learner.features[1].value_counts))
    self.assertEqual(1, len(learner.features[2].value_counts))
    self.assertEqual([2,1,0,0], learner.total.tolist())
    self.assertEqual("math", learner.classify([0,2,3]))


if __name__ == '__main__':
  unittest.main()
