import numpy as np
import random
from collections import Counter
import copy

class DecisionTree(object):
	''' Implementation on the decision tree classifier. Supports binary, discrete, 
	or continuous features. Supports an arbitrary number of output classes. To use
	first construct the tree using train() and then classify new instances using 
	classify.
	'''
	class TreeNode(object):
		''' Represents a single node within the decision tree. Contains a test and
		the distribution of training data at this node. Also contain children nodes.
		'''
		def __init__(self, test_function, data, num_classes):
			''' Initialize all parameters of the tree node.
			@param test_function A boolean function representing the test the node should perform.
			@param data The data used on this node of the tree used to calculate the distribution of training instances.
			@param num_classes The number of classes we can predict upon.
			'''
			# Initialize nodes that will go to upon the outcome of a test.
			self.false_node = None
			self.true_node = None
			self.parent = None
			# Save the test function.
			self.test_function = test_function
			# Calculate sample distribution.
			self.class_dist = [0.0] * num_classes
			for entry in data[:,-1]:
				self.class_dist[int(entry)] += 1
			self.class_dist = [x / float(len(data)) for x in self.class_dist]
			
		def perform_test(self, instance):
			''' Return the boolean result of performing a test on a data instance.
			@param instance The data instance we wish to perform the test on.
			@return True of False based on whether the test passes.
			'''
			return self.test_function(instance)
			
		def is_leaf(self):
			'''@return true if we are a leaf. '''
			return self.false_node == None and self.true_node == None
			
		def get_class(self, mode='MAX'):
			''' Find the class the data instance belongs too assuming this is a leaf node.
			@param mode = 'MAX' if we want to choose the majority class.
			@param mode = 'DIST' if we want to choose based upon the distribution of training data.
			@return An integer representing the class.
			'''
			if mode == 'MAX':
				return np.argmax(self.class_dist)
			elif mode == 'DIST':
				x = random.random()
				total = 0
				for ix, entry in enumerate(class_dist):
					total += entry
					if x <= total:
						return ix
				
			
	def __init__(self, test_size=0.2):
		''' Sets any hyperparameters of the model. 
		@param test_size Float between 0 and 1 represent the percent of the train set that should be used for pruning.
		'''
		#TODO: Implement any hyperparameters. Are there any? %train/test.
		self.test_size = test_size
		self.train_size = 1 - test_size
		#TODO: Include option to generate graphs (pruning graphs).
	
	def fit(self, train_xs, train_ys):
		''' Given a m*n matrix containing training instances, construct the tree using a greedy
		algorithm to maximize information gain for the classifier.
		@param data m*n matrix containing the training instances.
		'''
		def get_feature_types(data):
			''' Checks each column of the data set to check if its a binary, discrete, or continous feature. 
			Populates self.feature_types where the key is the index of the column and the value is either 
			BIN, DISC, or CONT.
			'''
			self.feature_types = {}
			for ix, column in enumerate(data[:,:-1].transpose()):
				continuous = False
				boolean = True		# Assumes binary values will be 0 or 1.
				for entry in column:
					if (not isinstance(entry, (int, long))) and (not entry.is_integer()):
						continuous = True
						break
					elif entry not in (0, 1):
						boolean = False	
				if continuous:
					self.feature_types[ix] = 'CONT'
				elif boolean:
					self.feature_types[ix] = 'BIN'
				else:
					self.feature_types[ix] = 'DISC'
		
		def get_number_classes(data):
			''' Count the number of classes in the training set. Stores result in self.num_classes. '''
			classes = []
			for entry in data[:, -1]:
				if entry not in classes:
					classes.append(entry)
			self.num_classes = max(classes) + 1
		print train_xs.shape
		train_ys = train_ys.reshape((train_ys.shape[0], 1))
		data = np.append(train_xs, train_ys, 1)
		print data.shape
		get_feature_types(data)
		get_number_classes(data)
		# Build the tree with train % of the data.
		self.predict_tree =self.build_tree(data[:data.shape[0]*self.train_size,:])
		#self.predict_tree = self.build_tree(data)
		# Prune the tree with test % of the data.
		#self.prune(data[data.shape[0]*self.train_size:, :])
		
	def build_tree(self, data, depth = 0):
		''' Recusively build a tree by calculating information gain for a test on each feature.
		Then select the feature that maximizes IG and build sub-trees for the data split on 
		this test. Recurses until classification is perfect.
		@param data
		'''
		def split_data_on_feature(p_data, p_feature):
			''' Given a feature, find out if that feature is binary, discrete, or continous then
			split the dataset on that data. 
			@param p_data The data we want to split.
			@param feature The id of the feature we want to split on.
			@return (TreeNode, TrueData, FalseData) The split data and their respective TreeNodes.
			'''
			# Find out type of feature.
			feature_type = self.feature_types[p_feature]
			# Create test based on feature type.
			if feature_type == 'BIN':
				def test(instance): return instance[p_feature] == 1;
			elif feature_type == 'DISC' or feature_type == 'CONT':
				mean = np.mean(p_data[:,p_feature])
				def test(instance): return instance[p_feature] >= mean;
			# Split data.
			indices = [test(row) for row in p_data]
			indices = np.array(indices, dtype=bool)
			if indices.size == 0:
				return -1
			false_data = data[~indices, :]
			true_data = data[indices, :]
			tree_node = DecisionTree.TreeNode(test, p_data, self.num_classes)
			return (tree_node, true_data, false_data)
			
		def calculate_entropy(p_data):
			''' Given a dataset, calculate its entropy.
			@param data The dataset which we want the entropy of.
			@return entropy of the data.
			'''
			# Calculate entropy. Note more than two classes.
			if len(p_data) == 0:
				return 0
			counts = np.bincount(p_data[:,-1].astype(int))
			probs = counts/float(len(p_data))
			terms = -1 * probs * np.log(probs)
			terms[np.isnan(terms)] = 0
			return np.sum(terms)
			
		# Split data on each feature and calculate IG for each split. Find test with max info gain.
		max_IG = 0
		best_test = None
		base_entropy = calculate_entropy(data)
		best_true = None
		best_false = None
		if base_entropy == 0:
			node = DecisionTree.TreeNode(None, data, self.num_classes) 
			return node
		for i in xrange(0, data.shape[1]-1):
			result = split_data_on_feature(data, i)
			if result == -1:
				continue
			(new_node, true_data, false_data) = result
			true_ratio = float(len(true_data))/float(len(data))
			new_entropy = true_ratio * calculate_entropy(true_data) + (1-true_ratio) * calculate_entropy(false_data)
			info_gain = base_entropy - new_entropy
			if info_gain >= max_IG:
				best_test = new_node
				max_IG = info_gain
				best_true = true_data
				best_false = false_data
		
		# If perfect classification, stop.
		if Counter(best_test.class_dist)[0] == len(best_test.class_dist) - 1:
			return best_test
		if depth > 10:
			return best_test
		# Recurse on branches of best test.
		if len(best_false) > 0:
			false_node = self.build_tree(best_false, depth + 1)
			false_node.parent = best_test
			best_test.false_node = false_node
		if len(best_true) > 0:	
			true_node = self.build_tree(best_true, depth + 1)	
			true_node.parent = best_test
			best_test.true_node = true_node
		return best_test
	
	def prune(self, test_data):
		''' Assumes the tree for the classifier has been built. Use a test set to recursively 
		the tree until the accuracy doesn't improve anymore.
		@param test_data
		'''
		def remove_test_recursive(p_test, root):
			''' Return a copy of the decision tree with the given test removed.
			@param test
			@return 
			'''
			# Recursively traverse the tree. If we're not a leaf, find the best accuracy from removing leaves in this sub tree.
			if not p_test.is_leaf():
				if p_test.false_node != None:
					(prune_false_acc, prune_false_tree) = remove_test_recursive(p_test.false_node, root)
				else:
					prune_false_acc = 0
				if p_test.true_node != None:
					(prune_true_acc, prune_true_tree) = remove_test_recursive(p_test.true_node, root)
				else:
					prune_true_acc = 0
				if (prune_false_acc > prune_true_acc):
					return (prune_false_acc, prune_false_tree)
				else:
					return (prune_true_acc, prune_true_tree)
			# If we are a leaf, remove and calculate accuracy on the root. Read leaf after.
			node = p_test
			# Remove the node form the parent tree.
			if p_test.parent.false_node == node:
				p_test.parent.false_node = None
			else:
				p_test.parent.true_node = None
			acc = calculate_accuracy(root)
			new_tree = copy.deepcopy(root)
			p_test.parent.false_node = node
			return (acc, new_tree)			
			
		def calculate_accuracy(p_tree):
			''' Given a tree, classify each instance from test_data and report the accuracy.
			@param tree
			@return
			'''
			num_correct = 0
			for ex in test_data:
				if (self.predict(ex[:-1], p_tree) == ex[-1]):
					num_correct += 1
			return num_correct / float(test_data.shape[0])
		# Loop until accuracy stops improving
		initial_accuracy = calculate_accuracy(self.predict_tree)
		print initial_accuracy
		new_accuracy = initial_accuracy+1
		while (new_accuracy >= initial_accuracy):
			initial_accuracy = new_accuracy
			# Try removing a test and keeping the tree with the best accuracy. Give it a copy each time.
			(new_accuracy, new_tree) = remove_test_recursive(self.predict_tree, self.predict_tree)
		print new_accuracy
		self.predict_tree = new_tree
		
	def predict(self, instance, tree=None):
		''' Given an unseen instance, classify it using the trained tree.
		@param instance
		@return
		'''
		if (tree == None):
			tree = self.predict_tree
		# Predict the class of instance.
		node = tree
		while (not node.is_leaf()):
			if node.perform_test(instance) == True:
				if node.true_node == None:
					break
				node = node.true_node
			else:
				if node.false_node == None:
					break
				node = node.false_node
		return node.get_class()

def test_num_classes(data, num_classes):
	dt = DecisionTree()
	dt.train(data)
	if dt.num_classes == num_classes:
		return True
	return False
	
def test_feature_types(data, feature_types):
	dt = DecisionTree()
	dt.train(data)
	if dt.feature_types == feature_types:
		return True
	return False

def test_boolean_functions():
	dt = DecisionTree()
	# X1 AND X2
	data_and = np.array([[1,1,1],[1,0,0],[0,1,0],[0,0,0]])
	# X1 XOR X2
	data_xor = np.array([[1,1,0],[1,0,1],[0,1,1],[0,0,0]])
	# (X1 AND X2) OR (X2 AND X3)
	data_long = np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,1,1,1],[1,0,0,0],[1,0,1,0],[1,1,0,1],[1,1,1,1]])
	# Train and test.
	dt.train(data_long)
	for test in data_long:
		features = test[:-1]
		print dt.predict(features) == test[-1]

if __name__ == '__main__':
	dt = DecisionTree()
	data_two_classes = np.array([[0, 1, 4.5, 0], [1, 1, 0, 1], [0, 3.0, 1, 0], [0, 7, 0, 1]])
	data_five_classes = np.array([[0, 1, 0], [1, 1, 1], [0, 3, 3], [4.5, 7, 1], [1, 0, 7], [0, 4, 0], [4.5, 8, 10], [1, 6, 1], [0, 5, 10], [4.5, 7, 1]])
	print test_feature_types(data_two_classes, {0:'BIN', 1:'DISC', 2:'CONT'})
	print test_num_classes(data_two_classes, 2)
	print test_num_classes(data_five_classes, 11)
	#dt.train(data_five_classes)
	#for test in data_five_classes:
	#features = test[:-1]
	#print dt.predict(features) == test[-1]
	test_boolean_functions()
	
