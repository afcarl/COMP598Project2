from sklearn import neighbors, naive_bayes, tree, svm, linear_model, ensemble
import numpy as np
import csv
from gensim import models, corpora,matutils
import get_lsi
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
from sklearn.decomposition import TruncatedSVD
import sklearn.metrics.pairwise as smp
from collections import Counter
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def generate_submission_bigrams(classifier, transform):

	sub = open('submission_bigram_lr_weighted.csv', 'w')
	sub.write('"id","category"\n')
	
	i = 0
	test_data = open("raw_test_data.txt").readlines()
	matrix = transform.transform(test_data)
	for entry in matrix:
		predict = classifier.predict(entry)
		if predict == 0:
			predict = "cs"
		elif predict == 1:
			predict = "math"
		elif predict == 2:
			predict = "stat"
		elif predict == 3:
			predict = "physics"
		sub.write('"%d","%s"\n' % (i, predict))
		i += 1
		print i
	sub.close()		
import copy
def generate_ngrams(corpus, l,h, cutoff, lsa=False, count=False):
	stopwords = ['to', 'the', 'between', 'recent', 'work', 'than', 'for', 'we', 'well', 'what', 'when', 'are', 'be', 'they', 'would', 'will', 'each', 'do', 'our', 'very', 'these', 'then', 'can', 'have', 'new', 'not', 'such', 'some', 'so', 'show', 'only', 'one', 'more', 'used', 'using', 'both', 'there', 'no', 'use', 'from', 'more', 'less', 'also','because', 'allow', 'allowed', 'allowing', 'affect', 'apply', 'give', 'become', 'where', 'were', 'while', 'many', 'most', 'about', 'like', 'after', 'before', 'all', 'even', 'ever', 'other', 'within', 'widely', 'whose', 'whether', 'useful', 'over', 'under', 'two', 'every', 'without', 'usually', 'through', 'three', 'those', 'thereby', 'therefore', 'suitable', 'suggest', 'suggested','study', 'still', 'since', 'given', 'several', 'same', 'propose', 'proposed', 'provide', 'provided', 'previously', 'present', 'presented', 'often', 'obtain', 'obtained', 'effective', 'made', 'highly', 'here', 'another','approach', 'article','based','being','called', 'cannot', 'case', 'consider', 'define', 'defined', 'denote', 'describe', 'describing', 'develop', 'developed', 'different', 'during', 'early', 'et', 'further', 'having', 'improve', 'important', 'into', 'introduce', 'known']
	# So numbers aren't included.
	for i in xrange(0, 10000):
		stopwords.append(str(i))
	n = 1
	if count == True:
		v = CountVectorizer(ngram_range=(l,h), min_df=cutoff,stop_words=set(stopwords))
	else:
		v = TfidfVectorizer(ngram_range=(l,h), min_df=cutoff,stop_words=set(stopwords))
	
	d = v.fit_transform(corpus)	
	if not lsa:
		return d, v

	svd = TruncatedSVD(n_components=100)
	return svd.fit_transform(d), svd, v

def convert_ys_to_int(raw_ys):
	y = np.zeros((len(raw_ys),))
	for i in xrange(0, len(raw_ys)):
		val = raw_ys[i].strip().strip("\"")
		if val == "cs":
			y[i] = 0
		elif val == "math":
			y[i] = 1
		elif val == "stat":
			y[i] = 2
		elif val == "physics":
			y[i] = 3
		else:
			y[i] = 0
	return y
	
def get_folds(corpus, ys, k):
	fold_size = len(corpus)/k
	x_folds = []
	y_folds = []
	for i in xrange(0, k):
		x_folds.append(corpus[i*fold_size:(i+1)*fold_size])
		y_folds.append(ys[i*fold_size:(i+1)*fold_size].tolist())
	return x_folds, y_folds

def get_train_and_test(x_folds, y_folds, i, k):
	train_xs = []
	test_xs = []
	train_ys = []
	test_ys = []
	for j in xrange(0, k):
		if i == j:
			test_xs += x_folds[j]
			test_ys += y_folds[j]
		else:
			train_xs += x_folds[j]
			train_ys += y_folds[j]

	return train_xs, test_xs, train_ys, test_ys

def oversample_stats(train_xs, train_ys):
	print "OVERSAMPLING"
	l = len(train_ys)
	for i in xrange(0, l):
		if train_ys[i] == 2:
			train_ys.append(train_ys[i])
			train_xs.append(train_xs[i])

	return train_xs, train_ys
		
from sklearn.metrics import roc_curve
def run_k_folds_number_features(corpus, ys, classifier, k, low, high):
	x_folds, y_folds = get_folds(corpus, ys, k)
	for size in [0.00015, 0.0005,0.0010,0.005,0.20,0.25]:
		overall_accuracy = 0
		for i in xrange(0, k):
			
			train_xs, test_xs, train_ys, test_ys = get_train_and_test(x_folds, y_folds, i, k)
			#train_xs, train_ys = oversample_stats(train_xs, train_ys)
			print "FOLD", i
			train_xs, transform = generate_ngrams(train_xs, low, high, size, False)
			print len(transform.get_feature_names())
			#print transform.get_feature_names()
			matrix = transform.transform(test_xs)
			z = 0
			classifier.fit(train_xs, train_ys)
			num_correct = 0
			wrong_dict = {0:0, 1:0, 2:0, 3:0}
			probs = classifier.predict_proba(matrix)
			predict = np.zeros((matrix.shape[0],1))
			for entry in matrix:
				predict[z] = classifier.predict(entry)
				if predict[z] == test_ys[z]:
					num_correct += 1
				else:
					wrong_dict[test_ys[z]] += 1
				z += 1
				#print float(num_correct)/z
			
			cm = confusion_matrix(predict, test_ys)
			labels = ['cs', 'math', 'stat', 'physics']
			fig = plt.figure()
			ax = fig.add_subplot(111)
			cax = ax.matshow(cm)
			fig.colorbar(cax)
			ax.set_xticklabels([''] + labels)
			ax.set_yticklabels([''] + labels)
			plt.xlabel('Predicted')
			plt.ylabel('True')
			plt.show()
			
			fpr = dict()
			tpr = dict()
			y_score = np.zeros((len(test_ys), 4))
			for j in xrange(0, len(test_ys)):
				y_score[j,test_ys[j]] = 1
			print probs.shape
			print y_score.shape
			for j in xrange(0, 4):
				print j
				fpr[j], tpr[j], _ = roc_curve(y_score[:, j], probs[:, j])
			
			plt.figure()
			classes = ['cs', 'math', 'stat', 'physics']
			for i in xrange(0,4):
				plt.plot(fpr[i], tpr[i], label='ROC curve of class {0}'
						                       ''.format(classes[i]))

			plt.plot([0, 1], [0, 1], 'k--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')

			plt.legend(loc="lower right")
			plt.show()
			print wrong_dict
			current_accuracy = float(num_correct)/len(test_ys)
			print i, ": ", current_accuracy
			overall_accuracy += current_accuracy
		overall_accuracy /= float(k)
		print "Size %d: %f" % (size, overall_accuracy)

def run_k_folds_n_grams(corpus, ys, classifier, k):
	x_folds, y_folds = get_folds(corpus, ys, k)

	for (low, high) in [(1,1), (2,2), (3,3), (1, 2), (1, 3), (2, 3)]:
		overall_accuracy = 0
		for i in xrange(0, k):
			train_xs, test_xs, train_ys, test_ys = get_train_and_test(x_folds, y_folds, i, k)
			
			train_xs, transform = generate_ngrams(train_xs, low, high, 50000, False)
			matrix = transform.transform(test_xs)
			z = 0
			classifier.fit(train_xs, train_ys)
			num_correct = 0
			for entry in matrix:
				if classifier.predict(entry) == test_ys[z]:
					num_correct += 1
				z += 1
				
			
			
			current_accuracy = float(num_correct)/len(test_ys)
			print i, ": ", current_accuracy
			overall_accuracy += current_accuracy
		overall_accuracy /= float(k)
		print "(%i, %i)-grams: %f" % (low, high, overall_accuracy)

def run_k_folds_representation(corpus, ys, classifier, k):
	corpus = corpus[0:20000]
	ys = ys[0:20000]
	x_folds, y_folds = get_folds(corpus, ys, k)

	for rep in ['lsi']:# ['count', 'tfidf', 'lsi']:
		overall_accuracy = 0
		for i in xrange(0, k):
			train_xs, test_xs, train_ys, test_ys = get_train_and_test(x_folds, y_folds, i, k)
			
			if rep == 'count':
				train_xs, transform = generate_ngrams(train_xs, 1, 1, 0.1, False, count=True)
				matrix = transform.transform(test_xs)
			elif rep == 'tfidf':
				train_xs, transform = generate_ngrams(train_xs, 1, 1, 0.1, False)
				matrix = transform.transform(test_xs)
			elif rep == 'lsi':
				train_xs, svd, transform = generate_ngrams(train_xs, 1, 1, 0.001, True)
				matrix = transform.transform(test_xs)
				matrix = svd.transform(matrix)
			z = 0
			classifier.fit(train_xs, np.array(train_ys))
			num_correct = 0
			predict = np.zeros((matrix.shape[0],1))
			for entry in matrix:
				predict[z] = classifier.predict(entry)
				if predict[z] == test_ys[z]:
					num_correct += 1
				z += 1
				
			cm = confusion_matrix(predict, test_ys)
			labels = ['cs', 'math', 'stat', 'physics']
			fig = plt.figure()
			ax = fig.add_subplot(111)
			cax = ax.matshow(cm)
			fig.colorbar(cax)
			ax.set_xticklabels([''] + labels)
			ax.set_yticklabels([''] + labels)
			plt.xlabel('Predicted')
			plt.ylabel('True')
			plt.show()
			
			plt.matshow(cm)
			plt.colorbar()
			plt.ylabel('True label')
			plt.xlabel('Predicted label')
			plt.show()
			
			current_accuracy = float(num_correct)/len(test_ys)
			print i, ": ", current_accuracy
			overall_accuracy += current_accuracy
		overall_accuracy /= float(k)
		print "%s: %f" % (rep, overall_accuracy)
	
	
	
import decision_tree
def run_k_folds_custom_dt(corpus, ys, k):
	x_folds, y_folds = get_folds(corpus, ys, k)
	classifier = decision_tree.DecisionTree()
	overall_accuracy = 0
	for i in xrange(0, k):
		train_xs, test_xs, train_ys, test_ys = get_train_and_test(x_folds, y_folds, i, k)
		train_xs, svd, transform = generate_ngrams(train_xs, 1, 2, 50000, True)
		matrix = transform.transform(test_xs)
		matrix = svd.transform(matrix)
		z = 0
		
		classifier.fit(train_xs[0:10000,:], np.array(train_ys[0:10000]))
		num_correct = 0
		predict = np.zeros((matrix.shape[0],1))
		for entry in matrix:
			predict[z] = classifier.predict(entry)
			if  predict[z] == test_ys[z]:
				num_correct += 1
			z += 1
			
		
		cm = confusion_matrix(predict, test_ys)
		plt.matshow(cm)
		plt.colorbar()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()
			
		current_accuracy = float(num_correct)/len(test_ys)
		print i, ": ", current_accuracy
		overall_accuracy += current_accuracy
	overall_accuracy /= float(k)
	print "Overall: %f" % (rep, overall_accuracy)
if __name__ == "__main__":
	knn = neighbors.NearestCentroid()
	nb = naive_bayes.GaussianNB()
	dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
	sv = linear_model.SGDClassifier()
	svc = svm.SVC()
	lr = linear_model.LogisticRegression()
	custom_dt = decision_tree.DecisionTree()
	classifier = lr
	
	input_reader = csv.reader(open('train_output.csv','rb'), delimiter=',')
	i = 0
	raw_ys = []
	for tid, subject in input_reader:
		raw_ys.append(subject)
		i += 1
		print "Reading CSV (%s): Line %i" % ('trainout', i)
	
	
	corpus = open("raw_train_data.txt").readlines()
	raw_ys = raw_ys[1:]
	y = convert_ys_to_int(raw_ys)	
	run_k_folds_number_features(corpus, y, classifier, 3, 1, 2)
	
	#classifier.fit(train_xs, train_ys)
	
	
	'''
	train_xs, transform = generate_ngrams(corpus, 1, 2, 50000, False)
	classifier.fit(train_xs, y)
	generate_submission_bigrams(classifier, transform)
	'''
	#run_k_folds_n_grams(corpus, y, classifier, 5)
	#run_k_folds_representation(corpus, y, classifier, 5)
	#run_k_folds_custom_dt(corpus, y, 5)
	#print "Training."
	#split = 0.8
	#l = xs.shape[0]
	#xs_train = xs[0:l*split, :]
	#xs_test = xs[int(l*split):, :]
	#ys_train = y[0:l*split]
	#ys_test = y[l*split:]
	#classifier.fit(xs_train, ys_train)
	#num_correct = 0
	#for i in xrange(0, len(ys_test)):
		#if classifier.predict(xs_test[i]) == ys_test[i]:
		#	num_correct += 1
		#print float(num_correct)/(i+1)
	#print float(num_correct)/len(ys_test)
	
	
	
	
