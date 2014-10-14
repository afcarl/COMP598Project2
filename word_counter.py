"""
A naive implementation of feature extraction. Words are simply counted and output
as a sparse matrix. The main method creates the following files:

train_examples.csv - the feature values of the training set
test_examples.csv - the feature values of the test set
header.csv - the word that each column in the above two files represents

Abstracts are split on white space, lowercased, and any tokens with non-alpha characters are removed.
Furthermore, words with counts lower than MIN_SUM, and words whose class counts are lower than MIN_RANGE are ignored.
"""

import re

WORD_FILTER = re.compile(r"[A-Za-z]{3,}")
MIN_SUM = 5000
MIN_RANGE = 700
LABEL_INDEX = {
        "math": 0,
        "stat": 1,
        "physics": 2,
        "cs": 3,
}


def get_documents(abstracts_file, labels_file=None):
    with open(abstracts_file) as f:
        if labels_file:
            with open(labels_file) as g:
                for line_docs, line_labels in zip(f, g):
                    parts_docs = line_docs.partition(",")
                    parts_labels = line_labels.partition(",")

                    doc_id = parts_docs[0]
                    label_id = parts_labels[0]
                    assert doc_id == label_id

                    doc = parts_docs[2][1:-3]
                    label = parts_labels[2][1:-3]
                    if label == 'category':
                        continue
                    yield doc, label
        else:
            for line_docs in f:
                    parts_docs = line_docs.partition(",")
                    doc = parts_docs[2][1:-3]
                    yield doc, None


def get_words(abstracts_file, labels_file=None):
    for doc, label in get_documents(abstracts_file, labels_file):
        words = []
        tokens = doc.split()
        for token in tokens:
            match = WORD_FILTER.match(token)
            if match:
                word = match.group(0)
                words.append(word.lower())
        yield words, label


def make_dictionary():
    word_counts = {}
    for words, label in get_words("train_input.csv", "train_output.csv"):
        label_index = LABEL_INDEX[label]
        for word in words:
            if word not in word_counts:
                word_counts[word] = [0, 0, 0, 0]
            word_counts[word][label_index] += 1
    dictionary_filtered = {}
    for word in word_counts:
        if is_interesting(word_counts[word]):
            dictionary_filtered[word] = word_counts[word]
    return dictionary_filtered.keys()


def is_interesting(counts):
    if sum(counts) < MIN_SUM:
        return False
    smallest = 100000
    largest = 0
    for count in counts:
        if count > largest:
            largest = count
        if count < smallest:
            smallest = count
    variance = largest - smallest
    if variance < MIN_RANGE:
        return False
    return True


if __name__ == "__main__":
    dictionary = make_dictionary()
    dict_iter = iter(dictionary)
    with open("header.csv", "w") as f:
        f.write(next(dict_iter))
        for word in dict_iter:
            f.write(",%s" % word)
    with open("train_examples.csv", "w") as g:
        for words, label in get_words("train_input.csv", "train_output.csv"):
            vector = [0] * len(dictionary)
            for word in words:
                try:
                    index = dictionary.index(word)
                    vector[index] += 1
                except ValueError:
                    continue
            g.write(",".join([str(x) for x in vector]) + "\n")
    with open("test_examples.csv", "w") as g:
        for words, label in get_words("test_input.csv"):
            vector = [0] * len(dictionary)
            for word in words:
                try:
                    index = dictionary.index(word)
                    vector[index] += 1
                except ValueError:
                    continue
            g.write(",".join([str(x) for x in vector]) + "\n")
            print(label)



