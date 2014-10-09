from collections import Counter, OrderedDict
import operator
import csv
import string


def cleanWords(w):
	lowerW = w.lower()
	noTrailW = lowerW.split('\'', 1)[0]
	return noTrailW

# filter words out of paragraph
def filterWords(words):
	filteredWords = [w for w in words if len(w) > 4]
	cleanedWords = [cleanWords(w) for w in filteredWords]
	return cleanedWords

def fillMap(absMap, allKeys):
	for k in allKeys:
		if k not in absMap:
			absMap[k] = 0
	return absMap

def sortMap(countMap):
	return sorted(countMap.items(), key=lambda t: t[0])


def getFeatures():
	for name in ("test", "train"):
		# Create full map of words
		with open(name + "_input.csv") as f:
			# get list of all words that exist
			csvReader = csv.reader(f, delimiter= ',')
			allWords = []

			for (num, abstract) in csvReader:
				if num == "id":
					continue
				exclude = set(string.punctuation)
				abstract = ''.join(ch for ch in abstract if ch not in exclude)	
				allWords += abstract.split()

			wrdMap = Counter(filterWords(allWords))
			allKeys = wrdMap.keys()
			allKeys.sort()

		# Write header CSV file
		with open(name + "_features/header.csv", 'wb') as headerf:
		    headerWr = csv.writer(headerf)
		    headerWr.writerow(["id"] + allKeys)

		# Get word count for each abstract
		with open(name + "_input.csv") as f, open(name + "_features/features.csv", 'wb') as featuresf:
			# get individual word count for each abstract
			csvReader = csv.reader(f, delimiter= ',')
			csvWriter = csv.writer(featuresf)

			for (num, abstract) in csvReader:
				if num == "id":
					continue
				absCount = Counter(filterWords(abstract.split()))
				absDict = dict(absCount)

				filledCount = sortMap(fillMap(absDict, allKeys))
				valsToWrite = [v for (k, v) in filledCount]

				csvWriter.writerow([num] + valsToWrite)

if __name__ == '__main__':
	getFeatures()