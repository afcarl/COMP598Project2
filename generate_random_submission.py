import csv
import random

categories = ['math', 'cs', 'stat', 'physics']

# Load test_set
test_set = []
with open('test_input.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader, None)  # skip the header
    for sample in reader:        
        test_set.append(sample)

# Write a random category to the csv file for each example in test_set
test_output_file = open('test_output_random.csv', "wb")
writer = csv.writer(test_output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL) 
writer.writerow(['id', 'category']) # write header
for sample in test_set:
    random_category = categories[random.randint(0,3)]
    row = [sample[0], random_category]
    writer.writerow(row)

test_output_file.close()
