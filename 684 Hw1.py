import pandas as pd
import sys
from collections import Counter
import math

# Input parameters
L = int(sys.argv[1])
K = int(sys.argv[2])
training_set_path = sys.argv[3]
validation_set_path = sys.argv[4]
test_set_path = sys.argv[5]
to_print = sys.argv[6]

# read in the training data
training_data = pd.read_csv(training_set_path)
training_data_column_values = list(training_data.columns.values)

class Node:

	def __init__(self, name=None, parent=None):
		self.name = name
		self.parent = parent
		self.chidren_list = None
		self.value_list = None

class Tree:

	def __init__(self):
		self.root = self.add_node()
		self.data = training_data
		self.target_attribute = training_data_column_values[-1]
		self.feature_list = training_data_column_values.remove(self.target_attribute)
		self.current_node = None

	
	# feature_to_split = choose_the_best_feature()
	# traverse till the leaf and add the node to the corresponding leaf
	def add_node(self, node):
		# determine if the root exists
		if self.current_node == None:
			root_name = self.choose_the_best_feature()	

	# Return the next attribute to split based on
	# the information gain calculation
	def choose_the_best_feature(self):
		entrophy = self.calculate_entrophy_list()

		# return max(IG_list)
		pass

	def calculate_entropy(data_list):
		# data_list includes all the target values 
		# for the current value under the feature
		entropy = 0
		cnt = Counter(target for target in data_list)
		total_count = len(data_list)
		for target in cnt:
			ratio = cnt[target]/total
			entropy += -ratio * math.log(posProb, 2)
		return entropy
		
	
	def calculate_entrophy_list(self):
		# if the current node hasn't been selected, this means 
		# the tree has no root and we are to calculate the 
		# entropy of the original whole data to select the root
		if self.current_node = None:
			old_entrophy = 
		pass

	def calculate_variance_impurity(self):
		pass

	def calculate_information_gain(self, old_entrophy):
		pass	

	def print_decision_tree():
		# start from the root using pre-order traversal
		# to print out the decision tree
		pass

	def calculateEntropyOfList(list):  
    cnt = Counter(x for x in list)
    totalInstances = len(list)
    proportionOfInstances = [x / totalInstances for x in cnt.values()]
    sumOfEntropies=0
    for x in proportionOfInstances:
        sumOfEntropies += -x*math.log(x,2)
    return sumOfEntropies

	root = add_root()
	root = self.current_node

	def calculateEntropyOfList(list):  
		cnt = Counter(x for x in list)
		totalInstances = len(list)
		proportionOfInstances = [x / totalInstances for x in cnt.values()]
		sumOfEntropies=0
		for x in proportionOfInstances:
			sumOfEntropies += -x*math.log(x,2)
		return sumOfEntropies

	def calculateInformationGain(trainingData, attributeToSplitOn, targetAttribute):
		grpByAttrbToSplitOn = trainingData.groupby(attributeToSplitOn)   
	
		# Group data by attribute to split on and apply aggregate function to each group on targetAttribute column
		aggregatedData = grpByAttrbToSplitOn.agg({targetAttribute : [calculateEntropyOfList, lambda x: len(x)/(len(trainingData.index))] })[targetAttribute]   

		aggregatedData.columns = ['Entropy', 'ProportionOfInstances']
		
		newEntropy = sum( aggregatedData['Entropy'] * aggregatedData['ProportionOfInstances'] )
		oldEntropy = calculateEntropyOfList(trainingData[targetAttribute])
		return oldEntropy-newEntropy
	
	def treeByInformationGainHeuristic(trainingData,targetAttribute,allAttributesList):
    
		#calculate info gain for all columns in given data except 'Class'. Pick the one with max gain
		infoGain=[calculateInformationGain(trainingData,x,targetAttribute) for x in allAttributesList]
		indexOfMaxGain=infoGain.index(max(infoGain))
		bestAttrbToSplitOn=allAttributesList[indexOfMaxGain]
		#how to create a tree and store all nodes and parents ?????????
		node = add_node(parent=self.current_node, name=bestAttrbToSplitOn)
		node = self.current_node

		remainingAttrbList=[x for x in allAttributesList if x!=bestAttrbToSplitOn]

		for attributeIndex,dataSubset in trainingData.groupby(bestAttrbToSplitOn):

			# Recursive call with smaller dataset 
			subtree= treeByInformationGainHeuristic(dataSubset,targetAttribute,remainingAttrbList)
			
		return tree


tree = Tree()
print(tree.root.negChild)