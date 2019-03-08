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
		self.value_direction = None
		self.is_leaf = False
		self.leaf_target = None # only assign value when is_leaf is True

class Tree:

	def __init__(self):
		self.root = None
		self.data = training_data
		self.target_attribute = training_data_column_values[-1]
		self.feature_list = training_data_column_values.remove(self.target_attribute)
		self.current_node = None
		self.current_entropy = None
		self.fringe = [] # Keep track of the nodes to be splitted

		# call build_tree to start building the tree
		self.build_tree()
	
	def build_tree(self):
		''' stop building the tree if either all the features 
			are splitted or the data set doesn't contain any feature'''
		if len(self.feature_list) == 0:
			return
		else:
			# determine if the root exists
			if self.current_node == None:
				self.root = Node()
				self.current_node = self.root
				self.root.name = self.choose_the_best_feature(self.data)
				# special case: when the whole data set is already pure
				if self.root.name == None:
					self.root.is_leaf = True
					self.root.leaf_target = self.data[self.target_attribute][0]
					return
				else:
					# start to recursively build the tree
					self.feature_list.remove(self.root.name)
					self.build_tree()
			else:
				# split the current feature by values and
				# create independent dataframes
				data_split = self.data.groupby(self.current_node.name)
				splitted_data_frames = [data_split.get_group(value) for value in data_split.groups]
				for data_frame in splitted_data_frames:
					new_node = Node(name=self.choose_the_best_feature(data_frame), parent=self.current_node)
					if new_node.name == None:
						# data set under the current value of
						# the splitted feature is pure, no more splitting
						# new_node becomes a leaf node
						new_node.is_leaf = True
						new_node.leaf_target = data_frame[self.target_attribute][0]
					else:
						
					



	# Return the next attribute to split based on
	# the information gain calculation
	def choose_the_best_feature(self, data_with_targets):
		''' calculate the E(S) of the current (splitted) dataframe '''
		self.current_entropy = self.calculate_entropy(data_with_targets[self.target_attribute])
		''' data is pure, no need to split the current node'''
		if self.current_entropy == 0:
			return None
		''' get the max value tuple based on the first tuple
		    value(information gain) in a list of tuples, then get
		    the feature name associated with this value '''
		return max([(information_gain(feature), feature) for feature in self.feature_list], key = lambda feature: feature[0])[1]

	def calculate_entropy(target_list):
		# count the target values by value
		cnt = Counter(target for target in target_list)
		num_of_instances = len(target_list)
		ratios = [value / num_of_instances for value in cnt.values()]
		# calculate entropy
		entropy = 0
		for ratio in ratios:
			entropy += -ratio * math.log(ratio, 2)
		return entropy

	# def calculate_variance_impurity(self):
	# 	pass

	def information_gain(self, feature_to_split):
		# split the data into groups of values by an attribute
		data_split = self.data.groupby(feature_to_split)
		'''calculate the entropy of this atrribute
		   First - use calculate_entropy() for each group of data to
		   get its entropy, then an anonymous lambda function is 
		   used to get the ratio for each group.
		   data_aggregate is the dataFrame storing these values.
		   ''' 
		data_aggregate = data_split.agg({self.target_attribute : [self.calculate_entropy, lambda group: len(group)/len(self.data)] })[self.target_attribute]
		data_aggregate.columns = ['Entropy', 'Ratios']
		'''Second - an weighted sum of the product of the entropy 
		   and the value of each value is the entropy of this feature'''
		entropy_of_the_feature = sum(data_aggregate['Entropy'] * data_aggregate['Ratios'])
		# old_entropy = self.calculate_entropy(self.data[self.target_attribute])
		return (self.current_entropy - entropy_of_the_feature)

	def print_decision_tree():
		# start from the root using pre-order traversal
		# to print out the decision tree
		pass