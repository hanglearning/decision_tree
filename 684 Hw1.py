import pandas as pd
import sys
from collections import Counter
import math
import copy

# Input parameters
# L = int(sys.argv[1])
# K = int(sys.argv[2])
# training_set_path = sys.argv[3]
# validation_set_path = sys.argv[4]
# test_set_path = sys.argv[5]
# to_print = sys.argv[6]

# read in the training data
training_data = pd.read_csv('/Users/chenhang91/Downloads/data_sets1/training_set.csv',delimiter=',')
training_data_column_values = list(training_data.columns.values)

class Node:

	def __init__(self, name=None, parent=None, value_direction=None, data_frame=None):
		self.name = name
		self.parent = parent #
		self.children_list = []
		self.value_direction = value_direction
		self.is_leaf = False # redundant, since can be verified if self.name==None. but a good practice to have for efficiency
		self.leaf_target = None # only assign value when is_leaf is True
		self.data_frame = data_frame # temp variable to store the dataframe to be splitted from this node, should be cleared to save memory after the tree is built

class Tree:

	def __init__(self):
		self.root = None
		self.target_attribute = training_data_column_values[-1]
		self.feature_list = training_data_column_values[:-1]
		self.current_node = None
		self.current_entropy = None # used for IG tree
		self.current_variance_impurity = None # used for VI tree
		self.fringe = [] # Keep track of the nodes to be splitted
	
	def build_tree(self, heuristic): 
		'''heuristic equals to either ig(information gain) or vi(variance impurity)'''
		# determine if the root exists
		if self.current_node == None:
			self.root = Node(value_direction="ALL")
			self.current_node = self.root
			if heuristic == "ig":
				# if building the tree with information gain heuristic
				self.root.name = self.choose_the_best_feature_by_information_gain(training_data, self.feature_list)
			else:
				# if building the tree with impurity gain heuristic
				self.root.name = self.choose_the_best_feature_by_variance_impurity(training_data, self.feature_list)
			# special case: when the whole data set is already pure
			if self.root.name == None:
				self.root.is_leaf = True
				self.root.leaf_target = training_data[self.target_attribute][0]
				return
			else:
				self.current_node.data_frame = training_data
				self.feature_list.remove(self.root.name)
				# put the node to the fringe ready to split
				self.fringe.append(self.current_node)
				# start to recursively build the tree
				self.build_tree(heuristic)
		else:
			''' stop building the tree if all the features necessary to be splitted are all splitted (either a pure data set is found under the branch(the leaf node is found in this branch) or this branch has used up all the features in the list. This is tracked by self.fringe)'''
			if len(self.fringe) == 0:
				return
			else:
				# a helpful new fringe list to avoid errors
				# when iterate through self.fringe
				current_fringe = copy.copy(self.fringe)
				# why for node in current_fringe, it always skips the first element
				for node_iterator in range(len(current_fringe)): 
					node = current_fringe[node_iterator]
					self.current_node = node
					# update the current feature list for this branch
					tmpNode = node
					current_feature_list = copy.deepcopy(self.feature_list)
					while tmpNode.parent != None:
						current_feature_list.remove(tmpNode.name)
						tmpNode = tmpNode.parent
					# split the current feature by values and
					# create independent dataframes
					data_split = self.current_node.data_frame.groupby(self.current_node.name)				
					splitted_data_frames = [data_split.get_group(value) for value in data_split.groups]
					# node has been splitted. removed from the fringe
					self.fringe.remove(self.current_node)
					for data_frame in splitted_data_frames:
						if heuristic == "ig":
							node_name = self.choose_the_best_feature_by_information_gain(data_frame, current_feature_list)
						else:
							node_name = self.choose_the_best_feature_by_variance_impurity(data_frame, current_feature_list)
						new_node = Node(name=node_name, parent=self.current_node, value_direction=data_frame[self.current_node.name].values[0], data_frame=data_frame)
						self.current_node.children_list.append(new_node)
						if new_node.name == None:
							# data set under the current value of
							# the splitted feature is pure, no more 
							# splitting new_node becomes a leaf node
							new_node.is_leaf = True
							new_node.leaf_target = data_frame[self.target_attribute].values[0]
						else:
							# add the node to the fringe
							self.fringe.append(new_node)
				self.build_tree(heuristic)	



	# Return the next attribute to split based on
	# the information gain heuristic calculation
	def choose_the_best_feature_by_information_gain(self, data_with_targets, current_feature_list):
		''' calculate the E(S) of the current (splitted) dataframe '''
		self.current_entropy = self.calculate_entropy(data_with_targets[self.target_attribute])
		''' data is pure, no need to split the current node'''
		if self.current_entropy == 0:
			return None
		''' get the max value tuple based on the first tuple
		    value(information gain) in a list of tuples, then get the feature name associated with this value '''
		return max([(self.information_gain(feature, data_with_targets), feature) for feature in current_feature_list], key = lambda feature: feature[0])[1]

	def calculate_entropy(self, target_list):
		# count the target values by value
		cnt = Counter(target for target in target_list)
		num_of_instances = len(target_list)
		ratios = [value / num_of_instances for value in cnt.values()]
		# calculate entropy
		entropy = 0
		for ratio in ratios:
			entropy += -ratio * math.log(ratio, 2)
		return entropy

	def information_gain(self, feature_to_split, data_with_targets):
		# split the data into groups of values by an attribute
		data_split = data_with_targets.groupby(feature_to_split)
		'''calculate the entropy of this atrribute
		   First - use calculate_entropy() for each group of data to get its entropy, then an anonymous lambda function is used to get the ratio for each group.
		   data_aggregate is the dataFrame storing these values.
		   ''' 
		data_aggregate = data_split.agg({self.target_attribute : [self.calculate_entropy, lambda group: len(group)/len(data_with_targets)] })[self.target_attribute]
		data_aggregate.columns = ['Entropy', 'Ratios']
		'''Second - an weighted sum of the product of the entropy and the value of each value is the entropy of this feature'''
		entropy_of_the_feature = sum(data_aggregate['Entropy'] * data_aggregate['Ratios'])
		# old_entropy = self.calculate_entropy(data_with_targets[self.target_attribute])
		return (self.current_entropy - entropy_of_the_feature)

	def choose_the_best_feature_by_variance_impurity(self, data_with_targets, current_feature_list):
		''' calculate the VI(S) of the current (splitted) dataframe '''
		self.current_variance_impurity = self.calculate_variance_impurity(data_with_targets[self.target_attribute])
		''' data is pure, no need to split the current node'''
		if self.current_variance_impurity == 0:
			return None
		''' get the max value tuple based on the first tuple
		    value(impurity gain) in a list of tuples, then get the feature name associated with this value '''
		return max([(self.impurity_gain(feature, data_with_targets), feature) for feature in current_feature_list], key = lambda feature: feature[0])[1]

	def calculate_variance_impurity(self, target_list):
		# count the target values by value
		cnt = Counter(target for target in target_list)
		num_of_instances = len(target_list)
		ratios = [value / num_of_instances for value in cnt.values()]
		# calculate variance impurity
		variance_impurity = 1
		for ratio in ratios:
			variance_impurity *= ratio
		return variance_impurity

	def impurity_gain(self, feature_to_split, data_with_targets):
		# split the data into groups of values by an attribute
		data_split = data_with_targets.groupby(feature_to_split)
		'''calculate the variance impurity of this atrribute. The process is similar to calculate its information gain''' 
		data_aggregate = data_split.agg({self.target_attribute : [self.calculate_variance_impurity, lambda group: len(group)/len(data_with_targets)] })[self.target_attribute]
		data_aggregate.columns = ['Variance Impurity', 'Ratios']
		variance_impurity_of_the_feature = sum(data_aggregate['Variance Impurity'] * data_aggregate['Ratios'])
		# old_entropy = self.calculate_entropy(data_with_targets[self.target_attribute])
		return (self.current_variance_impurity - variance_impurity_of_the_feature)


	''' Print the decision tree. When printing the whole decision tree using preorder, must pass in its root'''
	def print_decision_tree(self, node, current_depth=0):
		# start from the root using pre-order 
		# traversal to print out the decision tree
		if node.name==None:
			print('',end='')
		if node.is_leaf == True:
			print(node.leaf_target)
			print("{0}{1} = ".format((current_depth-1) * '| ', node.parent.name), end='')
			return
		else:
			print("\n{0}{1} = ".format(current_depth * '| ', node.name), end='')
			for child in node.children_list:
				print("{0} : ".format(child.value_direction), end='')
				self.print_decision_tree(child, current_depth+1)


# tree_ig = Tree()
# tree_ig.build_tree("ig")
# tree_ig.print_decision_tree(tree_ig.root)

tree_vi = Tree()
tree_vi.build_tree("vi")
tree_vi.print_decision_tree(tree_vi.root)