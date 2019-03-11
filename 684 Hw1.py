import pandas as pd
import sys
from collections import Counter
import math
import copy
import random


# Input parameters
L = int(sys.argv[1])
K = int(sys.argv[2])
training_set_path = sys.argv[3]
validation_set_path = sys.argv[4]
test_set_path = sys.argv[5]
to_print = sys.argv[6]

# read in the training data
training_set = pd.read_csv(training_set_path ,delimiter=',')
training_set_column_values = list(training_set.columns.values)
target_attribute = training_set_column_values[-1]

validation_set = pd.read_csv(validation_set_path ,delimiter=',')
test_set = pd.read_csv(test_set_path ,delimiter=',')

class Node:

	def __init__(self, name=None, parent=None, value_direction=None, data_frame=None, node_depth=0, node_sequence=1):
		self.name = name
		self.parent = parent #
		self.children_list = []
		self.value_direction = value_direction
		# redundant, since can be verified if self.name==None. but a good practice to have for efficiency
		self.is_leaf = False
		# only assign value when is_leaf is True
		self.leaf_target = None
		# temp variable to store the dataframe to be splitted from this node, also used for post-pruning
		self.data_frame = data_frame
		# helper attribute when printing out the tree
		self.node_depth = node_depth

class Tree:

	def __init__(self):
		self.root = None
		self.feature_list = training_set_column_values[:-1]
		self.current_node = None
		# used for IG tree
		self.current_entropy = None
		# used for VI tree
		self.current_variance_impurity = None 
		# Keep track of the leaves and nodes to be splitted
		self.fringe = [] 
		# helper attributes for print_decision_tree
		self.max_print_depth = 0 
		self.tree_print_output = ''
		# keep track of the non-leaf nodes used for post-pruning
		self.non_leaf_nodes = []
	
	def build_tree(self, heuristic): 
		'''heuristic equals to either ig(information gain) or vi(variance impurity)'''
		# determine if the root exists
		if self.current_node == None:
			self.root = Node(value_direction="ALL")
			self.current_node = self.root
			if heuristic == "ig":
				# if building the tree with information gain heuristic
				self.root.name = self.choose_the_best_feature_by_information_gain(training_set, self.feature_list)
			else:
				# if building the tree with impurity gain heuristic
				self.root.name = self.choose_the_best_feature_by_variance_impurity(training_set, self.feature_list)
			# special case: when the whole data set is already pure
			if self.root.name == None:
				self.root.is_leaf = True
				self.root.leaf_target = training_set[target_attribute][0]
				return
			else:
				self.non_leaf_nodes.append(self.current_node)
				self.current_node.data_frame = training_set
				self.feature_list.remove(self.root.name)
				# put the node to the fringe ready to split
				self.fringe.append(self.current_node)
				# start to recursively build the tree
				self.build_tree(heuristic)
		else:
			'''Stop building the tree if all the features necessary to be splitted are all splitted (either a pure data set is found under the branch(the leaf node is found in this branch) or this branch has used up all the features in the list. This is tracked by self.fringe). In this situation the nodes in the fringe should be all leaves.'''
			if self.check_if_fringe_has_all_leaves() == True:
				return
			else:
				# a helpful new fringe list to avoid errors
				# when iterate through self.fringe
				current_fringe = copy.copy(self.fringe)
				# why for node in current_fringe, it always skips the first element
				for node in current_fringe:
					if node.is_leaf == False: 
						pass
					else:
						continue
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
						# construct the new node
						new_node = Node(name=node_name, parent=self.current_node, value_direction=data_frame[self.current_node.name].values[0], data_frame=data_frame, node_depth=len(self.feature_list) - len(current_feature_list) + 1)
						self.current_node.children_list.append(new_node)
						if new_node.name == None:
							# data set under the current value of
							# the splitted feature is pure, no more 
							# splitting new_node becomes a leaf node
							new_node.is_leaf = True
							if len(current_feature_list) == 0:
								# if all the features are used but data is still not pure, assigned to the value that appear the most in the class value
								most_target_values = data_frame[target_attribute].mode()
								# if there are more than one value that appears the most, we make a random selection
								new_node.leaf_target = most_target_values[random.randint(0,len(most_target_values)-1)]
							else:
								# data pure, value will be assigned to the class value
								new_node.leaf_target = data_frame[target_attribute].values[0]
							# still add to the fringe used to check if the tree building process has to be stopped
							self.fringe.append(new_node)
						else:
							self.non_leaf_nodes.append(new_node)
							# just add the node to the fringe
							self.fringe.append(new_node)
				self.build_tree(heuristic)	



	# Return the next attribute to split based on
	# the information gain heuristic calculation
	def choose_the_best_feature_by_information_gain(self, data_with_targets, current_feature_list):
		''' calculate the E(S) of the current (splitted) dataframe '''
		self.current_entropy = self.calculate_entropy(data_with_targets[target_attribute])
		''' data is pure, no need to split the current node'''
		if self.current_entropy == 0 or len(current_feature_list) == 0:
			return None
		''' get the max value tuple based on the first tuple
		    value(information gain) in a list of tuples, then get the feature name associated with this value '''
		return max([(self.information_gain(feature, data_with_targets), feature) for feature in current_feature_list], key = lambda feature_tuple: feature_tuple[0])[1]

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
		data_aggregate = data_split.agg({target_attribute : [self.calculate_entropy, lambda group: len(group)/len(data_with_targets)] })[target_attribute]
		data_aggregate.columns = ['Entropy', 'Ratios']
		'''Second - an weighted sum of the product of the entropy and the value of each value is the entropy of this feature'''
		entropy_of_the_feature = sum(data_aggregate['Entropy'] * data_aggregate['Ratios'])
		return (self.current_entropy - entropy_of_the_feature)

	def choose_the_best_feature_by_variance_impurity(self, data_with_targets, current_feature_list):
		''' calculate the VI(S) of the current (splitted) dataframe '''
		self.current_variance_impurity = self.calculate_variance_impurity(data_with_targets[target_attribute])
		''' data is pure, no need to split the current node'''
		if self.current_variance_impurity == 0 or len(current_feature_list) == 0:
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
		data_aggregate = data_split.agg({target_attribute : [self.calculate_variance_impurity, lambda group: len(group)/len(data_with_targets)] })[target_attribute]
		data_aggregate.columns = ['Variance Impurity', 'Ratios']
		variance_impurity_of_the_feature = sum(data_aggregate['Variance Impurity'] * data_aggregate['Ratios'])
		# old_entropy = self.calculate_entropy(data_with_targets[target_attribute])
		return (self.current_variance_impurity - variance_impurity_of_the_feature)

	# a helper function used to check if the nodes in the fringe
	# are all leaves. If all leaves, stop building the tree
	def check_if_fringe_has_all_leaves(self):
		for node in self.fringe:
			if node.is_leaf == False:
				return False
		return True

	''' Print the decision tree. When printing the whole decision tree using preorder, must pass in its root'''
	def print_decision_tree(self, heuristic):
		whether_print = 'y'
		tree_print_length = len(self.non_leaf_nodes)
		if tree_print_length > 300:
			whether_print = input("This tree may be extremely long.\nIt has {0} nodes which means {1} lines of output.\nPlease decide whether continue to print.(y/n)".format(tree_print_length, tree_print_length * 2))
			while whether_print != 'y' and whether_print != 'n':
				whether_print = input("Please input y/n:")
		if whether_print == 'y':
			self.construct_decision_tree_output(self.root)
			for line in self.tree_print_output.splitlines():
				if len(line[-2:]) > 0 and line[-2:][0] == '=':
					continue
				else:
					print(line)
		else:
			pass
		

	''' core function for print_decision_tree.  print_decision_tree is needed to strip out the redundant leaf lines due to the difficulties of removing them in construct_decision_tree_output()'''
	def construct_decision_tree_output(self, node, is_right_most_leaf=False):
		self.max_print_depth = max(self.max_print_depth, node.node_depth)
		# start from the root using pre-order 
		# traversal to print out the decision tree
		if node.is_leaf == True:
			self.tree_print_output += str(node.leaf_target)
			if is_right_most_leaf == False:
				self.tree_print_output += ("\n{0}{1} = ".format((node.node_depth-1) * '| ', node.parent.name))
			else:
				pass
			return
		else:
			self.tree_print_output += ("\n{0}{1} = ".format(node.node_depth * '| ', node.name))
			for child_iter in range(len(node.children_list)):
				child = node.children_list[child_iter]
				if child_iter == len(node.children_list) - 1:
					if child.node_depth < self.max_print_depth:
						self.tree_print_output += ("\n{0}{1} = ".format(node.node_depth * '| ', node.name))
					self.tree_print_output += ("{0} : ".format(child.value_direction))
					self.construct_decision_tree_output(child, is_right_most_leaf=True)
				else:	
					self.tree_print_output += ("{0} : ".format(child.value_direction))
					self.construct_decision_tree_output(child, is_right_most_leaf=False)

	def evaluate_accuracy(self, data_set):
		tmpNode = self.root
		num_of_instances = len(data_set)
		correctly_classified = 0
		for index, row in data_set.iterrows():
			missing_value = False
			while tmpNode.is_leaf != True:
				search_attr_name = tmpNode.name
				search_direction = row[search_attr_name]
				for node_iter in range(len(tmpNode.children_list)):
					node = tmpNode.children_list[node_iter]
					if node.value_direction == search_direction:
						tmpNode = node
						break
					else:
						if node_iter == len(tmpNode.children_list) - 1:
							# we have missing values in the traning_set
							# we choose to ignore it
							missing_value = True
							break
				if missing_value == True:
					break
			if missing_value == False:
				prediction = tmpNode.leaf_target
				if prediction == row[target_attribute]:
					correctly_classified += 1
			else:
				pass
			tmpNode = self.root
		return correctly_classified/num_of_instances

# helper function for post_pruning_decision_tree to prune children and sub-children nodes under one node 
def prune_all_non_leaf_nodes_under_one_node(tree, node):
	for child in node.children_list:
		if child.is_leaf != True:
			tree.non_leaf_nodes.remove(child)
			prune_all_non_leaf_nodes_under_one_node(tree, child)

def post_pruning_decision_tree(decision_tree):
	the_best_tree = copy.deepcopy(decision_tree)
	the_best_tree_accuracy = the_best_tree.evaluate_accuracy(validation_set)
	for i in range(L):
		tmpTree = copy.deepcopy(decision_tree)
		M = random.randint(1, K)
		for j in range(M):
			N = len(tmpTree.non_leaf_nodes)
			if N == 0:
				# the root of this tree is a leaf node
				# no need to prune
				break
			else:
				P = random.randint(1, N)
				tmp_leaf_node = tmpTree.non_leaf_nodes[P - 1]
				# make this node leaf node
				tmp_leaf_node.is_leaf = True
				# pruning all its children and sub-children nodes
				prune_all_non_leaf_nodes_under_one_node(tmpTree, tmp_leaf_node)
				#empty the children list whatsoever
				tmp_leaf_node.children_list.clear()
				# remove itself from the non_leaf_nodes
				tmpTree.non_leaf_nodes.remove(tmp_leaf_node)
				# assign the target_value to the majority of its class
				most_target_values = tmp_leaf_node.data_frame[target_attribute].mode()
				# if there are more than one value that appears the most, we make a random selection
				tmp_leaf_node.leaf_target = most_target_values[random.randint(0,len(most_target_values)-1)]
		# evaluate the accuracy of this tree
		tmpTree_accuracy = tmpTree.evaluate_accuracy(validation_set)
		if tmpTree_accuracy > the_best_tree_accuracy:
			the_best_tree = copy.deepcopy(tmpTree)
	return the_best_tree


''' Start the program '''

''' Information Gain'''
print("Information Gain")
print("Building the tree using the training data by information gain...")
tree_ig = Tree()
tree_ig.build_tree("ig")
print()
print("Before pruning, tree accuracy on test data set:", tree_ig.evaluate_accuracy(test_set))
print("Pruning the tree...")
tree_ig_best = post_pruning_decision_tree(tree_ig)
print("After pruning, tree accuracy on test set:", tree_ig_best.evaluate_accuracy(test_set))
print()
# print out the tree
if to_print == "yes":
	print("Printing out the original tree")
	tree_ig.print_decision_tree("ig")
	print()
	print("Printing out the post pruned tree")
	tree_ig_best.print_decision_tree("ig")
	print()

''' Variance Impurity '''
print("Variance Impurity")
print("Building the tree using the training data by variance impurity...")
tree_vi = Tree()
tree_vi.build_tree("vi")
print()
print("Before pruning, tree accuracy on test data set:", tree_vi.evaluate_accuracy(test_set))
print("Pruning the tree...")
tree_vi_best = post_pruning_decision_tree(tree_vi)
print("After pruning, tree accuracy on test set:", tree_vi_best.evaluate_accuracy(test_set))
print()
# print out the tree
if to_print == "yes":
	print("Printing out the original tree")
	tree_vi.print_decision_tree("vi")
	print()
	print("Printing out the post pruned tree")
	tree_vi_best.print_decision_tree("vi")