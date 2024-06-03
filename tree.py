from helper_functions import fit_sigmoid_params, discrete_features

class TreeNode:
    def __init__(self, name, feature=None, left=None, right=None, class_label=None):
        self.name = name
        self.feature = feature
        self.left = left
        self.right = right
        self.class_label = class_label
        self.probability = None # Used by discrete features
        self.sigmoid_params = {'a': 1, 'b': 0}  # Default sigmoid parameters (a, b), used by continuous features

    def calculate_conditional_probabilities(self, data):
        if self.class_label is None and self.feature: # Only calculate if this is not a leaf node and the feature is discrete
            # Discrete feature
            if self.feature in discrete_features: 
                print(f"Calculating probabilities for discrete node: {self.name}")
                feature_values = set([item[self.feature] for item in data])
                self.probability = {}
                for value in feature_values:
                    filtered_data = [item for item in data if item[self.feature] == value]
                    total_count = len(filtered_data)
                    if total_count > 0:
                        left_count = len([item for item in filtered_data if item['class_label'] in self.get_class_labels(self.left)])
                        right_count = len([item for item in filtered_data if item['class_label'] in self.get_class_labels(self.right)])
                        self.probability[value] = {
                            'left': left_count / total_count,
                            'right': right_count / total_count
                        }

            # Continuous feature         
            else: 
                print(f"Calculating sigmoid parameters for continuous node: {self.name}")
                fit_sigmoid_params(self, data)

    # Function that returns all class labels in the tree located beneath current node
    def get_class_labels(self, node):
        if node.class_label is not None:
            return [node.class_label]
        else: # Recursively fetch class beneath node
            return self.get_class_labels(node.left) + self.get_class_labels(node.right)
        
def create_tree():
    # Define the classification tree based on the paper's description
    root = TreeNode('root', feature='mcg')

    # First level
    root.left = TreeNode('chg_or_gvh', feature='lip')
    root.right = TreeNode('im_or_cp', feature='alm1')

    # Second level for 'chg_or_gvh' branch
    root.left.left = TreeNode('imL_or_omL', feature='chg')
    root.left.right = TreeNode('imU_or_alm2', feature='gvh')

    # Second level for im_or_cp branch
    root.right.left = TreeNode('im', class_label=1)
    root.right.right = TreeNode('cp', class_label=0)

    # Third level for the chg branch
    root.left.left.left = TreeNode('imL', class_label=6)
    root.left.left.right = TreeNode('omL', class_label=5)

    # Third level for the gvh branch
    root.left.right.left = TreeNode('imU', class_label=3)
    root.left.right.right = TreeNode('aac_or_imS', feature='alm2')

    # Fourth level for 'aac_or_imS' branch
    root.left.right.right.left = TreeNode('om_or_pp', feature='aac')
    root.left.right.right.right = TreeNode('imS', class_label=7)

    # Fifth level for 'om_or_pp' branch
    root.left.right.right.left.left = TreeNode('om', class_label=4)
    root.left.right.right.left.right = TreeNode('pp', class_label=2)

    return root