# methods that will allow you to interpret a model quantitatively.

# Visualized a decision tree.
# Interpreted the decisions of a decision tree.
# Learned the difference between local and global interpretation.
import pandas as pd 
df_train = pd.read_csv('attrition_train.csv')
df_train.head()

# # Visualizing a decision tree
# the prediction quality is not the only important point. 
# It's often even more important to understand how the predictions are made. We talk about a machine learning model's **interpretability**. 
# If a model is easy to interpret, this has many advantages.
    # Errors that are difficult to detect are then also easier to find (e.g. if the data is biased).
    # Findings from the data are easier to generate. These in turn make it easier to recommend actions to be taken.
    # It's possible to justify specific decisions of the model.
    # A model can be more widely accepted.

target_train = df_train.iloc[:,0]
features_train = df_train.iloc[:,1:]

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(class_weight='balanced', max_depth=3, random_state=0)
model.fit(features_train, target_train)

# export_graphviz from sklearn.tree
# The return value of this function is a string which represents the tree in Graphviz's Dot format.

# export_graphviz(decision_tree=object,           # a fitted decision tree classifier                
#                 out_file=object,                # handle or name of the output file
#                 feature_names=[`list` of `str`],# names of the features
#                 class_names=[`list` of `str`],  # names of the target variable
#                 filled=bool,                   # colors the nodes
#                 impurity=bool                   # displays the criterion value
#                )   

from sklearn.tree import export_graphviz

tree_string = export_graphviz(decision_tree=model,  # a fitted decision tree classifier
                feature_names=df_train.columns[1:], # names of the features
                class_names='attrition',            # names of the target variable
                filled=True,                        # colors the nodes
                impurity=True                       # displays the criterion value
               )


from pydotplus import graph_from_dot_data
graph = graph_from_dot_data(tree_string)

graph.write_png('decision_tree.png')

from IPython.display import Image  
Image('decision_tree.png')



# With less leafs

model  = DecisionTreeClassifier(class_weight='balanced', max_depth=3, random_state=0, min_samples_leaf=20)
model.fit(features_train, target_train)

tree_string = export_graphviz(decision_tree=model,  # a fitted decision tree classifier
                feature_names=df_train.columns[1:], # names of the features
                class_names='attrition',            # names of the target variable
                filled=True,                        # colors the nodes
                impurity=True                       # displays the criterion value
               )

graph = graph_from_dot_data(tree_string)

graph.write_png('decision_tree.png')

Image('decision_tree.png')





# Remember:

# To visualize a decision tree with sklearn, you need the following functions:
# export_graphviz from sklearn.tree to convert the decision tree to Graphviz dot format and return it as str.
# graph_from_dot_data from pydotplus to create an image file.
# With global interpretation you want to understand how important individual features are for the model's prediction.
# With local interpretation, you want to understand why certain data points received the prediction they got.