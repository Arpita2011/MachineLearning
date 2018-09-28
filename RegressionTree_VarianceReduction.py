import numpy as np
import pandas as pd
from pprint import pprint
df_C= pd.read_csv("C:\\Users\\Arpita\\Desktop\\Book\\Fall 2018\\Machine Learning\\Assignment\Assignment 2\\compustat_annual_2000_2017_with link information.csv")

data = df_C.loc[:,['sale','cogs','xsga','ebit']]
#Y = df_C.loc[:,'ebit']
features = ['sale','cogs','xsga']
mean_data = np.mean(data.iloc[:,-1])

data.shape
data=data.loc[data['ebit'].isna()==False,:]
data.shape

data=data.dropna()

training_data = data.iloc[0:150]
testing_data = data.iloc[152:192]

def var(data,split_attribute_name,target_name="ebit"):
    
    feature_values = np.unique(data[split_attribute_name])
    feature_variance = 0
    for value in feature_values:
        #Create the data subsets --> Split the original data along the values of the split_attribute_name feature
        # and reset the index to not run into an error while using the df.loc[] operation below
        subset = data.query('{0}=={1}'.format(split_attribute_name,value)).reset_index()
        #Calculate the weighted variance of each subset            
        value_var = (len(subset)/len(data))*np.var(subset[target_name],ddof=1)
        #Calculate the weighted variance of the feature
        feature_variance+=value_var
    return feature_variance
    
def Build_tree(data,originaldata,features,target_attribute_name,parent_node_class = None):
    #print(features)
  
    #If the dataset is empty, return the mean target feature value in the original dataset
    if len(data)==0:
        return np.mean(originaldata[target_attribute_name])
    
 
    elif len(features) ==0:
        return parent_node_class    
   
    else:
        #Set the default value for this node --> The mean target feature value of the current node
        parent_node_class = np.mean(data[target_attribute_name])
        #Select the feature which best splits the dataset
        item_values = [var(data,feature) for feature in features] #Return the variance for features in the dataset
        best_feature_index = np.argmin(item_values)
        best_feature = features[best_feature_index]
        
        #Create the tree structure. The root gets the name of the feature (best_feature) with the minimum variance.
        tree = {best_feature:{}}
        
        
        #Remove the feature with the lowest variance from the feature space
        features = [i for i in features if i != best_feature]
        
        #Grow a branch under the root node for each possible value of the root node feature
        
        for value in np.unique(data[best_feature]):
            value = value
            #Split the dataset along the value of the feature with the lowest variance and therewith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()
            #print(len(sub_data))
            #Call the Calssification algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = Build_tree(sub_data,originaldata,features,'ebit',parent_node_class = parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
            
        return tree   
    
def predict(query,tree,default = mean_data):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
        
def test(data,tree):
    #Create new query instances by simply removing the target feature column from the original dataset and 
    #convert it to a dictionary
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = []
    #Calculate the RMSE
    for i in range(len(data)):
        predicted.append(predict(queries[i],tree,mean_data)) 
    RMSE = np.sqrt(mean_squared_error(testing_data["ebit"], predicted))
    return RMSE

tree = Build_tree(training_data,training_data,training_data.columns[:-1],'ebit')

pprint(tree)
print('#'*50)
print('Root mean square error (RMSE): ',test(testing_data,tree))