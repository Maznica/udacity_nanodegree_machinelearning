#!/usr/bin/python

import sys
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
import operator
import math
from itertools import combinations, product

sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options','bonus'] 


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# It is a dictionary where the key is the name of the person and
# the value is a dictionary with the features

###############################
# How many people are in dataset, how many and what features do we have? 

n_people = len(data_dict)
n_features = len(data_dict.itervalues().next())
features_names = data_dict.itervalues().next().keys()

print "\nThere are " + str(n_people) + " in our dataset, for each of which" 
print "we have " + str(n_features) + " features."
print "The list of features is as follows:\n", features_names

'''
There are 146 people in our dataset
There are 21 features available for each person.
The features are as follows:
['salary', 'to_messages', 'deferral_payments', 'total_payments', 
'exercised_stock_options', 'bonus', 'restricted_stock', 
'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 
'expenses', 'loan_advances', 'from_messages', 'other', 
'from_this_person_to_poi', 'poi', 'director_fees', 
'deferred_income', 'long_term_incentive', 'email_address', 
'from_poi_to_this_person']
'''

######################
# How many pois in the dataset

def n_pois(data):
    result=0
    listing=[]
    for item in data.keys():
        if data[item]['poi']==True:
            result+=1
            listing.append(item)
    return result, listing
    
num_pois=n_pois(data_dict)
num_nonpois = n_people-num_pois[0]

print "\nThere are " + str(num_pois[0]) + " persons of interest in our data."
print "The complete list can be seen below:\n",num_pois[1]

"""  
There are 18 persons of interest in our data.
The complete list can be found below:

['HANNON KEVIN P', 'COLWELL WESLEY', 'RIEKER PAULA H', 'KOPPER MICHAEL J',
  'SHELBY REX', 'DELAINEY DAVID W', 'LAY KENNETH L', 'BOWEN JR RAYMOND M',
  'BELDEN TIMOTHY N', 'FASTOW ANDREW S', 'CALGER CHRISTOPHER F',
  'RICE KENNETH D', 'SKILLING JEFFREY K', 'YEAGER F SCOTT', 'HIRKO JOSEPH',
  'KOENIG MARK E', 'CAUSEY RICHARD A', 'GLISAN JR BEN F']
"""

poi_file = "poi_names.txt"

list_pois=[]
with open(poi_file, 'rb') as f:
    reader = csv.reader(f)
    next(f)
    next(f )
    for row in reader:
        list_pois=list_pois+[row]
        


pois_total=len(list_pois)

print "\nIn total there are " + str(pois_total) + " persons of interest."
print "This means, that 17 pois are missing in our dataset"

"""
According to the poi_names.txt there are 35 pois. 
So, some of theme are missing in our data. This might lead
to several problems by our classification
1. we might have a selection bias - this means that the persons we have 
info for might differ from these we don't have information for.
2. The information for the training data might be to little. Only 
18 data points doesn't give us that many examples to learn from
"""

########################
# Missing data

def feature_missing(data):
    result={}
    for item in data.keys():
        for feature in data[item].keys():
            if data[item][feature]=='NaN':
                if feature not in result.keys():
                    result[feature]=1
                else:
                    result[feature] +=1
    for key in result.keys():
        result[str(key)+"_mis_share"]=round(result[key]/float(n_people),4)
    return result

feature_missing(data_dict)

"""
There is a lot of missing values. For example, there is no bonus information
by 43,84% people in our data.  
Moreover, we have email information only to 86 people. I.e. for the 60 people
(i.e. 41.1 %) the email data is missing.
The following features are features with the highest amount of missing data:
director fees (88.36%)
loan advances (97.26%)
restricted stock deferred (87.67%)
"""          

###########
# Missing data by pois and non-pois:

def feature_missing_by_poi(data):
    result={'poi':{},'nopoi':{}}
    for item in data.keys():
        if data[item]['poi']==True:
            for feature in data[item].keys():
                if data[item][feature]=='NaN':
                    if feature not in result['poi'].keys():
                        result['poi'][feature]=1
                    else:
                        result['poi'][feature] +=1     
        else:
            for feature in data[item].keys():
                if data[item][feature]=='NaN':
                    if feature not in result['nopoi'].keys():
                        result['nopoi'][feature]=1
                    else:
                        result['nopoi'][feature] +=1 
    fin_result={'poi':{},'nopoi':{}}
    for key in result['poi'].keys():
        fin_result['poi'][str(key)+"_mis_share"]=\
        round(result['poi'][key]/float(num_pois[0]),4)
    for key in result['nopoi'].keys():
        fin_result['nopoi'][str(key)+"_mis_share"]=\
        round(result['nopoi'][key]/float(num_nonpois),4)
    return fin_result


feature_missing_by_poi(data_dict)

"""
There are also big differences between pois and non-pois with respect to 
missing values which might influence the classification:
Bonus: pois: 11.11% mising vs. 48.44% by non-pois
Salary: pois: 5.56%, non-pois: 39.06%
By email data the difference is as follows:
pois: 22.22% and non-pois:43.75%

""" 

########################
# Missing values by person:

def per_feature_missing(data):
    result={}
    for item in data.keys():
        for feature in data[item].keys():
            if data[item][feature]=='NaN':
                if item not in result.keys():
                    result[item]=1
                else:
                    result[item] +=1
    for key in result.keys():
        result[key]=round(result[key]/float(n_features),4)
    return result

dict_per_missing = per_feature_missing(data_dict)

missing_outliers={}
for item in dict_per_missing:
    if dict_per_missing[item]>0.8:
        missing_outliers[item]=dict_per_missing[item]

missing_outliers  

### Task 2: Remove outliers

"""
We observed that by 11 observations more than 80% of the features are missing.
One of these observation is THE TRAVEL AGENCY IN THE PARK - which
is no person and therefore no poi.
We exclude it.
"""

data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

# In order to find outliers I will use visualisation

features_newlist=['salary', 'to_messages', 'deferral_payments', 
'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 
'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 
'expenses', 'loan_advances', 'from_messages', 'other', 
'from_this_person_to_poi', 'poi', 'director_fees', 
'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

data_outlier = featureFormat(data_dict, features_newlist)

for point in data_outlier:
    salary = point[features_newlist.index('salary')]
    bonus = point[features_newlist.index('bonus')]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.title("Bonus and Salary with Outliers")
plt.savefig("data_w_outliers.png")
plt.show()



# There is one very strong outlier in the dataset

for item in data_dict.keys():
    if data_dict[item]['salary']>10000000 and data_dict[item]['salary']!='NaN':
        print "Outlier:", item

# Total is the outlier - so let's remove it
data_dict.pop('TOTAL',0)

# Repeat the procedure once more
data_outlier_1 = featureFormat(data_dict, features_newlist)

# What are the additional outliers
for point in data_outlier_1:
    salary = point[features_newlist.index('salary')]
    bonus = point[features_newlist.index('bonus')]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.title("Bonus and Salary after removing values for TOTAL (outlier)")
plt.savefig("data_after_remove_out.png")
plt.show()

# There seems to be 4 additional outliers - but these are values we 
# definitely wants to leave in.

pos_outliers=[]
for item in data_dict.keys():
    if (data_dict[item]['salary']>1000000 and data_dict[item]['salary']!='NaN') \
    or (data_dict[item]['bonus']>=6000000 and data_dict[item]['bonus']!='NaN'):
        pos_outliers += [item]

pos_outliers

# ['LAVORATO JOHN J', 'LAY KENNETH L', 'SKILLING JEFFREY K', 'FREVERT MARK A']

# We can investigate the outlier issue also programmatically (by all features)

# Create dictionary of features with the list of values  
def distribution_features(data):
    distribution={}
    for item in data.keys():
        for key in data[item]:
            if key!='email_address' and key!='poi':
                if data[item][key]!='NaN':
                    if key not in distribution.keys():
                        distribution[key]=[data[item][key]]
                    else:
                        distribution[key]+= [data[item][key]]
    return distribution


# Boundaries for determining outliers
def boundaries(dictionary):
    percentile_result={}
    for item in dictionary.keys():
        percentile_result[item]={}
        perc_25 = np.percentile(dictionary[item],25)
        perc_75 = np.percentile(dictionary[item],75)
        lower = perc_25 - 1.5*(perc_75-perc_25)
        upper = perc_75 + 1.5*(perc_75-perc_25)
        percentile_result[item]['lower']=lower
        percentile_result[item]['upper']=upper
    return percentile_result

# Create a dictionary with name as a key and number of how many time outliers as value 
def outlier_dection(data):
    result={}
    dict_dist=distribution_features(data)
    bound=boundaries(dict_dist)
    for item in data.keys():
        for key in data[item]:
            if key!='email_address' and key != 'poi':
                if (data[item][key]!='NaN' and data[item][key]<=bound[key]['lower'])\
                or (data[item][key]!='NaN' and data[item][key]>=bound[key]['upper']):
                    if item not in result.keys():
                        result[item]=1
                    else:
                        result[item] +=1
    return result

outlier_dection(data_dict)

"""

Following people have the highest numbers of outliers:
Belden Timothy - poi
Frevert Mark - no-poi
Lavorato John - no-poi
Lay Kenneth - poi
Skilling Jeffrey - poi
Whalley Lawrence - no-poi

# http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
"""


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


for point in data_outlier_1:
    emails_to_poi = point[features_newlist.index('from_this_person_to_poi')]
    emails_from_poi = point[features_newlist.index('from_poi_to_this_person')]
    plt.scatter( emails_from_poi, emails_to_poi)
    if point[features_newlist.index('poi')]==1:
        plt.scatter(emails_from_poi,emails_to_poi,color='r')

plt.xlabel("Number of emails from POI")
plt.ylabel("Number of emails to POI")
plt.title("Relationship between number of emails from & to POI and being POI")
plt.savefig("from_to_poi_and_poi.png")
plt.show()

# The absolute numbers doesn't seem to be good indicator
# https://github.com/DariaAlekseeva/Enron_Dataset/blob/master/poi_id.py

## Creation of the fraction for emails from and to pois


def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if poi_messages=="NaN" or all_messages=="NaN":
        fraction=0.
    else:
        fraction = float(poi_messages)/float(all_messages)
    return fraction


# Store it in my data

for name in data_dict:
    data_point = data_dict[name]
    # fraction from poi
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi
    # fraction from person to poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi
    
# Examine the relationship between fractions and being POI

# Examining the relationship between emails from and to poi and the poi variable
# Create the feature list for the feature creatioön and selection

features_select_list =['poi', 'salary', 'to_messages', 'deferral_payments', 
'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 
'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 
'expenses', 'loan_advances', 'from_messages', 'other', 
'from_this_person_to_poi', 'director_fees', 
'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',
'fraction_from_poi','fraction_to_poi']

data_select = featureFormat(data_dict, features_select_list)

for point in data_select:
    emails_to_poi = point[features_select_list.index('fraction_to_poi')]
    emails_from_poi = point[features_select_list.index('fraction_from_poi')]
    plt.scatter( emails_from_poi, emails_to_poi)
    if point[features_select_list.index('poi')]==1:
        plt.scatter(emails_from_poi,emails_to_poi,color='r')

plt.xlabel("Share of emails from POI")
plt.ylabel("Share of emails to POI")
plt.title("Relationship between share of emails from & to POI and being POI")
plt.savefig("from_to_poi_and_poi_share.png")
plt.show()


my_dataset = data_dict

##################
# Feature Selection

# There are different ways how to select the best features.
# Univariate- SelectKBest or SelectKPercentile
# We can also use the decision tree

### Extract features and labels from  dataset for feature selection

labels_select,features_select = targetFeatureSplit(data_select)



# Generate dictionary with all the features as keys and 0 as value
# This helps us by the following feature selection function

def dict_feature(listing):
    dictionary={}
    for item in listing:
        if item!='poi':
            dictionary[item]=0.0
    return dictionary


def feature_select_fct(labels,features,listing,repition):
    dictionary=dict_feature(listing)
    result={}
    i=0
    while i < repition:
        # split the data into train and test
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3)
        # perform decision tree classifier
        tree_select=tree.DecisionTreeClassifier()
        tree_select.fit(features_train,labels_train)
        # Generate the importances
        importances=tree_select.feature_importances_
        for j in range(0,len(importances)):
            dictionary[listing[j+1]] += importances[j]
        i += 1
    for item in dictionary.keys():
        result[item]=dictionary[item]/repition
    return result

first_round=feature_select_fct(labels_select,features_select,features_select_list,1000.0)


#########################
# Second round - use only these features with score above 0.02

features_select_l2 = ['poi']
for item in first_round.keys():
    if first_round[item]>0.02:
        features_select_l2 += [item]
        
data_select_2 = featureFormat(data_dict, features_select_l2)    
labels_select_2,features_select_2 = targetFeatureSplit(data_select_2)

second_round=feature_select_fct(labels_select_2,features_select_2,features_select_l2,1000.0)   

###################
# Third round - use only these features with score above 0.03

features_select_l3 = ['poi']
for item in second_round.keys():
    if second_round[item]>0.03:
        features_select_l3 += [item]
    
data_select_3 = featureFormat(data_dict, features_select_l3)    
labels_select_3,features_select_3 = targetFeatureSplit(data_select_3)

third_round=feature_select_fct(labels_select_3,features_select_3,features_select_l3,1000.0) 


##########
# SelectKbest

def feature_select_kbest(labels,features,listing,repition,k_num):
    dictionary=dict_feature(listing)
    result={}
    i=0
    while i < repition:
        # split the data into train and test
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3)
        # perform selectKBest
        kbest=SelectKBest(f_classif,k=k_num)
        f_new = kbest.fit(features_train,labels_train)
        # Generate the importances
        importances= f_new.scores_
        for j in range(0,len(importances)):
            dictionary[listing[j+1]] += importances[j]
        i += 1
    for item in dictionary.keys():
        result[item]=dictionary[item]/repition
    return result

trial = feature_select_kbest(labels_select,features_select,features_select_list,1000.0,5)


# Create labels and features for testing performance of Importances DT

def create_imp_l_f(labels,features,listing,repition,k_num):
    result=['poi']
    trial=feature_select_fct(labels,features,listing,repition)
    sort_trial=sorted(trial.items(),key=operator.itemgetter(1))
    sort_trial.reverse()
    for item in sort_trial:
        if math.isnan(item[1]):
            sort_trial.remove(item)
    sort_trial=sort_trial[:k_num]
    for item in sort_trial:
        result += [item[0]] 
    data = featureFormat(data_dict, result)
    lab, feat = targetFeatureSplit(data)
    return lab, feat



# Create labels and features for testing performance of SeleckKBest
def create_l_f(labels,features,listing,repition,k_num):
    result=['poi']
    trial=feature_select_kbest(labels,features,listing,repition,k_num)
    sort_trial=sorted(trial.items(),key=operator.itemgetter(1))
    sort_trial.reverse()
    for item in sort_trial:
        if math.isnan(item[1]):
            sort_trial.remove(item)
    sort_trial=sort_trial[:k_num]
    for item in sort_trial:
        result += [item[0]] 
    data = featureFormat(data_dict, result)
    lab, feat = targetFeatureSplit(data)
    return lab, feat

###########
# Evaluation of algorithm by different kbest - see the section after
# crossvalidation function      
# Based on the results from that we can see that following features
# might play an important role 'exercised_stock_options', 'bonus',
#'shared_receipt_with_poi', 'total_stock_value', 'expenses', 'other',
#'fraction_to_poi'

features_combi =['exercised_stock_options', 'bonus',
'shared_receipt_with_poi', 'total_stock_value', 'expenses', 'other',
'fraction_to_poi']

def all_comb(listing):
    comb=['poi']
    result=[]
    for item in listing:
        comb += [item]
        result += [comb]
        comb=['poi']
    # All other combinations
    for i in range(2,len(listing)+1):
        for item in combinations(listing,i):
            comb += list(item)
            result += [comb]
            comb=['poi']
    return result
    

# Examination of each combination on the f1 score follows after crossvalidation


#####################
### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#######################
# Split in training / testing dataset
# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=40)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


# StratifiedShuffleSplit crossvalidation and storage of the evaluation metrics

def crossvalidation(classifier, features, labels,clf_string):
    accuracy=[]
    precision=[]
    recall=[]
    f1score=[]
    test_size=[]
    n_pois=[]
    t_pos=[]
    f_pos=[]
    t_neg=[]
    f_neg=[]

    # kfold
    kf=StratifiedShuffleSplit(labels,1000,test_size=0.3,random_state=40) 
    for train_indices, test_indices in kf:
        #make training and testing datasets
        features_train=[features[ii] for ii in train_indices]
        features_test=[features[ii] for ii in test_indices]
        labels_train=[labels[ii] for ii in train_indices]
        labels_test=[labels[ii] for ii in test_indices]
        # fit the classifier and make predictions
        classifier.fit(features_train, labels_train)
        pred=classifier.predict(features_test)
        
        # Evaluation metrics
        rec=recall_score(labels_test,pred)
        size=len(labels_test)
        pois=sum(labels_test)
        true_pos =rec*pois
        false_neg=pois-true_pos
        false_pos=sum(pred)-true_pos
        true_neg=size-true_pos-false_neg-false_pos
        
        # Store the evaluations

        test_size += [size]
        n_pois    += [pois]
        t_pos     += [true_pos]
        f_pos     += [false_pos]
        t_neg     += [true_neg]
        f_neg     += [false_neg]
        
    # Create averages
    test_size = sum(test_size)
    n_pois    = sum(n_pois)
    t_pos     = sum(t_pos)
    f_pos     = sum(f_pos)
    t_neg     = sum(t_neg)
    f_neg     = sum(f_neg)
    accuracy  = 1.0*(t_pos+t_neg)/test_size
    precision = 1.0*t_pos/(t_pos+f_pos)
    recall    = 1.0*t_pos/(t_pos+f_neg)
    f1score   = 2.0*t_pos/(2*t_pos+f_pos+f_neg)

    print "The accuracy score by " + clf_string + ": " + str(accuracy)
    print "Precision by " + clf_string + ": " + str(precision)
    print "Recall by " + clf_string + ": " + str(recall)
    print "F1 score by " + clf_string + ": " + str(f1score)
    print "The number of true positives by " + clf_string + ": " + str(t_pos)
    print "The number of false positives by " + clf_string + ": " + str(f_pos)
    print "The number of true negatives by " + clf_string + ": " + str(t_neg)
    print "The number of false negatives by " + clf_string + ": " + str(f_neg)
    print "---------------------------------------------"

    return accuracy,precision,recall,f1score,test_size,n_pois,t_pos,f_pos,t_neg,f_neg





#####################
# Features Selection - Performance by different kbest
####################

# Performance testing of importances DT

def perf_importances(classifier,labels,features,listing,repition,clf_string):
    k_list=[]
    prec_list=[]
    rec_list=[]
    f1_list=[]
    for i in range(1,20):
        lab,feat=create_imp_l_f(labels,features,listing,repition,i)
        temp_result=crossvalidation(classifier,feat,lab,clf_string)
        k_list += [i]
        prec_list +=[temp_result[1]]
        rec_list +=[temp_result[2]]
        f1_list +=[temp_result[3]]
    return k_list,prec_list,rec_list,f1_list


# Performance testing of kbest

def perf_kbest(classifier,labels,features,listing,repition,clf_string):
    k_list=[]
    prec_list=[]
    rec_list=[]
    f1_list=[]
    for i in range(1,20):
        lab,feat=create_l_f(labels,features,listing,repition,i)
        temp_result=crossvalidation(classifier,feat,lab,clf_string)
        k_list += [i]
        prec_list +=[temp_result[1]]
        rec_list +=[temp_result[2]]
        f1_list +=[temp_result[3]]
    return k_list,prec_list,rec_list,f1_list
    
# Feature Selection can be found by each Classifier separately
    

#################
# What features are important


def score_comb(listing, dataset, classifier,repition,clf_string):
    combination=all_comb(listing)
    dictionary=dict((tuple(item), {}) for item in combination)

    # Store the results
    for item in combination:
        data = featureFormat(my_dataset, item, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        temp_result=crossvalidation(classifier,features,labels,clf_string)
        dictionary[tuple(item)]['precision']=temp_result[1]
        dictionary[tuple(item)]['recall']=temp_result[2]
        dictionary[tuple(item)]['f1score']=temp_result[3]
    return dictionary
        

# Find max f1 score

def max_f1_find(listing, dataset, classifier,repition,clf_string):
    dictionary=score_comb(listing, dataset, classifier,repition,clf_string)
    maximum=0
    result={}

    for item in dictionary.keys():
        if dictionary[item]['f1score']>=maximum:
            maximum=dictionary[item]['f1score']
            key=item
    result[key]=dictionary[key]
    return result


##############################
# Naive Bayes (Gaussian)
#################################

nb=GaussianNB()
crossvalidation(nb,features,labels,"Gausian NB")

# feature selection

# importances
k,p_k,r_k,f_k= perf_importances(nb,labels_select,features_select,features_select_list,1000.0,"Gausian NB")

plt.plot(k,p_k,"g",label='Precision')
plt.plot(k,r_k,"b",label='Recall')
plt.plot(k,f_k,"r",label='F1 Score')
plt.xticks(k,k)
plt.xlabel("K best Features (DT Importance)")
plt.ylabel("Score")
plt.legend(loc='upper left')
plt.title(" Feature Importances and Evaluation Metrics by Gaussian NB")
plt.savefig("k_imp_NB.png")
plt.show()


# kbest
k,p_k,r_k,f_k= perf_kbest(nb,labels_select,features_select,features_select_list,1000.0,"Gausian NB")

plt.plot(k,p_k,"g",label='Precision')
plt.plot(k,r_k,"b",label='Recall')
plt.plot(k,f_k,"r",label='F1 Score')
plt.xticks(k,k)
plt.xlabel("K best Features")
plt.ylabel("Score")
plt.legend(loc='upper left')
plt.title("K best Features and Evaluation Metrics by Gaussian NB")
plt.savefig("kbest_NB.png")
plt.show()

# find optimal combi

max_f1_find(features_combi, my_dataset, nb,1000,"Gaussian NB")


##############################
# Decision Tree
#################################

dt=tree.DecisionTreeClassifier(random_state=40)
crossvalidation(dt,features,labels,"default decision tree")

###############
# Comment out due to performance

"""

# importances
k,p_k,r_k,f_k= perf_importances(dt,labels_select,features_select,features_select_list,1000.0,"default decision tree")

plt.plot(k,p_k,"g",label='Precision')
plt.plot(k,r_k,"b",label='Recall')
plt.plot(k,f_k,"r",label='F1 Score')
plt.xticks(k,k)
plt.xlabel("K best Features (DT Importance)")
plt.ylabel("Score")
plt.legend(loc='upper right')
plt.title(" Feature Importances and Evaluation Metrics by Default Decision Tree")
plt.savefig("k_imp_DT.png")
plt.show()

# kbest
k,p_k,r_k,f_k= perf_kbest(dt,labels_select,features_select,features_select_list,1000.0,"default decision tree")

plt.plot(k,p_k,"g",label='Precision')
plt.plot(k,r_k,"b",label='Recall')
plt.plot(k,f_k,"r",label='F1 Score')
plt.xticks(k,k)
plt.xlabel("K best Features")
plt.ylabel("Score")
plt.legend(loc='upper right')
plt.title("K best Features and Evaluation Metrics by Default Decision Tree")
plt.savefig("kbest_DT.png")
plt.show()


# find optimal combi

max_f1_find(features_combi, my_dataset, dt,1000,"default decision tree")
"""

##############################
# KNN
#################################

neigh = KNeighborsClassifier()
crossvalidation(neigh,features,labels,"default KNN")

###############
# Comment out due to performance
"""
# importances
k,p_k,r_k,f_k= perf_importances(neigh,labels_select,features_select,features_select_list,1000.0,"default KNN")

plt.plot(k,p_k,"g",label='Precision')
plt.plot(k,r_k,"b",label='Recall')
plt.plot(k,f_k,"r",label='F1 Score')
plt.xticks(k,k)
plt.xlabel("K best Features (DT Importance)")
plt.ylabel("Score")
plt.legend(loc='upper right')
plt.title(" Feature Importances and Evaluation Metrics by Default KNN")
plt.savefig("k_imp_KNN.png")
plt.show()

# kbest
k,p_k,r_k,f_k= perf_kbest(neigh,labels_select,features_select,features_select_list,1000.0,"default KNN")

plt.plot(k,p_k,"g",label='Precision')
plt.plot(k,r_k,"b",label='Recall')
plt.plot(k,f_k,"r",label='F1 Score')
plt.xticks(k,k)
plt.xlabel("K best Features")
plt.ylabel("Score")
plt.legend(loc='upper right')
plt.title("K best Features and Evaluation Metrics by default KNN")
plt.savefig("kbest_knn.png")
plt.show()

# find optimal combi

max_f1_find(features_combi, my_dataset, neigh,1000,"default KNN")
"""

##############################
# Random Forest
#################################

random_f=RandomForestClassifier(random_state=40)
crossvalidation(random_f,features,labels,"default random forest")

###############
# Comment out due to performance
"""
# importances
k,p_k,r_k,f_k= perf_importances(random_f,labels_select,features_select,features_select_list,1000.0,"default random forest")

plt.plot(k,p_k,"g",label='Precision')
plt.plot(k,r_k,"b",label='Recall')
plt.plot(k,f_k,"r",label='F1 Score')
plt.xticks(k,k)
plt.xlabel("K best Features (DT Importance)")
plt.ylabel("Score")
plt.legend(loc='upper right')
plt.title(" Feature Importances and Evaluation Metrics by Default Random Forest")
plt.savefig("k_imp_RF.png")
plt.show()

# kbest
k,p_k,r_k,f_k= perf_kbest(random_f,labels_select,features_select,features_select_list,1000.0,"default random forest")

plt.plot(k,p_k,"g",label='Precision')
plt.plot(k,r_k,"b",label='Recall')
plt.plot(k,f_k,"r",label='F1 Score')
plt.xticks(k,k)
plt.xlabel("K best Features")
plt.ylabel("Score")
plt.legend(loc='upper right')
plt.title("K best Features and Evaluation Metrics by Default Random Forest")
plt.savefig("kbest_RF.png")
plt.show()


# find optimal combi

max_f1_find(features_combi, my_dataset, random_f,1000,"default random forest")
"""

#####################################
# Adaboost
#####################################

adaboost=AdaBoostClassifier(random_state=40)
crossvalidation(adaboost,features,labels,"default adaboost")

###############
# Comment out due to performance
"""
# importances
k,p_k,r_k,f_k= perf_importances(dt,labels_select,features_select,features_select_list,1000.0,"default adaboost")

plt.plot(k,p_k,"g",label='Precision')
plt.plot(k,r_k,"b",label='Recall')
plt.plot(k,f_k,"r",label='F1 Score')
plt.xticks(k,k)
plt.xlabel("K best Features (DT Importance)")
plt.ylabel("Score")
plt.legend(loc='upper right')
plt.title(" Feature Importances and Evaluation Metrics by Default Adaboost")
plt.savefig("k_imp_adaboost.png")
plt.show()

# kbest
k,p_k,r_k,f_k= perf_kbest(adaboost,labels_select,features_select,features_select_list,1000.0,"default adaboost")

plt.plot(k,p_k,"g",label='Precision')
plt.plot(k,r_k,"b",label='Recall')
plt.plot(k,f_k,"r",label='F1 Score')
plt.xticks(k,k)
plt.xlabel("K best Features")
plt.ylabel("Score")
plt.legend(loc='upper center')
plt.title("K best Features and Evaluation Metrics by Default Adaboost")
plt.savefig("kbest_adaboost.png")
plt.show()


# find optimal combi

max_f1_find(features_combi, my_dataset, adaboost,1000,"default adaboost")
"""

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

######
#  Tuning parameters  - Decision Tree GridCV

  
param_grid = {
         'criterion': ['gini','entropy'],
         'min_samples_split': [2,6,10,20,50],
          }

cv=StratifiedShuffleSplit(labels,1000,test_size=0.3,random_state=40)

clf_opt = GridSearchCV(tree.DecisionTreeClassifier(random_state=40), param_grid,scoring="f1",cv=cv)
clf_opt=clf_opt.fit(features,labels)
print "Best estimator found by grid search:"
print clf_opt.best_estimator_




############
# Random forest - optimized 

###############
# Comment out due to performance
"""
param_grid = {
         'criterion': ['gini','entropy'],
         'min_samples_split': [2,6,10,20,50],
         'n_estimators': [2,5,10,20,50,80],
          }
       
clf_opt = GridSearchCV(RandomForestClassifier(random_state=40), param_grid,scoring="f1",cv=cv)
clf_opt=clf_opt.fit(features,labels)
print "Best estimator found by grid search:"
print clf_opt.best_estimator_
"""

clf=RandomForestClassifier(min_samples_split=2, n_estimators=50, \
            random_state=40,criterion='entropy')

crossvalidation(clf,features,labels,"optimized random forest")


###################
# A P E N D I X
###################

###################
# No POI's featrues

features_list_ap1 = ['poi','fraction_to_poi',\
'bonus','exercised_stock_options'] 

data_ap1 = featureFormat(data_dict, features_list_ap1, sort_keys = True)
labels_ap1, features_ap1 = targetFeatureSplit(data_ap1)


# Comparison with and without fraction to poi by DT and RF

crossvalidation(dt,features_ap1,labels_ap1,"default decision tree")
crossvalidation(clf,features_ap1,labels_ap1,"optimized random forest")

"""
# Optimal Random Forest in this case

rf_optimal=RandomForestClassifier(min_samples_split=6, n_estimators=2, \
            random_state=40)

crossvalidation(rf_optimal,features_ap1,labels_ap1,"optimized random forest")

"""

#################
# PCA Analysis

# Preparation of the features and labels for pca

features_pca_list =['poi', 'salary', 'to_messages', 'deferral_payments', 
'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 
'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 
'expenses', 'loan_advances', 'from_messages', 'other', 
'from_this_person_to_poi', 'director_fees', 
'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',
'fraction_from_poi','fraction_to_poi']

data_pca = featureFormat(data_dict, features_pca_list)

labels_pca, features_pca = targetFeatureSplit(data_pca)

features_ptrain, features_ptest, labels_ptrain, labels_ptest = \
    train_test_split(features_pca, labels_pca, test_size=0.3, random_state=40)

# Pipeline (PCA and Classifier)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()), ('nb', GaussianNB())]

clf_prep = Pipeline(estimators)

# Optimizing - After trying out different algorithms I have ended up with NB

param_grid = {
         'reduce_dim__n_components': [2,4,5,10,14,20],
          }

cv_pca=StratifiedShuffleSplit(labels_pca,1000,test_size=0.3,random_state=40)

clf_opt = GridSearchCV(clf_prep, param_grid,scoring="f1",cv=cv_pca)
clf_opt = clf_opt.fit(features_pca, labels_pca)
print "Best estimator found by grid search:"
print clf_opt.best_estimator_ 

############
# Optimal

estimators_opt = [('reduce_dim', PCA(n_components=2)),\
('nb',GaussianNB())]

clf_pca=Pipeline(estimators_opt)

crossvalidation(clf_pca,features_pca,labels_pca,"optimized PCA with NB")





### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)