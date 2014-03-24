"""
To do:
	[] need to be able the get a CV accuacy automatically
		[] do everything for a fold at a time
			- i.e. dont let k-fold be the inner loop

	[] need to search through all sub
	[] need to vett code
"""

# python modules
import csv
import copy
import sys

# 3rd party modules
import numpy as np 
import pandas as pd 
import scipy

# custom modules
import analysis_handler
import NB_suite

#############################################################################################
###  Creating Training Set
#############################################################################################

def create_training_set():
	"""
	need to do some cleaning by hand
	"""
	# inputs
	f_in = "../../data/train.csv"
	f_out = "../../data/train_nb_purchase_records.csv"
	# only include needed features
	col_to_include = ['customer_ID', 'state', 'location', 'group_size', 
					  'homeowner', 'car_age', 'car_value', 'risk_factor', 
					  'age_oldest', 'age_youngest', 'married_couple', 'C_previous',  
					  'duration_previous', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
	df_data = pd.read_csv(f_in)
	df_data = df_data[df_data['record_type'] == 1]
	df_data =  df_data[col_to_include]
	# clean data
	mask = 	df_data['risk_factor'].isnull()
	df_data.ix[mask,['risk_factor']] = -1

	mask = 	df_data['C_previous'].isnull()
	df_data.ix[mask,['C_previous']] = 0

	mask = 	df_data['duration_previous'].isnull()
	df_data.ix[mask,['duration_previous']] = 0
	# write data
	df_data.to_csv(f_out, index=False)
	return

#############################################################################################
###  Naive Bayes Analysis
#############################################################################################

class UniGaussian:
	"""
	stores Gaussian parameters
	"""
	def __init__(self, mean, var):
		self.mean = mean
		self.var = var
		return

	def likelihood(self, x):
		if self.var == 0:
			ret = 0
		else:
			ret = 1.0/np.sqrt(2.0*np.pi*self.var) * np.exp(-((x-self.mean)**2.0)/(2.0*np.pi*self.var))
		return ret

class HistDict:
	"""
	creates a histogram based discrete probability distribution
	"""
	def __init__(self, data):
		ls_data = list(data)
		sz = len(ls_data)
		hist = {x: ls_data.count(x)/float(sz) for x in set(ls_data)}
		self.hist = hist 
		return
	
	def get_prob(self, key):
		if key in self.hist.keys():
			prob = self.hist[key]
		else:
			prob = 0.0
		return prob


def smoothing_fn(prob, instance_ct, smoothing_factor=0.00001):
	"""
	smoothing function for likelihoods
	"""
	return (prob + smoothing_factor) / float(smoothing_factor*instance_ct+1) 


def get_plan(df_quote):
	"""
	get plan from quote
	"""
	A = str(df_quote['A'].values[0])
	B = str(df_quote['B'].values[0])
	C = str(df_quote['C'].values[0])
	D = str(df_quote['D'].values[0])
	E = str(df_quote['E'].values[0])
	F = str(df_quote['F'].values[0])
	G = str(df_quote['G'].values[0])
	plan = ''.join([A,B,C,D,E,F,G])
	return plan	


def df_get_k_folds(df_data, num_folds, seed=1987):
	"""
	splits data into k groups
	by customer ID
	"""
	np.random.seed(seed)

	na_cust_id = np.array(np.unique(df_data['customer_ID']))
	sz = na_cust_id.shape[0]
	mask = range(sz)
	np.random.shuffle(mask)
	na_cust_id = na_cust_id[mask]

	folded_data = []
	for k in range(num_folds):
		l = round(k * (sz/float(num_folds)))
		r = round((k+1) * (sz/float(num_folds)))
		folded_data.append(df_data[df_data['customer_ID'].isin(na_cust_id[l:r])])
	print folded_data[0]['customer_ID'].head()
	return folded_data


def df_split_by_class(df_data, label_cols):
	"""
	uses recursion to split data into it's classes
	very poor computations time because of transfer between df and na and back to df, 
	but we're hacking here -- if you want some production code, pay me
	"""
	ls = []
	col = label_cols[0]
	keys = list(np.unique(df_data[col]))
	if len(label_cols) == 1:
		for k in keys:
			ls.append(df_data[df_data[col] == k]) 
	else:
		for k in keys:
			sub_ls = df_split_by_class(df_data[df_data[col] == k], label_cols[1:])
			for grp in sub_ls:
				ls.append(grp)
	return ls


def NB_train(df_train, ft_cols, lbl_cols, ls_cont_ft, ls_disc_ft):
	"""
	returns:
		- prior probabilities
		- parameters for likelihood calculations 
			- Gaussian if features are coninutous
				- histogram frequnecy probabilities if discrete
		- list of clas ids used laster for accuracy checking 
	"""
	# split df by class
	ls_train_by_class = df_split_by_class(df_train, lbl_cols)
	# init parameters
	ls_priors = []
	ls_params = []
	ls_class_id = []
	for class_train_data in ls_train_by_class:
		# reset index
		class_train_data.index = range(class_train_data.shape[0]) 
		# get class id
		class_id = ''.join([str(cid) for cid in class_train_data[lbl_cols].ix[0]])
		ls_class_id.append(class_id)
		# get priors
		ls_priors.append(class_train_data.shape[0]/float(df_train.shape[0]))
		# get univariate gaussian parameters for each feature vector
		temp_params = []
		for ft in ft_cols:
			if ft in ls_cont_ft:
				mean = class_train_data[ft].mean()
				var = class_train_data[ft].var()
				temp_params.append(UniGaussian(mean,var))
			else:
				temp_params.append(HistDict(class_train_data[ft]))
		ls_params.append(temp_params)
	return ls_priors, ls_params, ls_class_id


def NB_test(df_test, ls_priors, ls_params, ls_class_id, ft_cols, lbl_cols, ls_cont_ft, instance_ct):
	"""
	- Compares the prior likelihood for each class for each feature vector in the test data
	- Returns the accuracy of predicting the test_data
	- Additive smoothing is used to help instances with 0 probability
	"""
	d_pred = {}
	d_act = {}
	match_ct = 0
	tot_ct = 0
	for idx in df_test.index:
		X = df_test.ix[idx]
		cust_id = X['customer_ID']
		ls_prob = []
		for cls in range(len(ls_class_id)):
			# init logprob with logprior
			log_cumprob = np.log(ls_priors[cls])
			# add loglikelihoods
			for i in range(len(ft_cols)):
				ft = ft_cols[i]
				if ft in ls_cont_ft:
					gauss_obj = ls_params[cls][i]
					prob = gauss_obj.likelihood(X[ft])
				else:
					hist_obj = ls_params[cls][i]
					prob = hist_obj.get_prob(X[ft])
				# smoothing probabilities
				prob = smoothing_fn(prob, instance_ct)
				log_cumprob += np.log(prob)
			ls_prob.append(log_cumprob)
		# choose class with largest log prob
		y_pred = ls_class_id[np.argmax(ls_prob)]
		d_pred[cust_id] = y_pred
		# compare with actual
		y_act = ''.join(list(X[lbl_cols].astype(str)))
		d_act[cust_id] = y_act		
		if y_pred == y_act:
			match_ct += 1
		else:
			pass
		tot_ct += 1
	acc = match_ct / float(tot_ct) 
	return acc, d_pred, d_act


def Gauss_NB_kfold(df_data, num_folds, ft_cols, lbl_cols, ls_cont_ft, ls_disc_ft):
	"""
	kfold cross-validation for Gaussian
	"""
	ls_folded_data = df_get_k_folds(df_data, num_folds)
	CV_acc = 0
	d_pred = {}
	d_act = {}
	for i in range(num_folds):
		print "fold %d / %d" % (i+1, num_folds)
		# group into training/test sets
		df_test_set = pd.DataFrame(None)
		df_train_set = pd.DataFrame(None)
		for j in range(num_folds):
			if i == j:
				df_test_set = df_test_set.append(ls_folded_data[j], ignore_index=True)
			else:
				df_train_set = df_train_set.append(ls_folded_data[j], ignore_index=True)
		# training
		ls_priors, ls_params, ls_class_id = NB_train(df_train_set, ft_cols, lbl_cols, ls_cont_ft, ls_disc_ft) 
		# testing
		instance_ct = df_train_set.shape[0]
		acc, d_pred_temp, d_act_temp = NB_test(df_test_set, ls_priors, ls_params, ls_class_id, ft_cols, lbl_cols, ls_cont_ft, instance_ct) 
		CV_acc += acc
		d_pred.update(d_pred_temp)
		d_act.update(d_act_temp)
	CV_acc = CV_acc / float(num_folds)
	return CV_acc, d_pred, d_act


def Gaussian_NaiveBayes():
	# inputs
	f_train = "../../data/train_nb_purchase_records.csv"
	f_test = "../../data/test_v2.csv"
	f_res = "../../results/nb_purchased_rec.csv"
	
	num_folds = 5
	ft_cols = ['state', 'location', 'group_size', 'homeowner', 
			   'car_age', 'car_value', 'risk_factor', 
			   'age_oldest', 'age_youngest', 'married_couple', 
			   'C_previous',  'duration_previous']
	ls_lbl_cols = [['A'], ['B'], ['C'], ['D'], ['E'], ['F'], ['G']]
	# lbl_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

	ls_cont_ft = ['car_age', 'age_oldest', 'age_youngest', 'duration_previous']
	ls_disc_ft = ['state', 'location', 'group_size', 'homeowner', 'car_value',
				  'risk_factor', 'married_couple', 'C_previous']

	df_train = pd.read_csv(f_train)
	df_train = df_train.ix[:10000]
	# df_test = pd.read_csv(f_test)

	pred_quote = {}
	act_quote = {}
	for lbl_cols in ls_lbl_cols:
		print lbl_cols
		cv_acc, d_pred, d_act  = Gauss_NB_kfold(df_train, num_folds, ft_cols, lbl_cols, ls_cont_ft, ls_disc_ft)
		pred_quote[lbl_cols[0]] = d_pred  
		act_quote[lbl_cols[0]] = d_act
		print "    %s cv_acc: %f" % (lbl_cols[0], cv_acc)
	df_pred_quote = pd.DataFrame(pred_quote)
	print "prediction"
	print df_pred_quote.head()
	df_act_quote = pd.DataFrame(pred_quote)
	print "\nactual"
	print df_act_quote.head()
	return



if __name__ == "__main__":
	# create_training_set()
	# test()
	Gaussian_NaiveBayes()