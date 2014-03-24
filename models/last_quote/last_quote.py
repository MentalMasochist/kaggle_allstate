"""
file: last_quote.property
author: Richard Brooks
contact: richardbrks@gmail.com
description: predicting the quote by assming it to be the last shopping quote quoted

To do:
	[] need to be able the get a CV accuacy automatically
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


def get_k_folds(df_data, num_folds, seed=None):
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
	return folded_data


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


def main_predict(df_data, ls_cust_id):
	"""
	apply model to test data
	"""
	ls_res = []
	for cust_id in ls_cust_id:
		last_pt = df_data[(df_data['customer_ID'] == cust_id) & (df_data['record_type'] == 0)]['shopping_pt'].max()
		last_interaction = df_data[(df_data['customer_ID'] == cust_id) & (df_data['shopping_pt'] == last_pt)]
		plan = get_plan(last_interaction)
		ls_res.append([cust_id, plan])
	return ls_res


def check_res(df_data, ls_res):
	"""
	checks results from prediction
	"""
	tot_ct = 0
	match_ct = 0
	for res in ls_res:
		tot_ct += 1
		cust_id = res[0]
		pred_plan = res[1]
		purchase_quote = df_data[(df_data['customer_ID'] == cust_id) & (df_data['record_type'] == 1)]
		act_plan = get_plan(purchase_quote)
		if (pred_plan == act_plan):
			match_ct += 1
	acc = match_ct/float(tot_ct)
	return acc


def CV_test(df_data, num_folds):
	"""
	cross-validation testing
	"""
	iter_ct = 0
	cv_acc = 0 
	folded_data = get_k_folds(df_data, num_folds)
	for df_fold in folded_data:
		print "fold %d / %d" % (iter_ct+1, num_folds)
		ls_cust_id = list(np.unique(df_fold['customer_ID']))
		ls_res = main_predict(df_fold, ls_cust_id)
		acc = check_res(df_fold, ls_res)
		cv_acc += acc
		iter_ct += 1
	cv_acc = cv_acc/float(num_folds)
	return cv_acc


def main():
	f = '../../data/train.csv'
	df_data = pd.read_csv(f)
	ls_num_folds = [2,5,10]
	for num_folds in ls_num_folds:
		cv_acc = CV_test(df_data, num_folds)
		print "acc: ", cv_acc
	return


if __name__ == "__main__":
	main()