import numpy as np
import pandas as pd 
import fire
import datetime
import sys
sys.path.append("..")

def txt_to_list(list_txt):
	# list_txt: 'ensemble_list.txt'
	file = open(list_txt)
	npy_result_list = [line.strip() for line in file.readlines()]
	return npy_result_list

def ensemble(npy_result_list):
	t = datetime.datetime.now()
	t = str(t).split('.')[0]

	prob_list = [np.load('../result/'+npy_file_name) for npy_file_name in npy_result_list]
	pre_res = np.zeros(prob_list[0].shape)
	for prob in prob_list:
		pre_res += prob
	final_res = pre_res / len(npy_result_list)

	ensemble_result = np.argmax(final_res, axis=1)
	test_id = range(len(final_res))
	
	test_pred = pd.DataFrame({'id':test_id,'class':ensemble_result})
	test_pred['class'] = (test_pred['class']+1).astype(int)
	test_pred[['id','class']].to_csv('result/'+'ensemble_result_'+t+'.csv',index=None)

	print('{} saved.'.format('ensemble_result_'+t+'.csv'))

def main(list_txt):
	#list_txt: 'ensemble_list.txt' 将需要集成的模型结果放到这个txt文件中，然后放到ensemble文件夹下
	#在ensemble文件夹下执行该py文件
	npy_result_list = txt_to_list(list_txt)
	ensemble(npy_result_list)

if __name__ == '__main__':
	fire.Fire()

# python ensemble.py main --list_txt='ensemble_list.txt'
