#由于服务器内存可能不足，gen_test_result.py不能完整执行，就把result结果保存下来
#在本地拼接出最后结果

import pandas as pd 
import fire

def main(name):
	#name: FastText_word1_0.7407834190815604_predict
	#注意，带predict的csv文件表示从服务器上进行预测的数据表，需要将其和test_set的id拼接到一起，生成最终结果
	result_path = './result/'+'_'.join( name.split('_')[:-1])

	print('Loading test_set.csv')
	test = pd.read_csv('../new_data/test_set.csv')

	print('Loading test_predict.csv')
	test_predict = pd.read_csv('./test_predict/' + name + '.csv')
	'''
    test = pd.read_csv('./datasets/test_set.csv')
    test_id = test['id'].copy()
    test_pred = pd.DataFrame({'id': test_id, 'class': result})
    test_pred['class'] = (test_pred['class']+1).astype(int)

    test_pred[['id', 'class']].to_csv('{}.csv'.format(result_path), index=None)
    print('Result {}.csv saved!'.format(result_path))
	'''
	test_id = test['id'].copy()
	result = test_predict['class'].copy().astype(int)

	test_pred = pd.DataFrame({'id':test_id,'class':result})

	test_pred[['id', 'class']].to_csv('{}.csv'.format(result_path), index=None)
	print('Result {}.csv saved!'.format(result_path))

if __name__ == '__main__':
	fire.Fire()