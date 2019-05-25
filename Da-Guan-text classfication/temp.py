import torch
import time
import torch.nn.functional as F
import models
from torchtext.vocab import Vectors
from data import buildDataset
import pandas as pd
import os
import fire
import numpy as np
from torch.nn import init
from torchtext import data

def main(name, device):
	#name:FastText_word_0.7407834190815604.pth
	#device:测试时请指定要用的GPU，因为这和模型初试用的GPU不一定相同

	#--------------load_model
	print('Loading model:',name,'...................')
	stateDict_and_config = torch.load('./snapshot/'+name)
	args = stateDict_and_config['config']

	print('model sucessfully load in.')
	args.device = device
	args.print_config()

	#load data
	print('Building test Iterator.................')
	test_iter = load_test_data(args)
	print('test Iterator made.')

	#init model
	print('Init',args.model,'..................')
	model = getattr(models, args.model)(args)
	print(model)

	if args.cuda:
		torch.cuda.set_device(device)
		torch.cuda.manual_seed(args.seed)  # set random seed for gpu
		model.cuda()

	##################################
	model.load_state_dict(stateDict_and_config['state_dict'])#####
	##################################

	#--------------------test
	if not os.path.exists('result/'):
		os.mkdir('result/')
	probs, test_pred = test(model, test_iter, args)
	result_path = 'result/' + '{}_{}_{}'.format(args.model, args.Id, args.best_score)
	np.save('{}.npy'.format(result_path), probs)
	print('Prob result {}.npy saved!'.format(result_path))

	test_pred[['id', 'class']].to_csv('{}.csv'.format(result_path), index=None)
	print('Result {}.csv saved!'.format(result_path))

def load_test_data(option):

	#======
	Text_filed = data.Field(sequential=True, fix_length=option.max_text_len)
	#Label_field = data.Field(sequential=False, use_vocab=False)

	#======
	test_path = option.data_path + option.text_type + '/test_set.csv'
	#if option.aug:
		#print('make augementation datasets!')

	test = buildDataset(test_path, text_field=Text_filed, label_field=None, text_type=option.text_type, test=True) 

	#======
	cache = '.vector_cache'
	if not os.path.exists(cache):
		os.mkdir(cache)
	embedding_path = '{}/{}_{}_.txt'.format(option.embedding_path, option.text_type, option.emb_size)
	print('embedding_path:',embedding_path)#
	
	vectors = Vectors(name=embedding_path, cache=cache)
	print('load word2vec vectors from {}'.format(embedding_path))
	vectors.unk_init = init.xavier_uniform_  
	#如何指定 Vector 缺失值的初始化方式: vector.unk_init = init.xavier_uniform 这种方式指定完再传入 build_vocab

	#======构建vocab
	print('building {} vocabulary......'.format(option.text_type))
	Text_filed.build_vocab(test, min_freq=option.min_freq, vectors=vectors)
	print('vocabulary has been made!\n')

	#======构建Iterator
	'''
	1. 在 test_iter, shuffle, sort, repeat一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
	2. 如果输入变长序列，sort_within_batch需要设置成true，使每个batch内数据按照sort_key降序进行排序
	'''
	print('building {} Iterator......'.format(option.text_type))
	test_iter = data.Iterator(dataset=test, batch_size=option.batch_size, shuffle=False, sort=False, repeat=False,
								device=option.device)
	print('Iterator has been made!\n')

	return test_iter

def test(model, test_data, args):
	# 将模型设置为测试模式
	# 生成测试集预测数据
	model.eval()

	result = np.zeros((0,))
	probs_list = []


	with torch.no_grad():
		print('begin testing. just waiting...........')
		for batch in test_data:

			text = batch.text
			if args.cuda:
				text = text.cuda()

			outputs = model(text)

			probs = F.softmax(outputs, dim=1)
			probs_list.append(probs.cpu().numpy())

			pred = outputs.max(1)[1]
			result = np.hstack((result, pred.cpu().numpy()))

	prob_cat = np.concatenate(probs_list, axis=0)

	test = pd.read_csv('./datasets/test_set.csv')
	test_id = test['id'].copy()
	test_pred = pd.DataFrame({'id': test_id, 'class': result})
	test_pred['class'] = (test_pred['class']+1).astype(int)

	return prob_cat, test_pred

if __name__ == '__main__':
	fire.Fire()

#python temp.py main --name='FastText_word_0.7407834190815604.pth' --device='1'
#python temp.py main --name='FastText_word_0.7381621000940348.pth' --device='1'
#python temp.py main --name='TextCNN_word_0.7425071189854272.pth' --device='0'

#GRU_word_0.7681885459985601.pth

#python temp.py main --name='GRU_word_0.7681885459985601.pth' --device='0'