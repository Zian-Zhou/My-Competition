# -*- coding: utf-8 -*-

from torchtext import data
import pandas as pd
from torchtext.vocab import Vectors
from tqdm import tqdm
from torch.nn import init
import random
import os
import numpy as np

class buildDataset(data.Dataset):
	name = 'Generate Dataset for torch.'

	def __init__(self, path, text_field, label_field, text_type='word', test=False, aug=False, **kwargs):
		'''
		path: 数据路径（train、val、test）
		text_fielf,label_field: data.Filed()
		text_type: 'word' or 'article'
		test: False-当前为训练数据集；True-当前为测试数据集，则不需要加载label
		aug: 是否要做数据集增强，当test=False的时候有效
		'''
		fields = [('text',text_field),('label',label_field)]
		examples = []

		print('reading data from {}........'.format(path))
		csv_data = pd.read_csv(path)
		print('finished.\n')

		if text_type == 'word':
			text_type='word_seg'

		if test:
			for text in tqdm(csv_data[text_type]):
				#tqdm在循环体中用，展示进度条
				examples.append(data.Example.fromlist([text,None], fields))
				#data.Example: 用来表示一个样本
		else:
			for text, label in tqdm(zip(csv_data[text_type],csv_data['class'])):
				if aug:
					#做数据集增强
					rate = random.random()
					if rate>0.5:
						text = self.dropout(text)
					else:
						text = self.shuffle(text)
				examples.append(data.Example.fromlist([text,label-1], fields))

		super(buildDataset, self).__init__(examples, fields, **kwargs)

	def shuffle(self, text):
		text = np.random.permutation(text.strip().split())
		#strip(): 去除首尾空格
		return ' '.join(text)

	def dropout(self, text, p=0.7):
		text = text.strip().split()
		len_text = len(text)
		indexs = np.random.choice(len_text,int(len_text * p))
		for i in indexs:
			text[i] = ''
		return ' '.join(text)

	@staticmethod
	def sort_key(ex):
		return len(ex.text)

def load_data(option):
	#======
	Text_filed = data.Field(sequential=True, fix_length=option.max_text_len)
	Label_field = data.Field(sequential=False, use_vocab=False)

	#======
	train_path = option.data_path + option.text_type + '/train_set.csv'
	val_path = option.data_path + option.text_type + '/val_set.csv'
	test_path = option.data_path + option.text_type + '/test_set.csv'
	if option.aug:
		print('make augementation datasets!')
		
	train = buildDataset(train_path, text_field=Text_filed, label_field=Label_field, text_type=option.text_type, test=False,
							aug=option.aug)
	val = buildDataset(val_path, text_field=Text_filed, label_field=Label_field, text_type=option.text_type, test=False)
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
	Text_filed.build_vocab(train, val, test, min_freq=option.min_freq, vectors=vectors)
	print('vocabulary has been made!\n')

	#======构建Iterator
	'''
	1. 在 test_iter, shuffle, sort, repeat一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
	2. 如果输入变长序列，sort_within_batch需要设置成true，使每个batch内数据按照sort_key降序进行排序
	'''
	print('building {} Iterator......'.format(option.text_type))
	train_iter = data.BucketIterator(dataset=train, batch_size=option.batch_size, shuffle=True, sort_within_batch=False,
										repeat=False, device=option.device)
	val_iter = data.Iterator(dataset=val, batch_size=option.batch_size, shuffle=False, sort=False, repeat=False,
								device=option.device)
	test_iter = data.Iterator(dataset=test, batch_size=option.batch_size, shuffle=False, sort=False, repeat=False,
								device=option.device)
	print('Iterator has been made!\n')

	return train_iter, val_iter, test_iter, len(Text_filed.vocab), Text_filed.vocab.vectors