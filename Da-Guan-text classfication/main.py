# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F 
import models
import data
import pandas as pd 
import numpy as np
import os
import fire
import time
from config import Config
from sklearn import metrics

best_score = 0.0
t1 = time.time()

def test(model, test_data, args):
	# 将模型设置为测试模式
	# 生成测试集预测数据
	model.eval()

	result = np.zeros((0,))
	probs_list = []

	with torch.no_grad():

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

	'''#old version: have some problems about memory
	test = pd.read_csv('./datasets/test_set.csv')
	test_id = test['id'].copy()
	test_pred = pd.DataFrame({'id': test_id, 'class': result})
	test_pred['class'] = (test_pred['class']+1).astype(int)
	'''
	#new version: 2019/4/26
	test_id = range(len(result))
	test_pred = pd.DataFrame({'id': test_id, 'class': result})
	test_pred['class'] = (test_pred['class']+1).astype(int)

	return prob_cat, test_pred

def val(model, dataset, args):
	#将模型设置为验证模式
	#计算模型在验证集上的分数
	model.eval()

	acc_count = 0
	val_count = 0
	predict = np.zeros((0,),dtype=np.int32)
	ground_truth = np.zeros((0,),dtype=np.int32)

	with torch.no_grad():
		for batch in dataset:
			
			text, label = batch.text, batch.label
			if args.cuda:
				text,label = text.cuda(),label.cuda()

			outputs = model(text)
			pred = outputs.max(1)[1]

			acc_count += (pred==label).sum().item()
			val_count += label.size(0)
			predict = np.hstack((predict,pred.cpu().numpy()))
			ground_truth = np.hstack((ground_truth,label.cpu().numpy()))

	acc = acc_count/val_count
	f1score = np.mean(metrics.f1_score(predict,ground_truth,average=None))

	print('* Test Acc: {:.3f}%({}/{}), F1 Score: {}'.format(acc*100., acc_count, val_count, f1score))

	return f1score


def main(**kwargs):
	args = Config()
	args.parse(kwargs)
	if not torch.cuda.is_available():
		args.cuda = False
		args.device = None
		torch.manual_seed(args.seed)

	if args.pretrained_emd:#默认为True
		train_iter, val_iter, test_iter, args.vocab_size, vectors = data.load_data(args)
	else:
		train_iter, val_iter, test_iter, args.vocab_size, _ = data.load_data(args)
		vectors = None
	args.print_config()

	global best_score

	#------------------init model
	model = getattr(models, args.model)(args, vectors)#getattr 类似 import models.GRU 从而调用GRU()
	print(model)

	#------------------save path of model
	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)
	save_path = os.path.join(args.save_dir, '{}_{}.pth'.format(args.model, args.Id))

	if args.cuda:
	    torch.cuda.set_device(args.device)
	    torch.cuda.manual_seed(args.seed)  # set random seed for gpu
	    model.cuda()

	#-------------------cost function & optimizer
	criterion = F.cross_entropy
	if vectors is None:#如果不采用预训练词向量
		lr1=args.lr1
		lr2=lr1
	else:
		lr1,lr2 = args.lr1, args.lr2
	optimizer  = model.get_optimizer(lr1,lr2, args.weight_decay)

	for i in range(args.max_epochs):
		total_loss = 0.0
		correct = 0
		total = 0

		model.train()

		for idx, batch in enumerate(train_iter):
			# 训练模型参数
			# 使用BatchNorm层时，batch size不能为1
			if len(batch)==1:
				continue
			text, label = batch.text, batch.label
			if args.cuda:
				text, label = text.cuda(), label.cuda()

			optimizer.zero_grad()
			pred = model(text)
			loss = criterion(pred, label)
			loss.backward()
			optimizer.step()

			#更新统计指标
			total_loss += loss.item()
			predicted = pred.max(1)[1]
			total += label.size(0)
			correct += predicted.eq(label).sum().item()

			if idx % 80 ==79:
				print('[{}, {}] loss: {:.3f} | Acc: {:.3f}%({}/{})'.format(i + 1, idx + 1, total_loss / 20,
																		100. * correct / total, correct, total))
				total_loss = 0.0

		# 验证集评估，并相应调整学习率
		f1score = val(model, val_iter, args)

		if f1score>best_score:
			best_score = f1score
			checkpoint = {
				'state_dict':model.state_dict(),
				'config':args
			}
			torch.save(checkpoint, save_path)
			print('Best tmp model f1score: {}'.format(best_score))

		if f1score < best_score:
			model.load_state_dict(torch.load(save_path)['state_dict'])
			lr1 *= args.lr_decay 
			lr2 = 2e-4 if lr2 == 0 else lr2 * 0.8
			optimizer = model.get_optimizer(lr1, lr2, 0)
			print('* load previous best model: {}'.format(best_score))
			print('* model lr:{}  emb lr:{}'.format(lr1, lr2))
			if lr1 < args.min_lr:
				print('* training over, best f1 score: {}'.format(best_score))
				break

	#-------------------save final model
	args.best_score = best_score
	final_model = {
		'state_dict': model.state_dict(),
		'config': args
	}
	best_model_path = os.path.join(args.save_dir, '{}_{}_{}.pth'.format(args.model, args.text_type, best_score))
	torch.save(final_model, best_model_path)
	print('Best Final Model saved in {}'.format(best_model_path))


	#--------------------test
	if not os.path.exists('result/'):
		os.mkdir('result/')
	probs, test_pred = test(model, test_iter, args)
	result_path = 'result/' + '{}_{}_{}'.format(args.model, args.Id, args.best_score)
	np.save('{}.npy'.format(result_path), probs)
	print('Prob result {}.npy saved!'.format(result_path))

	test_pred[['id', 'class']].to_csv('{}.csv'.format(result_path), index=None)
	print('Result {}.csv saved!'.format(result_path))

	t2 = time.time()
	print('time use: {}'.format(t2 - t1))

if __name__ == '__main__':
	fire.Fire()
