# -*- coding: utf-8 -*-

import torch
import models
import data
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn import metrics
import fire

def read_model_list(list_txt):
    file = open(list_txt)
    model_list = [line.strip() for line in file.readlines()]
    return model_list

def gen_test(list_txt, device):
    list_txt = './snapshot/' + list_txt
    model_list = read_model_list(list_txt)#['xxxx.pth','xxxxxx.pth']

    for name in model_list:
        print('*********************model: {} ****************'.format(name))
        saved_model = torch.load('./snapshot/'+name)
        args = saved_model['config']
        args.device = int(device)
        #print('Load model from {}'.format(saved_model))

        _, _, test_iter, _, _ = data.load_data(args)

        model = getattr(models, args.model)(args)
        model.load_state_dict(saved_model['state_dict'])
        torch.cuda.set_device(args.device)
        model = model.cuda()
        print(model)

        model.eval()

        probs_list = []
        result = np.zeros((0,))
        with torch.no_grad():
            print('begin prediction..........................')
            for batch in test_iter:
                text = batch.text
                text = text.cuda()
                outputs = model(text)
                probs = F.softmax(outputs, dim=1)
                probs_list.append(probs.cpu().numpy())

                pred = outputs.max(1)[1]
                result = np.hstack((result, pred.cpu().numpy()))
        print('finished')

        prob_cat = np.concatenate(probs_list, axis=0)

        result_path = 'result/' + '{}_{}_{}'.format(args.model, args.Id, args.best_score)

        np.save('{}.npy'.format(result_path), prob_cat)
        print('Prob result {}.npy saved!'.format(result_path))

        test_id = range(len(result))
        test_pred = pd.DataFrame({'id': test_id, 'class': result})
        test_pred['class'] = (test_pred['class']+1).astype(int)
        test_pred[['id', 'class']].to_csv('{}.csv'.format(result_path), index=None)
        print('Result {}.csv saved!'.format(result_path))
        ###
        '''
        result_dataframe = pd.DataFrame({'class':result})
        result_dataframe['class'] = (result_dataframe['class']+1).astype(int)
        result_dataframe.to_csv('{}.csv'.format(result_path + '_predict'), index=None)#FastText_word1_0.7407834190815604_predict
        print('Result {}_predict.csv saved!'.format(result_path))
        '''
        ###
        '''
        test = pd.read_csv('./datasets/test_set.csv')
        test_id = test['id'].copy()
        test_pred = pd.DataFrame({'id': test_id, 'class': result})
        test_pred['class'] = (test_pred['class']+1).astype(int)

        test_pred[['id', 'class']].to_csv('{}.csv'.format(result_path), index=None)
        print('Result {}.csv saved!'.format(result_path))
    	'''

if __name__ == '__main__':
    fire.Fire()

#python gen_test_result.py gen_test --list_txt='model_list.txt' --device='0'

