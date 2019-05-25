#将训练集、测试集所有数据导入，并导出为txt文件用于训练词向量、字向量
import pandas as pd

print('Loading datasets.......')
train_data = pd.read_csv('../datasets/train_set.csv')
test_data = pd.read_csv('../datasets/test_set.csv')

print('{} lines in train datasets'.format(len(train_data)))
print('{} lines in test datasets'.format(len(test_data)))

print('making word.txt......')
with open('./word.txt', 'w') as f:
    f.writelines([text + '\n' for text in train_data['word_seg']])
    f.writelines([text + '\n' for text in test_data['word_seg']])

print('making article.txt......')
with open('./article.txt', 'w') as f:
    f.writelines([text + '\n' for text in train_data['article']])
    f.writelines([text + '\n' for text in test_data['article']])

    