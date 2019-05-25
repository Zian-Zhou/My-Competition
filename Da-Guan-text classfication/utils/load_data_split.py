import pandas as pd
from sklearn.utils import shuffle
import sys
sys.path.append("..")
from config import Config

C = Config()

#=====================Load origin data=====================
print('Loading train_set.csv ................')
train_set = pd.read_csv('../datasets/train_set.csv')
num_of_train_set = train_set.shape[0]
print('finished, {0} rows data in total'.format(num_of_train_set))

print('Loading test_set.csv ................')
test_set = pd.read_csv('../datasets/test_set.csv')
print('Finished, {0} rows data in total'.format(test_set.shape[0]))


#=====================slpit train/val data=====================
print('Split train / val data................')
train_set = shuffle(train_set,random_state=C.split_ramdom_state)
num_of_train_data = int(num_of_train_set*C.split_rate)
train_data = train_set.iloc[range(num_of_train_data)]
val_data = train_set.iloc[num_of_train_data:]
print('Split train & val finished!')

#=====================made word data=====================
print('Made word data................')
train_data[['word_seg', 'class']].to_csv('../datasets/word/train_set.csv')
val_data[['word_seg', 'class']].to_csv('../datasets/word/val_set.csv')
test_set[['id', 'word_seg']].to_csv('../datasets/word/test_set.csv')
print('Word data made!')

#=====================made article data=====================
print('Made article data................')
train_data[['article', 'class']].to_csv('../datasets/article/train_set.csv')
val_data[['article', 'class']].to_csv('../datasets/article/val_set.csv')
test_set[['id', 'article']].to_csv('../datasets/article/test_set.csv')
print('Article data made!')
