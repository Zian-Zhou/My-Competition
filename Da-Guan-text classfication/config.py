# -*- coding: utf-8 -*-

class Config():
	#word embedding
	pretrained_emd = True#是否使用预训练的词向量
	emb_size = 300#1. size of word embedding during training word2vec
				  #2. size of word embedding during training model, set the model size


	split_ramdom_state = 0
	split_rate = 0.9#train:val = 9:1

	max_text_len = 2000#之后可以处理成变长的

	#data.py
	embedding_path = '/data0/zhouzian/T-Z_competition/emb_build'
	data_path = './datasets/'
	text_type = 'word'
	min_freq = 5
	batch_size = 64
	device = 0# 指定GPU
	aug = 0

	#===========训练参数
	lr1 = 1e-3  # learning rate
	lr2 = 0  # embedding层的学习率（训练的时候要用一个更小的学习率微调词向量）
	min_lr = 1e-5  # 当学习率低于这个nvi值时，就退出训练
	lr_decay = 0.8  # 当一个epoch的损失开始上升时，lr ＝ lr*lr_decay,降低学习率
	decay_every = 10000  # 每多少个batch  查看val acc，并修改学习率
	weight_decay = 0  # 2e-5 # 权重衰减
	max_epochs = 50
	cuda = True

	#=========main.py: for training\validating\testing
	seed = 666
	model = 'GRU' # 使用的模型，名字必须与models/__init__.py中的名字一致 可以在命令行执行python文件的时候指定模型
	Id = 'default'# 模型存储命名，指定存储id名称以区分
	save_dir = 'snapshot/' #模型存储位置
	best_score = 0

	#=========training setting:
	lr1 = 1e-3
	lr2 = 0
	weight_decay = 0  # 2e-5 # 权重衰减
	lr_decay = 0.8  # 当一个epoch的损失开始上升时，lr ＝ lr*lr_decay 



	#=========model
	vocab_size = 10000#没有意义，预处理阶段就确定下来了——由数据集决定
	label_size = 19#类别数

	#GRU
	kmax_pooling = 2
	embedding_dim = emb_size#during training model
	hidden_dim = 256#隐藏层维度
	lstm_layers = 1#隐藏层层数（GRU个数）
	lstm_dropout = 0.5#只有当GRU个数大于1的时候，dropout才有作用
	kmax_pooling = 2
	bidirectional = True#双向
	linear_hidden_size = 100#全连接层维度（连接到分类层）

	#TextCNN
	kernel_num = 200  # number of each kind of kernel
	kernel_sizes = [1,2,3,4,5]  # kernel size to use for convolution   ##########2019-4-27
	dropout_rate = 0.5  # the probability for dropout

	# RCNN / RCNN1
	rcnn_kernel = 512

	def parse(self, kwargs):
		'''
		根据字典kwargs 更新 config参数
		'''
		# 更新配置参数
		for k, v in kwargs.items():
			if not hasattr(self, k):
				raise Exception("Warning: config has not attribute <%s>" % k)
			setattr(self, k, v)

	def print_config(self):
		# 打印配置信息
		print('user config:')
		for k, v in self.__class__.__dict__.items():
			if not k.startswith('__') and k != 'parse' and k != 'print_config':
				print('    {} : {}'.format(k, getattr(self, k)))
