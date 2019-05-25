import word2vec
import sys
sys.path.append("..")
from config import Config

C = Config()

#files = ['word.txt','article.txt']
#sizes = [300,300]

files = ['word.txt','article.txt','word.txt']
sizes = [128,128,256]


def main():
	for size,file in zip(sizes,files):
		emb_file = file.split('.')[0] + '_' + str(size) +'.bin'
		word2vec.word2vec(file, emb_file, min_count=5, size=size, verbose=True)
		embedding_transform(emb_file)

def embedding_transform(emb_file):
	model = word2vec.load(emb_file)
	vocab, vectors = model.vocab, model.vectors
	print(emb_file)
	print('setting size of word embedding: {0}'.format(vectors.shape))

	new_file = emb_file.split('.')[0]+'_.txt'
	print('Transforming.....')

	with open(new_file, 'w') as f:
		for word,vec in zip(vocab, vectors):
			f.write(str(word)+' '+' '.join(map(str, vec))+'\n')
	print('fransform finished.')

if __name__ == '__main__':
	main()
