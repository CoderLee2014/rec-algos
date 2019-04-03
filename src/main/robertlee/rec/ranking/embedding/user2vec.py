import fasttext
import errno
import sys
from gensim.models import KeyedVectors


date = sys.argv[1]
print date

model = fasttext.skipgram('./dataset/users/data_'+ date +'.txt', './models/users/model_'+date)


words =  model.words

print(str(len(words)) + " " + str(model.dim))
i = 0
with open("./uservecs/wordvec_" + date +".txt", 'w') as writer:
	for w in words:
		v = model[w]
		vstr = ""
		for vi in v:
			vstr += " " + str(vi)
		try:
			writer.write(date + ' ' + w + vstr + "\n")
		except IOError as e:
			if e.errno == errno.EPIPE:
				pass
	writer.close()

en_model = KeyedVectors.load_word2vec_format('models/users/model_'+date+'.vec')

# Finding out similar words [default= top 10]
with open("./simvecs/users/sim_vec_"+date+".txt", 'w') as writer:
	for word in model.words:
		writer.write(word + " " + ",".join([tp[0] for tp in en_model.similar_by_word(word)]) + "\n")
	writer.close()