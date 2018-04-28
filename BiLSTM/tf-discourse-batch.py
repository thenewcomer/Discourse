#!/usr/bin/python

import os,sys,gzip,argparse,random,subprocess,time
from collections import *
from itertools import *
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='Train discourse model 1>discarded 2>progress')
parser.add_argument('trainfile',help='train data file')
parser.add_argument('testfile',help='test data, format is the same with train data')
parser.add_argument('output_model_prefix',help='train model file and log file prefix')
parser.add_argument('-showvariable',help='print variable to <output_model_prefix>.var',default=None,type=str)
parser.add_argument('-savevariable',help='print variable to <output_model_prefix>.var',default=None,type=str)
parser.add_argument('-floatx',help='float32 or float64',default='float32',type=str)
parser.add_argument('-epoches',help='maximum number of training epoches',default=200,type=int)
parser.add_argument('-L2',help='L2 regularization weight',default=0,type=float)
parser.add_argument('-all_dropout',help='dropout probability',default=0,type=float)
parser.add_argument('-dropout',help='dropout keep probability',default=1.0,type=float)
parser.add_argument('-embedding_dropout',help='keep probability',default=0.9,type=float)
parser.add_argument('-lstm_dropout',help='dropout after LSTM',default=0.5,type=float)
parser.add_argument('-pooling_dropout',help='dropout probability',default=0.25,type=float)
parser.add_argument('-final_dropout',help='dropout probability',default=0.9,type=float)
parser.add_argument('-word2vec',help='the 1st word must be <unk>',default=None,type=str)
parser.add_argument('-learn_rate',help='initial learning rate',default=0.1,type=float)
parser.add_argument('-embeddingDim',help='dropout probability',default=50,type=int)
parser.add_argument('-hiddenSize',help='hidden layer size',default=50,type=int)
parser.add_argument('-embedding_learn_rate',help='setting the learning rate for embedding will set embeddingTrainable to True',default=0.0,type=float)
parser.add_argument('-normalize_embedding',help='normalize initial embedding',default=False,action='store_true')
parser.add_argument('-mask_zero',help='LSTM use zero mask',default=False,action='store_true')
parser.add_argument('-init_model',help='load model',default=None)
parser.add_argument('-padding',help='padding: pre/post',default='pre')
parser.add_argument('-truncating',help='truncating: pre/post',default='post')
parser.add_argument('-mode',help='model: att/gru',default='att')
parser.add_argument('-rslices',help='number of slice',default=2,type=int)
parser.add_argument('-batch_size',help='batch size',default=32,type=int)
parser.add_argument('-maxSentenceLength',help='maxSentenceLength',default=51,type=int)
parser.add_argument('-activation',help='activation function',default='sigmoid',type=str)
parser.add_argument('-classify_mode',help='binary will use Dense(1) followed by a sigmoid activation; 2-way will use Dense(2) followed by categorical cross-entropy; default is N-way',default=['N-way'],type=str,nargs='+')
parser.add_argument('-optimizer',help='Adagrad / Adam / Momentum / GradientDescent / RMSProp / Adadelta',default='Adagrad',type=str)
parser.add_argument('-optimizer_options',help='additional optimizer options',default='',type=str)
parser.add_argument('-sampling',help='+ve: up-sample; -ve: down-sample; value: ratio w.r.t. reference',default=[0.0],type=float,nargs='+')
parser.add_argument('-limit_vocab',help='limit vocab size',default=None,type=int)
parser.add_argument('-pool_size',help='pooling size',default=3,type=int)
parser.add_argument('-tfidf_on_attention',help='apply TFIDF on attention weights',default=False,action='store_true')
parser.add_argument('-tfidf_on_embedding',help='apply TFIDF on word embedding',default=False,action='store_true')
parser.add_argument('-tensorboard',help='launch tensorboard',default=False,action='store_true')
parser.add_argument('-testmode',help='test a model on testfile and report the score',default=False,action='store_true')
parser.add_argument('-saveeveryepoch',help='save after every epoch',default=False,action='store_true')
parser.add_argument('-use_feature',help='specify additional sparse or dense features in column N (starts from 0), e.g., -use_feature s4 _use_feature d5',default=[],action='append')
parser.add_argument('-print_heat_map',help='print heat map',default=None,type=str)
parser.add_argument('-objective',help='objective function, cross_entropy / F1',default='cross_entropy',type=str)
parser.add_argument('-init_scale',help='initialization random std',default=1.0,type=float)
parser.add_argument('-seed',help='random seed',default=1337,type=int)
opt=parser.parse_args()
globals().update(vars(opt))

random.seed(seed)
np.random.seed(seed)  # for reproducibility
tf.set_random_seed(seed)
exec 'floatX=tf.'+floatx
embeddingTrainable = (embedding_learn_rate!=0)

if all_dropout!=0:
	dropout=embedding_dropout=lstm_dropout=pooling_dropout=final_dropout=all_dropout

use_feature=[f.lower() for f in use_feature]

log_file=open(output_model_prefix+'.log','w')          
def LOG(*args):                                           
	print >>sys.stderr, ' '.join([str(_) for _ in args])  
	if showvariable==None:
		print >>log_file, ' '.join([str(_) for _ in args])
		log_file.flush()                                  

def score(ref,sys,neutralset=set()):
	def div(a,b):
		return float(a)/b if b!=0 else 0
	N_pres=Counter(ref)
	N_pred=Counter(sys)
	N_corr=Counter([sys[i] for i in xrange(len(sys)) if ref[i]==sys[i]])
	labs=sorted(N_pres.keys())+['Total']
	for lab in N_pres.keys():
		if lab not in neutralset:
			N_corr['Total']+=N_corr[lab]
			N_pres['Total']+=N_pres[lab]
			N_pred['Total']+=N_pred[lab]
	prt=[[lab, N_corr[lab], N_pres[lab], N_pred[lab], div(N_corr[lab],N_pred[lab]), div(N_corr[lab],N_pres[lab]), div(N_corr[lab]*2.0, N_pred[lab]+N_pres[lab])] for lab in labs]
	out=[['%-12s'%'Label']+[('%12s'%lab) for lab in ['N_correct','N_present','N_predict','Precision','Recall','F1']]]
	for row in prt:
		its=[('%-12s'%row[0]) if len(row[0])<=12 else row[0]]+[('%12s'%w) for w in row[1:4]]+['%12s'%('%.4f'%w) for w in row[4:]]
		if len(its[0])>12:
			ex=12-len(its[1].strip())
			its[0]=its[0][0:12+ex]
			its[1]=its[1][ex:]
		out+=[its]
	out+=[['Average F1: ',str(np.mean([div(N_corr[lab] * 2.0, N_pred[lab] + N_pres[lab]) for lab in N_pres.keys()]))]]
	return '\n'.join([''.join(row) for row in out]), div(N_corr['Total']*2.0, N_pred['Total']+N_pres['Total'])

def tfidf(D):
	df=Counter([w for sent in D for w in set(sent)])
	Den=np.log(len(D))
	return Counter({w:np.log(len(D)/(df[w]+1.0))/Den for w,f in df.iteritems()})

# Load wordvec
if word2vec!=None:
	LOG('Loading wordvec ...')
	wordvec=[L.split() for L in open(word2vec)]
	embeddingDim=len(wordvec[0])-1
	int2word=['<PADDING>']+[_[0] for _ in wordvec]
	word2int=defaultdict(lambda:1, {w:i+1 for i,w in enumerate([its[0] for its in wordvec])})
	WORDVEC=np.array([[0.0]*embeddingDim]+[map(float,its[1:]) for its in wordvec])
	if normalize_embedding:
		WORDVEC = WORDVEC / (np.linalg.norm(WORDVEC, axis=1, keepdims=True)+1.0e-8)
	del wordvec

def load_file(fn):
	data=[[its.strip() for its in L.split('|||')] for L in open(fn)]
	arg1=[its[0].split() for its in data]
	arg2=[its[1].split() for its in data]
	denses,sparses=[[] for its in data],[[] for its in data]
	for f in use_feature:
		col=int(f[1:])
		if f[0]=='d':
			denses=[map(float,its[col].split()) for its in data]
		elif f[0]=='s':
			sparses=[its[col] for its in data]
		else:
			raise 'Illegal feature type '+f
	labels=[its[2] for its in data]
	vocab_set=Counter([w for its in arg1+arg2 for w in its])
	label_set=set(labels)
	return arg1,arg2,labels,denses,sparses,vocab_set,label_set

def reweight_sample(arg1,arg2,labels,denseF,sparseF,sampling):
	assert len(sampling)==len(set(labels))
	cls_idxs = [[] for i in range(len(sampling))]
	for ii, jj in enumerate(labels):
		cls_idxs[jj] += [ii]
	for i in range(len(sampling)):
		random.shuffle(cls_idxs[i])
	orig_cls_cnts = [len(idx) for ii, idx in enumerate(cls_idxs)]
	cls_cnts = [int(len(idx)*sampling[ii]+0.5) for ii,idx in enumerate(cls_idxs)]
	new_idx = [cls_idxs[i][idx%orig_cls_cnts[i]] for i,cls_cnt in enumerate(cls_cnts) for idx in range(cls_cnt)]
	random.shuffle(new_idx)
	return [arg1[_] for _ in new_idx], [arg2[_] for _ in new_idx], [labels[_] for _ in new_idx], (None if denseF==None else [denseF[_] for _ in new_idx]), (None if sparseF==None else [sparseF[_] for _ in new_idx]), new_idx

def resample(arg1,arg2,labels,denseF,sparseF,sampling):
	sampling=sampling[0]
	if sampling==0:
		return arg1,arg2,labels,denseF,sparseF,range(len(arg1))
	def makeupto(idx,N):
		sz=len(idx)
		return [idx[i%sz] for i in range(N)]
	cls_idx=[ii for ii,jj in enumerate(labels) if jj==1]
	noncls_idx=[ii for ii,jj in enumerate(labels) if jj==0]
	major_idx,minor_idx=((cls_idx,noncls_idx) if len(cls_idx)>len(noncls_idx) else (noncls_idx,cls_idx))
	if sampling>0:
		random.shuffle(minor_idx)
		minor_idx=[minor_idx[_%len(minor_idx)] for _ in xrange(int(abs(sampling)*len(major_idx)))]
	elif sampling<0:
		random.shuffle(major_idx)
		major_idx=[major_idx[_] for _ in xrange(int(abs(sampling)*len(minor_idx)))]
	new_idx=major_idx+minor_idx
	random.shuffle(new_idx)
	return [arg1[_] for _ in new_idx], [arg2[_] for _ in new_idx], [labels[_] for _ in new_idx], (None if denseF==None else [denseF[_] for _ in new_idx]), (None if sparseF==None else [sparseF[_] for _ in new_idx]), new_idx

def resampleNway(arg1,arg2,labels,denseF,sparseF,sampling):
	if len(sampling)>1:
		return reweight_sample(arg1,arg2,labels,denseF,sparseF,sampling)
	else:
		sampling=sampling[0]
	if sampling==0:
		return arg1,arg2,labels,denseF,sparseF,range(len(arg1))
	def makeupto(idx,N):
		sz=len(idx)
		return [idx[i%sz] for i in range(N)]
	cls_idxs = [[] for i in set(labels)]
	for ii, jj in enumerate(labels):
		cls_idxs[jj] += [ii]
	cls_cnt = [len(idx) for idx in cls_idxs]
	major_cnt, minor_cnt = max(cls_cnt), min(cls_cnt)
	if sampling>0:
		for cls in set(labels):
			if cls_cnt[cls]!=major_cnt:
				random.shuffle(cls_idxs[cls])
				cls_idxs[cls]=[cls_idxs[cls][_%len(cls_idxs[cls])] for _ in xrange(int(abs(sampling)*major_cnt))]
	elif sampling<0:
		for cls in set(labels):
			if cls_cnt[cls]!=minor_cnt:
				random.shuffle(cls_idxs[cls])
				cls_idxs[cls]=[cls_idxs[cls][_%len(cls_idxs[cls])] for _ in xrange(int(abs(sampling)*minor_cnt))]
	new_idx = [idx for idxs in cls_idxs for idx in idxs]
	random.shuffle(new_idx)
	return [arg1[_] for _ in new_idx], [arg2[_] for _ in new_idx], [labels[_] for _ in new_idx], (None if denseF==None else [denseF[_] for _ in new_idx]), (None if sparseF==None else [sparseF[_] for _ in new_idx]), new_idx

def biMap(dct,*args):
	return [[[dct[i] for i in l] for l in arg] for arg in args]
def monoMap(dct,*args):
	return [[dct(l) for l in arg] for arg in args]

LOG('Running parameters:', vars(opt))
LOG('Loading training and testing data ...')
trainingSentenceOne,trainingSentenceTwo,trainingLabel,train_denseF,train_sparseF,train_vocab,train_label=load_file(trainfile)
testingSentenceOne,testingSentenceTwo,testingLabel,test_denseF,test_sparseF,test_vocab,test_label=load_file(testfile)
denseDim=len(train_denseF[0])
if denseDim>0:
	LOG('Dense feature has dimension',denseDim)
labels=sorted(list(train_label|test_label))
if word2vec==None:
	int2word=["<PADDING>","<UNK>"]+list(train_vocab)
	word2int=defaultdict(lambda:1, {w:i for i,w in enumerate(["<PADDING>","<UNK>"]+list(train_vocab))})
if limit_vocab!=None:
	word_freq=sorted([[k,v] for k,v in train_vocab.iteritems()], key=lambda t:t[1], reverse=True)
	vocab=set([_[0] for _ in word_freq[0:limit_vocab]])
	word2int=defaultdict(lambda:1, {_:(word2int[_] if _ in vocab else 1) for _ in word2int})
vocab=word2int.copy()
vocab['']=0	#padding
trainingSentenceOne,trainingSentenceTwo,testingSentenceOne,testingSentenceTwo=biMap(word2int,trainingSentenceOne,trainingSentenceTwo,testingSentenceOne,testingSentenceTwo)
trainingLabel=[labels.index(w) for w in trainingLabel]
testingLabel=[labels.index(w) for w in testingLabel]
maxSentenceLength=max([len(its) for its in trainingSentenceOne+trainingSentenceTwo]) if maxSentenceLength==0 else maxSentenceLength

TFIDF=tfidf(trainingSentenceOne+trainingSentenceTwo+testingSentenceOne+testingSentenceTwo)

# convert binary label and do up-down-sampling
if classify_mode[0]!='N-way':
	Ibinary_classify=labels.index(classify_mode[1])
	trainingLabel=[(1 if _==Ibinary_classify else 0) for _ in trainingLabel]
	testingLabel=[(1 if _==Ibinary_classify else 0) for _ in testingLabel]
	trainingSentenceOne,trainingSentenceTwo,trainingLabel,train_denseF,train_sparseF,train_idx = \
		resample(trainingSentenceOne,trainingSentenceTwo,trainingLabel,train_denseF,train_sparseF,sampling)
else:
	trainingSentenceOne, trainingSentenceTwo, trainingLabel, train_denseF, train_sparseF, train_idx = \
		resampleNway(trainingSentenceOne, trainingSentenceTwo, trainingLabel, train_denseF, train_sparseF, sampling)

trainingSentenceOneTFIDF,trainingSentenceTwoTFIDF,testingSentenceOneTFIDF,testingSentenceTwoTFIDF=biMap(TFIDF,trainingSentenceOne,trainingSentenceTwo,testingSentenceOne,testingSentenceTwo)

# padding will return sequence lengths
def pad(seq, maxlen, pad_dir, truncate_dir):
	padded_seq=np.array([(_+[0]*maxlen)[0:maxlen] for _ in seq])
	length=np.array([len(_) for _ in seq])
	return padded_seq, length
	return (np.array([(_+[0]*maxlen)[0:maxlen] for _ in seq]), np.array([len(_) for _ in seq]))
train_x_one,train_x_two,test_x_one,test_x_two=[pad(seq, maxSentenceLength, padding, truncating) for seq in [trainingSentenceOne,trainingSentenceTwo,testingSentenceOne,testingSentenceTwo]]
train_x_one_tfidf,train_x_two_tfidf,test_x_one_tfidf,test_x_two_tfidf=[pad(seq, maxSentenceLength, padding, truncating) for seq in [trainingSentenceOneTFIDF,trainingSentenceTwoTFIDF,testingSentenceOneTFIDF,testingSentenceOneTFIDF]]
train_x_dense,test_x_dense=np.array(train_denseF),np.array(test_denseF)

n_classes=1 if classify_mode[0]=='binary' else (2 if classify_mode[0]=='2-way' else len(train_label))

class BatchGenerator(object):
	def __init__(self, Xs, Y, shuffle=False):
		self.Xs,self.Y,self.batch_id,self.shuffle = Xs,Y,0,shuffle
	def reset(self):
		self.batch_id=0
	def hasNext(self):
		return self.batch_id<len(self.Y)
	def next(self, batch_size):
		if self.batch_id == len(self.Y):
			self.batch_id = 0
		if self.batch_id==0 and self.shuffle:
			ids=range(len(self.Y))
			np.random.shuffle(ids)
			self.Xs,self.Y = [[X[i] for i in ids] for X in self.Xs], [self.Y[i] for i in ids]
		last_id,self.batch_id = self.batch_id, min(self.batch_id + batch_size, len(self.Y))
		return [X[last_id:self.batch_id] for X in self.Xs], self.Y[last_id:self.batch_id]

trainset = BatchGenerator([train_x_one[0],train_x_one[1],train_x_two[0],train_x_two[1],train_idx],trainingLabel, shuffle=True)
testset = BatchGenerator([test_x_one[0],test_x_one[1],test_x_two[0],test_x_two[1]],testingLabel, shuffle=False)


LOG('Compiling functions ...')
execfile(mode)

# cost and evaluation
with tf.name_scope('Trainer'):
	if classify_mode[0]=='binary':
		posterior1=tf.squeeze(posterior)
		cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(posterior1, y))
		pred = tf.round(tf.nn.sigmoid(posterior1,'output_sigmoid'))
	else:
		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(posterior, tf.cast(y,tf.int32)))
		pred = tf.argmax(tf.nn.softmax(posterior),1)
	_optimizer = eval('tf.train.'+optimizer+'Optimizer')
	kwargs={_.split('=')[0]:float(_.split('=')[1]) for _ in optimizer_options.split()}
	if embeddingTrainable:
		train_step = _optimizer(learning_rate=learn_rate, **kwargs).minimize(cost, var_list=[v for v in tf.trainable_variables() if v!=Emb])
		train_step2 = _optimizer(learning_rate=embedding_learn_rate, **kwargs).minimize(cost, var_list=[Emb])
	else:
		train_step = _optimizer(learning_rate=learn_rate, **kwargs).minimize(cost)
		train_step2 = tf.no_op()

tf.scalar_summary('cost', cost)
merged_summaries = tf.merge_all_summaries()

def print_dbg(obj):
	np.set_printoptions(edgeitems=100,linewidth=1000000)
	print >>open(output_model_prefix+'.dbg','w'), obj
	print >>sys.stderr, train_x_one.shape
	sys.exit(0)

def convertLabel(predict, testingLabel):
	if classify_mode[0]!='N-way':
		labels2=['False',classify_mode[1]]
	predictLabels,correctLabels=[],[]
	for i in range(len(predict)):
		if classify_mode[0]=='N-way':
			predictLabels += [labels[predict[i]]]
			correctLabels += [labels[testingLabel[i]]]
		elif classify_mode[0]=='2-way':
			predictLabels += [labels2[predict[i]]]
			correctLabels += [labels2[testingLabel[i]]]
		else:
			predictIndex = int(predict[i]+0.5)
			predictLabels += [labels2[predictIndex]]
			correctLabels += [labels2[testingLabel[i]]]
	return predictLabels,correctLabels

def duplicate_for_context(x):
	return np.pad(x,((0,0),(Ncontext,Ncontext)),'constant',constant_values=0) if 'Ncontext' in globals() else x

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
	sess.run(tf.initialize_all_variables())

	#usage: saver.save/restore(session,fn)
	saver=tf.train.Saver(max_to_keep=0)

	if init_model!=None:
		LOG('Loading initial model',init_model,'...')
		saver.restore(sess,init_model)

	if showvariable:
		out = eval('sess.run('+showvariable+')')
		np.set_printoptions(edgeitems=np.inf,linewidth=np.inf)
		print >>open(output_model_prefix+'.var','w'), out
		try:
			print >>sys.stderr, out.shape
		except:
			pass
		sys.exit(0)
	
	def get_heat_map(word_ids,alpha):
		arr=[[int2word[i],alpha[i]] for i,wid in enumerate(word_ids) if wid!=0]
		return ' '.join([('%10s'%p[0]) for p in arr])+'\n'+' '.join(['%.6f'%p[1] for p in arr])+'\n'
	if print_heat_map:
		fp=open(print_heat_map,'w')
		exec 'model = Model(input=[sentenceOneInput,sentenceTwoInput,sentenceOneMask,sentenceTwoMask,sentenceOneTFIDF,sentenceTwoTFIDF],output=[a1,a2])'
		hm1,hm2 = model.predict([test_x_one,test_x_two,test_x_one_mask,test_x_two_mask,test_x_one_tfidf,test_x_two_tfidf,test_x_dense],batch_size=batch_size,verbose=1)
		for i in xrange(len(test_x_one)):
			print >>fp, get_heat_map(test_x_one[i],hm1[i])+get_heat_map(test_x_two[i],hm2[i])
		sys.exit(0)
	
	if testmode:
		predict = model.predict([test_x_one,test_x_two,test_x_one_mask,test_x_two_mask,test_x_one_tfidf,test_x_two_tfidf,test_x_dense],batch_size=batch_size,verbose=1)
		predictLabels,correctLabels = convertLabel(predict, testingLabel)
		rep,res = score(correctLabels, predictLabels, 'False')
		LOG(rep)
		sys.exit(0)
	
	# main training loop
	best_res=-99999
	if tensorboard:
		summary_path = 'summary_' + str(os.getpid())
		train_writer = tf.train.SummaryWriter(summary_path, sess.graph)
		LOG('Summaries are stored in', summary_path)
		subprocess.Popen('tensorboard --reload_interval 10 --logdir='+summary_path, shell=True)

	LOG('Total number of trainable parameters = ',np.sum([tf.reshape(v, [-1])._shape_as_list()[0] for v in tf.trainable_variables()]))

	if savevariable:
		trainset.reset()
		all_idx, all_var = [], []
		while trainset.hasNext():
			print >> sys.stderr, 'save-variable:',trainset.batch_id, "/", len(trainset.Y), "\r",
			[batch_x1Ori, batch_seqlen1Ori, batch_x2Ori, batch_seqlen2Ori, batch_idxOri], batch_labelsOri = trainset.next(batch_size)
			exec 'variable = sess.run(' + savevariable + ',' + 'feed_dict={x1: duplicate_for_context(batch_x1Ori), x2: duplicate_for_context(batch_x2Ori), y: batch_labelsOri,' + \
			     'seqlen1: batch_seqlen1Ori, seqlen2: batch_seqlen2Ori, current_batch_size: len(batch_labelsOri), ' + \
				 '_embedding_dropout: 1.0, _final_dropout: 1.0, _dropout: 1.0})'
			all_idx += batch_idxOri
			all_var += list(variable)
		NN=max(all_idx)
		assert len(all_idx)==len(all_var)
		_M = {idx:all_var[i] for i,idx in enumerate(all_idx)}
		np.set_printoptions(edgeitems=np.inf, linewidth=np.inf)
		#print >> open(output_model_prefix +'.var', 'w'), [_M[i] for i in range(100)+range(NN-100,NN)]
		print >> open(output_model_prefix +'.var', 'w'), [list(_M[i]) for i in xrange(NN+1)]
		sys.exit(0)

	for e in range(epoches):
		LOG('Training Epoch:', e)
		trainset.reset()
		total_train_cost = total_test_cost = 0
		time_start = time.time()
		while trainset.hasNext():
			print >>sys.stderr,trainset.batch_id,"/",len(trainset.Y),"\r",
			[batch_x1, batch_seqlen1, batch_x2, batch_seqlen2, batch_idx], batch_labels = trainset.next(batch_size)
			_,_,cost1,summary1 = sess.run([train_step,train_step2,cost,merged_summaries], feed_dict={
				x1: duplicate_for_context(batch_x1), x2: duplicate_for_context(batch_x2), y: batch_labels, seqlen1: batch_seqlen1, seqlen2: batch_seqlen2, current_batch_size: len(batch_labels),
				_embedding_dropout:embedding_dropout, _final_dropout:final_dropout, _dropout:dropout })
			total_train_cost += cost1
			if tensorboard: train_writer.add_summary(summary1, e)
		LOG('Training took (in seconds)', time.time()-time_start)
	
		LOG('Testing Epoch: ', e)
		predict = []
		testset.reset()
		time_start = time.time()
		while testset.hasNext():
			print >>sys.stderr, testset.batch_id,"/",len(testset.Y),"\r",
			[batch_x1, batch_seqlen1, batch_x2, batch_seqlen2], batch_labels = testset.next(batch_size)
			pred1,cost1 = sess.run([pred,cost], feed_dict={
				x1: duplicate_for_context(batch_x1), x2: duplicate_for_context(batch_x2), y: batch_labels, seqlen1: batch_seqlen1, seqlen2: batch_seqlen2, current_batch_size: len(batch_labels),
				_embedding_dropout: 1.0, _final_dropout: 1.0, _dropout: 1.0 })
			predict += list(pred1)
			total_test_cost += cost1
		LOG('Testing took (in seconds)', time.time() - time_start)

		LOG('Training_cost=',total_train_cost,'; testing_cost=',total_test_cost)

		predictLabels,correctLabels = convertLabel(predict, testingLabel)
	
		#print_dbg(predictLabels)
		rep,res = score(correctLabels, predictLabels, 'False')
		LOG(rep)
		if saveeveryepoch:
			saver.save(sess, output_model_prefix+'.'+str(e)+'.ckpt')
		if res > best_res:
			saver.save(sess, output_model_prefix+'.ckpt')
			best_res = res
			if savevariable:
				trainset.reset()
				all_idx, all_var = [], []
				while trainset.hasNext():
					print >> sys.stderr, 'save-variable:',trainset.batch_id, "/", len(trainset.Y), "\r",
					[batch_x1Ori, batch_seqlen1Ori, batch_x2Ori, batch_seqlen2Ori, batch_idxOri], batch_labelsOri = trainset.next(batch_size)
					exec 'variable = sess.run(' + savevariable + ',' + 'feed_dict={x1: duplicate_for_context(batch_x1Ori), x2: duplicate_for_context(batch_x2Ori), y: batch_labelsOri,' + \
					     'seqlen1: batch_seqlen1Ori, seqlen2: batch_seqlen2Ori, current_batch_size: len(batch_labelsOri), ' + \
						 '_embedding_dropout: 1.0, _final_dropout: 1.0, _dropout: 1.0})'
					all_idx += batch_idxOri
					all_var += list(variable)
				assert len(all_idx)==len(all_var)
				_M = {idx:all_var[i] for i,idx in enumerate(all_idx)}
				np.set_printoptions(edgeitems=200, linewidth=1000000)
				print >> open(output_model_prefix +'.'+ savevariable + '.'+str(e)+'.var', 'w'), np.array([_M[i] for i in xrange(len(_M))])
				print >> sys.stderr, 'save-variable:', variable.shape

print 'exit ok'

