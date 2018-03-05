import pandas as pd
import nltk
from nltk.util import ngrams
import os
import string
import re
import math

def cleantweets(dataframe):
	df = pd.DataFrame.copy(dataframe)
	#clean punctuation from tweets
	tweetpuncts='!-"$%&\'()*+,./:;<=>?[\]^_`{|}~¡¿'
	#keep linebreaks as 'linebreak'
	df['text']=df['text'].apply(lambda x: re.sub('\r\r\n', ' – ', x))
	df['text']=df['text'].apply(lambda x: re.sub('['+tweetpuncts+']', '', x))
	#clean URLs from tweets
	df['text']=df['text'].apply(lambda x: re.sub(r'http[^\s]+', '', x))
	#clean emoji from tweets
	df['text']=df['text'].apply(lambda x: re.sub(r'\\ud', 'EMOJIud', x))
	df['text']=df['text'].apply(lambda x: re.sub(r'EMOJI\w+', '', x))
	return df

def removeautomation(dataframe, tokenizer):
	df=pd.DataFrame.copy(dataframe)
	#tokenize dataframe
	df['text']=df['text'].apply(lambda x:tokenizer.tokenize(x))
	autodict={}
	totaltokens=0
	totaltweets=0
	autolesstokens=0
	autolesstweets=0
	#calculate statistics about corpus before automated material removed
	for row in df['text']:
		totaltokens+=len(row)
		totaltweets+=1

	avgtokenlen=int(totaltokens/totaltweets)
	print('Total Tokens in Corpus:', totaltokens, 'Tokens')
	print('Average Tweet Length:', avgtokenlen, 'Tokens')

	#create a dictionary of ngrams 2 tokens less than of the average full length of a tweet to find automated material
	for row in df['text']:
		for gram in list(ngrams(row, avgtokenlen-2)):
			if gram not in autodict:
				autodict[gram] = 1
			else:
				autodict[gram] +=1
	#find most frequent large-scale ngrams and remove from corpus
	fdist = nltk.FreqDist(autodict)
	for gram, number in fdist.most_common(50):
		df = df[[set(([gram])).issubset(set(list(ngrams(row, avgtokenlen-2)))) == False  for row in df['text']]]
	for row in df['text']:
		autolesstokens+=len(row)
		autolesstweets+=1
	#show statistics after automation is cleaned
	avgautolesstokenlen=int(totaltokens/totaltweets)
	print('Total Tokens in Corpus Automated Tweets Removed:', autolesstokens, 'Tokens')
	print('Average Tweet Length Automated Tweets Removed:', avgautolesstokenlen, 'Tokens')
	return df


def metaling(dataframe):
	df= pd.DataFrame.copy(dataframe)
	reportedspeechdf=pd.DataFrame()
	#find tweets containing words tuteo, voseo, ustedeo, and variants
	reportedspeechdf = reportedspeechdf.append(df[df['text'].apply(lambda x: bool(re.search(r'(tus?te[oa]+|usted?e[oa]+|vost?ed?e?o+|hablar de vos|hablar de usted|hablar de tu)',' '.join(x))))])
	reportedspeechdf['text']=reportedspeechdf['text'].apply(lambda x: ' '.join(x))
	return reportedspeechdf

#run the code and save spreadsheet to directory
if __name__ == '__main__':
	tokenizer = nltk.TweetTokenizer(preserve_case=False)
	filename='totaltweetsallmeta.csv'
	df = pd.read_csv(filename, encoding='latin-1')
	cleandf = cleantweets(df)

	autolessdf=removeautomation(cleandf, tokenizer)

	reportedspeechdf = metaling(autolessdf)
	reportedspeechdf.to_csv('metaling.csv')