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
	tweetpuncts='!"$%&\'()*+,-./:;<=>?[\]^_`{|}~¡¿'
	df['text']=df['text'].apply(lambda x: re.sub('['+tweetpuncts+']', '', x))
	#clean URLs from tweets

	df['text']=df['text'].apply(lambda x: re.sub(r'http[^\s]+', '', x))
	#clean emoji from tweets
	df['text']=df['text'].apply(lambda x: re.sub(r'\\ud', 'EMOJIud', x))
	df['text']=df['text'].apply(lambda x: re.sub(r'EMOJI\w+', '', x))
	return df

def removeautomation(dataframe, tokenizer):
	df=dataframe
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
def subtweets(dataframe, n):
	df = dataframe
	ngramdf=pd.DataFrame()
	hapaxgrams=[]
	ngramdict={}
	#find most frequent ngrams as defined in the function with n
	for row in df['text']:
		for gram in list(ngrams(row, n)):
			if gram not in ngramdict:
				ngramdict[gram] = 1
			else:
				ngramdict[gram] +=1
	fdist=nltk.FreqDist(ngramdict)
	#create a dataframe of ngrams and their counts
	ngramdf=pd.DataFrame.from_dict(fdist, orient='index').reset_index()
	ngramdf = ngramdf.rename(columns={'index':'ngram', 0:'count'})
	#create a dataframe of ngrams only appearing one time in the corpus
	hapaxdf = ngramdf[ngramdf['count'] ==1]
	hapaxset=set(list(hapaxdf['ngram']))
	#find tweets that do not contain an @mention and that are not in reply to another user 
	#that also contain a unique ngram
	df=df[df['text'].apply(lambda x: '@' not in x)]
	df=df[df['in_reply_to_user_id_str'].apply(lambda x: math.isnan(x))]
	df=df[df['text'].apply(lambda x: bool(set(list(ngrams(x, n))) & hapaxset))]	
	df['text']=df['text'].apply(lambda x: ' '.join(x))
	return df

#run the code and save spreadsheet to directory
if __name__ == '__main__':
	tokenizer = nltk.TweetTokenizer(preserve_case=False)
	filename='totaltweetsallmeta.csv'
	df = pd.read_csv(filename, encoding='latin-1')
	cleandf = cleantweets(df)
	autolessdf=removeautomation(cleandf, tokenizer)
	finaldf = subtweets(autolessdf, 4)
	finaldf.to_csv('subtweets.csv')