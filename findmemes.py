import pandas as pd
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import os
import string
import re


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
def ngramsfreq(dataframe, n):
	df = dataframe
	df3 = pd.DataFrame()
	ngramdict={}
	#find most frequent ngrams as defined in the function with n
	for row in df['text']:
		for gram in list(ngrams(row, n)):
			if gram not in ngramdict:
				ngramdict[gram] = 1
			else:
				ngramdict[gram] +=1
	fdist=nltk.FreqDist(ngramdict)
	#using the most common 100 ngrams, append tweets containing those common ngrams to a dataframe with the ngram as its label
	for gram, number in fdist.most_common(100):
		df2=pd.DataFrame()
		df2 = df2.append(df[[set([gram]).issubset(set(list(ngrams(row,n)))) == True for row in df['text']]])
		df2 = df2.assign(ngram=str(gram))
		df3 = df3.append(df2)
	df3['text']=df3['text'].apply(lambda x: ' '.join(x))
	df3= df3.drop_duplicates(subset='text')
	return df3

#run the code and save spreadsheet
if __name__ == '__main__':
	tokenizer = nltk.TweetTokenizer(preserve_case=False)
	filename='totaltweetsallmeta.csv'
	df = pd.read_csv(filename, encoding='latin-1')
	cleandf = cleantweets(df)
	autolessdf=removeautomation(cleandf, tokenizer)
	finaldf = ngramsfreq(autolessdf, 4)
	finaldf.to_csv('memes.csv')