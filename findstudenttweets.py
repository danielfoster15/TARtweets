import pandas as pd
import nltk
from nltk.util import ngrams
import os
import string
import re



def cleantweets(dataframe):
	df = dataframe
	#clean punctuation from tweets
	tweetpuncts='!"$%&\'()*+,-./:;<=>?[\]^_`{|}~¡¿'
	df['text']=df['text'].apply(lambda x: re.sub('['+tweetpuncts+']', '', x))
	#clean URLs from tweets

	df['text']=df['text'].apply(lambda x: re.sub(r'http[^\s]+', '', x))
	#clean emoji from tweets
	df['text']=df['text'].apply(lambda x: re.sub(r'\\ud', 'EMOJIud', x))
	df['text']=df['text'].apply(lambda x: re.sub(r'EMOJI\w+', '', x))

	df['description']=df['description'].apply(lambda x: str(x))
	df['description']=df['description'].apply(lambda x: re.sub('['+tweetpuncts+']', '', x))
	#clean URLs from tweets

	df['description']=df['description'].apply(lambda x: re.sub(r'http[^\s]+', '', x))
	#clean emoji from tweets
	df['description']=df['description'].apply(lambda x: re.sub(r'\\ud', 'EMOJIud', x))
	df['description']=df['description'].apply(lambda x: re.sub(r'EMOJI\w+', '', x))

	return df

def removeautomation(dataframe, tokenizer):
	df=dataframe
	#tokenize dataframe
	df['text']=df['text'].apply(lambda x:tokenizer.tokenize(x))
	df['description']=df['description'].apply(lambda x:tokenizer.tokenize(str(x)))
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
#find tweets with 'estudiante' in the description
def ngramsfreq(dataframe):
	df = dataframe
	df=df[df['description'].apply(lambda x: 'estudiante' in set(x))]
	df['text']=df['text'].apply(lambda x: ' '.join(x))
	df= df.drop_duplicates(subset='text')
	return df

#run the code and save spreadsheet to directory
if __name__ == '__main__':
	tokenizer = nltk.TweetTokenizer(preserve_case=False)
	filename='totaltweetsallmeta.csv'
	df = pd.read_csv(filename, encoding='latin-1')
	cleandf = cleantweets(df)
	autolessdf=removeautomation(cleandf, tokenizer)
	finaldf = ngramsfreq(autolessdf)
	finaldf.to_csv('studentweets.csv')