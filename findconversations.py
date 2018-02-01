import pandas as pd
import nltk
from nltk.util import ngrams
import os
import string
import re
import math
import time

def cleantweets(dataframe):
	df = pd.DataFrame.copy(dataframe)
	#clean punctuation from tweets
	tweetpuncts='!"$%&\'()*+,-./:;<=>?[\]^_`{|}~¡¿'
	df['text']=df['text'].apply(lambda x: re.sub('['+tweetpuncts+']', '', x))
	#clean URLs from tweets
	df['text']=df['text'].apply(lambda x: re.sub(r'http[^\s]+', '', x))
	#clean emoji from tweets
	df['text']=df['text'].apply(lambda x: re.sub(r'EMOJI\w+', '', x))
	#set creation time to python format
	df['created_at'] =df['created_at'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(x,'%a %b %d %H:%M:%S +0000 %Y')))
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
	#find most frequent large-scale ngrams and remove tweets containing them from corpus
	fdist = nltk.FreqDist(autodict)
	for gram, number in fdist.most_common(50):
		df = df[[set(list(gram)).issubset(set(row)) == False  for row in df['text']]]
	for row in df['text']:
		autolesstokens+=len(row)
		autolesstweets+=1
	#show statistics after automation is cleaned
	avgautolesstokenlen=int(totaltokens/totaltweets)
	print('Total Tokens in Corpus Automated Tweets Removed:', autolesstokens, 'Tokens')
	print('Average Tweet Length Automated Tweets Removed:', avgautolesstokenlen, 'Tokens')
	return df


def conversations(dataframe):
	df= dataframe
	allconvosdf=pd.DataFrame()
	ngramdict={}
	#prepare user id information
	df['in_reply_to_user_id_str'] = df['in_reply_to_user_id_str'].apply(lambda x: str(int(x)) if math.isnan(x) == False else 'NA')
	df['user_id_str'] = df['user_id_str'].apply(lambda x: str(int(x)))
	#create a dataframe of tweets where the user being replied to also appears as a user who has sent a tweet
	replydf = df.loc[df['in_reply_to_user_id_str'].isin(df['user_id_str'])]
	#create a dataframe of tweets where the user also appears as a user being replied to
	firsttweetdf=df.loc[df['user_id_str'].isin(replydf['in_reply_to_user_id_str'])]
	#for every unique user who gets a reply tweet, gather all the user's tweets and replies to that user, then order by time
	for user in firsttweetdf['user_id_str'].unique():
		convodf=pd.DataFrame()
		convodf = convodf.append(df.loc[df['user_id_str'] == user])
		convodf = convodf.append(df.loc[df['in_reply_to_user_id_str'] == user])
		convodf = convodf.sort_values(by='created_at')
		#if the user is only replying to themselves, do not include these tweets
		if len(convodf['in_reply_to_user_id_str'][convodf['in_reply_to_user_id_str']!= 'NA'].unique())==1:
			pass
		else:
			allconvosdf=allconvosdf.append(convodf)
	print('Number of users with replies:', len(allconvosdf['user_id_str'].unique()))
	allconvosdf['text']=allconvosdf['text'].apply(lambda x: ' '.join(x))
	return allconvosdf
#run the code and save spreadsheet to directory
if __name__ == '__main__':
	tokenizer = nltk.TweetTokenizer(preserve_case=False)
	filename='totaltweetsallmeta.csv'
	df = pd.read_csv(filename, encoding='latin-1')
	cleandf = cleantweets(df)
	autolessdf=removeautomation(cleandf, tokenizer)
	allconvosdf = conversations(autolessdf)
	allconvosdf.to_csv('convos.csv')