# Using Roberta model for sentiment analysis
#Using pretrained model from huggingface
# importing libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Taking Input :
comment = input("Enter the Comment Or Review  : ")

# precprcess tweet
comment_words = []

for word in comment.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    
    elif word.startswith('http'):
        word = "http"
    comment_words.append(word)

comment_proc = " ".join(comment_words)
# print(comment_proc)
# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis
encoded_tweet = tokenizer(comment_proc, return_tensors='pt')

# output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
output = model(**encoded_tweet)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

# Showing the Probability score of each sentiment
for i in range(len(scores)):
    l = labels[i]
    s = scores[i]
    print(l,s)

print("The Sentiment of the Comment is : ",labels[scores.argmax()])