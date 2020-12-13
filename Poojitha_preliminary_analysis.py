import pandas as pd
import matplotlib.pyplot as plt


def loaddata(filename):
    col_names =["col1","col2","col3","col4","similarity","sen1","sen2"]
    return pd.read_csv(filename, sep= "\t", names=col_names, usecols=["similarity","sen1","sen2"])

train_df=loaddata('sts-train.csv')

#Describing the data
print("Diplaying train_df head\n ",train_df.head())
print("Describing the columns of train_df\n", train_df.describe(include='all'))

#Checking for null values
print("Checking for null values column wise:\n", train_df.isnull().sum())
print("Checking for total number of null values in the dataframe:\n", train_df.isnull().sum().sum())

#Mean of similarity scores
print("Printing the mean of the similarity scores", train_df['similarity'].mean())

#Frequency of similarity scores
print("Printing the frequency of similarity scores\n ", train_df['similarity'].astype(int).value_counts())
cnt=0
cnt = [cnt+1 for i in train_df['similarity'] if i<0]
print("Checking for negative values in similarity score: ", cnt)

#Plotting the histogram of similarity scores in train_df
n, bins, patches = plt.hist(train_df["similarity"], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.xlabel('Similarity score')
plt.ylabel('Frequency')
plt.title('Plot for similarity-frequency for training data')
plt.xlim(0, max(train_df['similarity']))
plt.ylim(100, 750)
#plt.grid(True)
plt.show()

def clean_text(sentence):
    # Removing non ASCII chars
    sentence = sentence.replace(r'[^\x00-\x7f]',r' ') 
    sentence = sentence.lower()
    #punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        sentence = sentence.replace(char, ' ')
    return sentence

train_df['sen1']= clean_text(train_df['sen1'].astype(str))
train_df['sen2']= train_df['sen2'].map(clean_text)
print(train_df.head())





