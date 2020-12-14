## Sentence to Sentence Semantic Similarity

## Team 3:
- Zoe Lie - zoelie
- Sonia Mannan - smannan
- Poojitha Vaddey - poojithavaddey
- Dylan Zhang

## Abstract

Our team aimed to evaluate sentence semantic similarity using a variety of pre-processing approaches. Semantic
similarity means that the sentence and word context is accounted for when determining how similar two sentences
are. For example, "This article is not about a dog" and "I love articles about dogs!" both contain the word "dog"
but are not talking about the same thing. Semantic similarity is useful across a wide range of NLP tasks such as:
returning relevant articles in a search result, communicating with virtual assistants, or making recommendations
for new posts to view on a social media platform.

Text must be preprocessed into numerical vectors before analyzing because most algorithms cannot work with text
data firsthand. Once processed the numerical vectors can be scored and compared to a baseline similarity determined
by human readers for evaluation. Our team used the STS Benchmark dataset comprised of thousands of news headline
article pairs, scored by human readers for similarity. We applied various preprocessing techniques including: BERT,
Infosert, and Sent2Vec to convert text data into numerica vectors, scored similarity between sentence pairs using
cosine distance, and evaluted our calculated similarities to the human reader's using Pearson's correlation coefficient.

A higher correlation between calculated and actual scores indicates better performance of the preprocessing
technique at representing the sentences. In addition to determining similarities within article headlines our team also wanted
to cluster the headlines to see if any apparent patterns or topics arose. We used the highest scoring preprocessing
technique to prepare the documents and applied k-means clustering, then looked at individual clusters to determine topics.
From this we discovered the headlines clustered into common news topics such as: Foreign Affairs, Military, Dogs, and Finance.

## Experiments/Analysis

### BERT

### Roberta

### Universal Sentence Encoders

There are a variety of methods to preprocess text such as Bag of Words and TF-IDF but so far these methods have failed to
accurately represent word context and have not performed well at determining similarity. Recently research has instead focused
on neural and deep learning networks to discover context between words.

Neural nets are composed of neurons, units that take the weighted sum of a set of inputs and apply a non-linear function on 
the ouput. Neurons are connected in layers, with connections to previous and future layers. The weight of each connection 
indicates how important the previous neuron is to the next. Finally, weights are updated during a process called backpropagation
to minimize some loss function.

In 2013 a team of researchers at Google published a paper proposing the method "Word2Vec", using neural networks to capture
word context and numerically reprsent text. They proposed two neural networks: CBOW and Skipgram. CBOW takes a window of words
around a target word, and given the window, or context, tries to predict the target word. The output vector, or embedding, contains weights
indicating how related each word is to every other in the vocabulary. After one pass, each context word should be most related
to the target. This loss is then used to update weights during backpropagation such that the output vector after training
accurately represents each word and its context.

![CBOW](report_images/cbow_model.png)

Skipgram is similar but does the opposite of CBOW - given a target word it attempts to predict its context, which words will
appear around it. It updates output weights to minimize the loss between predicted weights for the context, target pair.

![Skipgram](report_images/skipgram_model.png)

The above techniques are used to numerically represent individual words but fall short at representing sentences and documents.
In 2018 Cer et al, another team of researchers at Google, proposed the "Universal Sentence Encoder" model to solve this.
They proposed two model types to combine individual words embeddings: a Deep Averaging Network (DAN) and a Transformer network.

The DAN averages embeddings for words and bigrams then inputs them into another deep feedforward neural network to produce
condensed sentence embedding. This approach is simpler and more efficient but doesn't take word ordering into account, although
performs well on classification tasks.

The transformer uses an attention mechanism to focus on specific words, producing an embedding for the word that accounts for
both order and context. The sentence embedding is then produced by taking the sum of the weights for each word from all the
attention vectors. Transformers are more accurate but also less efficient to train and predict.

The researchers trained the initial model on Wikipedia data and used transfer learning to fine-tune the model to other domains
such as: movie reviews, sentiments, customer reviews, and the STS Benchmark. They used Skipgram as the initial embeddings and
found that the transformer model performed better on the STS Benchmark than the DAN but took longer to train.

Pretrained models are available on tensorflow hub and were used evaluate sentence similarity on our dataset. The following heatmap
shows similarities between 5 sample sentences from the dataset. Most sentences are not similar to each other, but some
sentences such as "A man is playing a large flute" are similar to "Three men are playing chess" and "A man is playing the cello".

![Sent2Vec Heatmap](sonia-mannan-sentence-embeddings-results/sample_heatmap.png)


## Comparisons:

The goal for this project is to develop accurate, condensed, numeric representations for text such that the contextual
similarity for two documents can be compared. Text must be pre-processed before analyzed because most algorithms cannot
work with textual data firsthand, so this preprocessing step is key in analyzing text data and is evaluted using the methods
below:

Afterwards, we used cosine similarity to measure the distance between the vectors, creating our own calculated "score".
Calculated scores were compared against real scores determined by human readers using Pearson's correlation, which
measures the strength in correlation between two variables. Ideally calculated scores should be highly correlated to
actual scores, so a higher Pearson's would indicate a better calculated score and pre-processing technique.

Cosine similarity measures the cosine of the angle between two vectors and is a good distance measure for comparing
documents that are different sizes. It is calculated by taking the dot product between the vectors and dividing by
their magnitudes (see formula below). Cosine similarities can range between 0-1, where more similar documents will
have a score of 1.

![Cosine Similarity Formula](report_images/cosine_sim_formula.png)

Pearson's Correlation measures the linear correlation between two variables from -1 to 1, with 1 being complete positive
correlation, 0 being no correlation, and -1 being complete negative correlation. It's calculated by dividing the covariance
between the two variables, divided by their standard deviation (see formula below). Covariance measures the joint variability
between two variables (if they behave the same it is positive and otherwise negative) while standard deviation measures
the dispersion in a dataset (low standard deviation means values are all close to the mean, higher means they are spread out
over a wide range).

![Pearsons Formula](report_images/pearsons_formula.png)

## Conclusion


## Citation
1. Sentence Transformers: https://pypi.org/project/sentence-transformers/
2. BERT: https://keras.io/examples/nlp/semantic_similarity_with_bert/
3. Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity#:~:text=In%20the%20case%20of%20information,be%20greater%20than%2090%C2%B0.
4. Pearsons Correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
5. CBOW and Skipgram Diagrams: https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa
6. Universal Sentence Encoder: https://arxiv.org/pdf/1803.11175.pdf
