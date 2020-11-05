## Sentence to Sentence Semantic Similarity

## Team 3:
1. Zoe Lie - zoelie
2. Sonia Mannan - smannan
3. Poojitha Vaddey - poojithavaddey
4. Dylan Zhang - ddyy 814

## Description
We propose evaluating sentence similarity based on thousands of sentence pairings from various news articles, 
online forums, and images captions since the Semmantic Textual Similarity(STS) measures the degree of equivalence
in the underlying semantics of paired snippets of text. The STS benchmark tasks are building algorithms and computational models to solve the 
deep natural language understanding problem and provide a standard setup for training, development, and testing on three selected genres.

We select the STS benchmark because the SemEval (Semantic Evaluation) shared task has been updated dataset annually since 2012 in order to stimulate research in semantic 
analysis and encourage the develpment of creative new methods to the modeling sentence level.

Sentence similarity is used in a range of Natural Language Processing tasks - when you type in a search and Google is able to recommend similar searches, when you ask 
Siri a question it can return a variety of results based on similar requests, or correcting grammatically incorrect sentences. These use cases demonstrate natural 
language processing has huge potential to change how well we communicate online, which is why we chose to focus on sentence similarity.


## Dataset
STS Benchimark: http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark

## Balance dataset before training

We consider use re-sampling tecniques to balance the training datset.

1. oversampling (draw more sample without substitution) from underrepresented class
2. Undersampling (draw less sample with/without substitution) from overrepresente3d class
3. Over-sampling followed by under-sampling (SMOTE + Tomek links or SMOTE + ENN)
4. Ensemble classifier using samplers internally (such as Easy Ensemble classifier)

## Potential methods
In the process of natural language processing, we often encounter scenarios where we need to find similar sentences or find approximate expressions of sentences.
For the problem of distance calculation, we will take a look at how to use Python to calculate sentence similarity.

We propose various pre-preocessing techniques and distance measures to evaluage how similar two sentences are. A distance measure is a mathematical formula specifying
how similar two sentences are. Distance measures therefore require text data to be pre-preocessed before it can be evaluated and the pre-preocessing technique used can
heavily influence the similarity evaluation of a sentence pair.

We propose the following pre-precessing techniques:

1. TF-IDF score: Each word in the sentence gets a score weighted by its frequency and importance to the sentence. Each sentence will be represented by a vector of TF-IDF scores.
2. Bag of Words: Each sentence is represented by list of 1's and 0's where as 1 indicates the sentence contains a specific word. We can experiment with both words and
N-germs (N-length substrings in the sentence).
3. Word embeddings: Use a neural network to weight each word based on their similarities and context within the sentence. Each sentence is represented by a compressed 
N-length vector of weights.
4. Sentence embeddings: A similar method to word embeddings but weights are learned based on sentence similarity, not word.

And the following distance measure:
1. Edit Distance(Levenshtein): The edit distance algorithm refers to the minimum number of edit operations required to convert two strings from one to the other.
The greater the distance between them, the more different they are. The permitted edting operations inclue replacing one character with another, inserting a character
, and deleting a character. This method does not require any pre-processing.

2. Jaccard index(Jaccard) - The Jaccard index is used to compare the similarities and differences between a limited sample dataset. The larger the Jaccard index coefficient
value, the higher the sample similarity. The calculaiton method of the Jaccard index is straightforward. It is the value obtained by dividing the intersection
of two sample by the union. When the two samples are the same, the result is 1, and when the two samples are completely different, the result is 0. This method can be
used to evaluate a Bag of Words Model.

3. Cosine Similarity: Measures the cosine of the angle between two vectors. The similar the angle, the larger the consine similarity is, and the more similar two documents are.
This measurement is good for documents that can be different sizes. This measure can be used to evaluate TF-IDF and embedding vecotors.

## Meausre success in thi project

The STS dataset is manually annotated by humans who score each sentence pair from 0 - 5 based on how similar they are. To evaluate we can scale each of our distance
measures between 0 - 5 and compare the distance measure we calculated to the annoated similarity based on RMSE for each pre-preocessing technique.

RMSE or Root mean squared error is an average measure of the magnitude of error from calculated similarities to actual ones. A lower RMSE would indicate a more 
accurate pre-processing technique.

## Cite
1. Imbalanced-learn: https://github.com/scikit-learn-contrib/imbalanced-learn
2. How to fix imbalanced dataset? https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb
3. TF-IDF: https://monkeylearn.com/blog/what-is-tf-idf/
4. Dataset Hub: https://github.com/brmson/dataset-sts
