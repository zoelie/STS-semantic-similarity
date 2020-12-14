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
article pairs, scored by human readers for similarity.

## Experiments/Analysis

### BERT

### Roberta

### Universal Sentence Encoders

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
