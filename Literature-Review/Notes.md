#### Sentiment Analysis on Product Reviews Using Machine Learning Techniques; Rajkumar S. Jagdale, Vishal S. Shirsat and Sachin N. Deshmuk
* Amazon product reviews: Camera, Laptops, Mobile phones, tablets, TVs, video surveillance (approx 13k total reviews)
* Classified reviews as positive or negative
* Naïve Bayes: 98.17% accuracy, SVM: 93.54% for Camera Reviews
* Process Overview:
  * Split review by sentence
  * Preprocess each sentence (tokenization, stop word removal, stemming, punctuation marks removal)
  * Convert to bag of words (BOW)
  * Calculate sentiment score by comparing BOW with opinion lexicon
  * ML on sentiment scores
* Future work: aspect level analysis could improve (e.g. Camera’s quality, megapixel, picture size, structure, lens, picture quality, etc.)

#### Yelp Dataset Challenge: Review Rating Prediction; Nabiha Asghar
* **Summary**: Very detailed overview of methods, lots of great info, but underwhelming results.
* **Helpful definition / problem overview for presentation**:
  * Review Rating Prediction (RRP) = The problem of predicting a user's star rating for a product, given the user's text review for that product.
  * On famous websites like Amazon and Yelp, many products and businesses receive tens or hundreds of reviews, making it impossible for readers to read all of them. Generally, readers prefer to look at the star ratings only and ignore the text. However, the relationship between the text and the rating is not obvious, as illustrated in Figure 1. In particular, several questions may be asked:
    * Why exactly did this reviewer give the restaurant 3/5 stars?
    * In addition to the quality of food, variety, size and service time, what other features of the restaurant did the user implicitly consider, and what was the relative importance given to each of them?
    * How does this relationship change if we consider a different user's rating and text review?
  * The process of predicting this relationship for a generic user (but for a specific product/business) is called Review Rating Prediction.
    * Concretely, given the set S = {(r1, s1), …, (rn, sn)} for a product P, where ri is the ith user's text review of P and si is the ith user's numeric rating for P, the goal is to learn the best mapping from a word vector r to a numeric rating s.
  * Review Rating Prediction is a useful problem to solve, because it can help us decide whether it is enough to look at the star ratings of a product and ignore its textual reviews.
  * However, it is a hard problem to solve because two users who give a product the same rating, may have very different reasons for doing so. User A may give a restaurant 2/5 stars because it does not have free wifi and free parking, even though the food is good. User B may give the same restaurant a rating of 2/5 because he does not care about the wifi and parking, and thinks that the food is below average. Therefore, the main challenge in building a good predictor is to effectively extract useful features of the product from the text reviews and to then quantify their relative importance with respect to the rating.
* **Note**: About 66% of these reviews rate the corresponding restaurants very highly (at least 4 stars); the other classes are smaller. (In my (Julia's) opinion, this could be a reason for the poor results (unbalanced classes))
* **Preprocessing**: removed capitalizations, stop words and punctuations
* **Feature Extraction**: feature vector for each review from four different methods:
  * Unigrams (AKA Bag of Words): each unique word in the pre-processed review corpus is considered as a feature.
     * word-review matrix is constructed, where entry (i,j) is the frequency of occurrence of word i in the jth review
     * Apply the TF-IDF (Term Frequency - Inverse Document Frequency) weighting technique to this matrix to obtain the final feature matrix (weighting technique assigns less weight to words that occur more frequently across reviews (e.g. “food") because they are generally not good distinguishers between any pair of reviews and a high weight to more rare words.)
     * Each column of this matrix is a feature vector of the corresponding review.
  * Unigrams & Bigrams:
     * To capture the effect of phrases such as ‘tasty burger’ and ‘not delicious’, we add bigrams to the unigrams model.
     * Now, the dictionary additionally consists of all the 2-tuples of words (i.e. all pairs of consecutive words) occurring in the corpus of reviews.
     * The matrix is computed as before; it has more rows now. As before, we apply TF-IDF weighting to this matrix so that less importance is given to common words and more importance is given to rare words.
  * Unigrams, Bigrams, & Trigrams:
     * The same trigram would rarely occur across different reviews, because two different people are unlikely to use the same 3-word phrase in their reviews. Therefore, the results of this model are not expected to be very different from the unigrams+bigrams model.
  * Latent Semantic Indexing (LSI):
     * more sophisticated method of lexical matching, which goes beyond exact matching of words. It finds ‘topics' in reviews, which are words having similar meanings or words occurring in a similar context.
     * In LSI, we first construct a word-review matrix M, of size m x t, using the unigrams model, and then do Singular Value Decomposition (SVD) of M.
     * The SVD function outputs three matrices: the wordtopic matrix U of size m x m, the rectangular diagonal matrix S of size m x t containing t singular values, and the transpose of the topic-review matrix V of size t x t. We use V as the feature matrix.
     * The singular values matrix S has t non-zero diagonal entries that are the singular values in decreasing order of importance.
     * The columns of S correspond to the topics in the reviews. The ith singular value is a measure of the importance of the ith topic.
     * By default, t = the size of vocabulary (i.e. 171,846). However, the first t*  topics can be chosen as the most important ones, and thus the top t*  rows of V can be used as the feature matrix.
     * Determining the value of t* is crucial, and this can be done by examining a simple plot of the singular values against their importance, and looking for an ‘elbow' in the plot.
* **ML Models**:
  * Logistic regression
  * Multinomial Naive Bayes classification
  * Perceptron (n_iterations = 50)
  * SVM (linear, tolerance = 0.001)
     * For each feature extraction method, we do internal 3-fold cross validation to choose the value of C that gives the highest accuracy. It turns out that C = 1.0 works best every time.
* **Performance Metrics**:
  * 80/20 train/test split
  * 3-fold cross validation on the training set and compute two metrics, Root Mean Squared Error (RMSE) and accuracy, for the training fold as well as the validation fold
* **Results**: Logistic Regression achieved the highest accuracy of 64% using the top 10,000 Unigrams & Bigrams as features, followed very closely by Linear SVC which achieved 63% accuracy using the top 10,000 Unigrams & Bigrams. (I took more extensive notes on the rest of the results but am not including them here for brevity's sake).
* **Future Work**:
  * more sophisticated feature engineering methods, such as Parts-of-Speech (POS) tagging and spell-checkers, to obtain more useful n-grams (e.g. instead of considering all possible bigrams, we can extract all the adjective-noun pairs or all the noun-noun pairs to get more meaningful 2-tuples)
  * We can try more elaborate experiments for LSI that consider more than 200 features. Moreover, instead of performing singular value decomposition of unigrams only, we can add other n-grams.
  * ordered/ordinal logistic regression (model takes into consideration the fact that the class labels 1 and 2 are closer to each other, than the labels 1 and 4.)
  * Non-linear classifier 

#### Sentiment Analysis: A Systematic Case Study with Yelp Scores; Wenping Wang Et al.
* ...

#### Ensemble Sentiment Analysis Using Bi-LSTM and CNN; Puneet Singh Lamba Et al.
* ...

#### Sentiment Analysis of Restaurant Reviews using Combined CNN-LSTM; Naimul Hossain Et al.
* ...

#### Sentiment Analysis of Yelp Reviews by Machine Learning; Hemalatha S, Ramathmika
* ...

#### Sentiment Analysis on Food Review using Machine Learning Approach; Nourin Islam, Ms. Nasrin Akter, Abdus Sattar
* ...

#### Sentiment Analysis using Machine Learning Techniques on Python; Ratheee Et al.
* ...

#### Sentiment Analysis of Yelp‘s Ratings Based on Text Reviews; Xu Et al.
* Goal to apply existing supervised learning algorithms to predict a review’s rating on a number scale base on text alone.
o	  Experiment with Naïve Bayes, Perceptron, and Multiclass Support Vector Machine (SVM). Compare predictions with actual ratings.
o	Uses precision and recall measuring effectiveness of the algorithms and compare results of the different approaches.
o	Explore various feature selection algorithms. Sentiment dictionary, own built feature set, remove stop words & stemming.
o	Found other algorithms that are not suitable to use for this.
•	Data – 1,125,458 user reviews from 5 different cities. Wrote a python parser to read in the json data files. Ignore information other than star ratings and text reviews. 
•	Use hold-out cross validation (randomly split the data into training and test data. Sample size of 1,000,000. (70% training, 30% test).
•	Bernoulli sampling could improve sampling, to reduce dominance of training set by certain business categories. 
•	Preprocessing – removed all punctuation and spaces from the review text. Convert capital letters to lower case (helps with feature selection). 
•	Feature selection – 1) using an existing opinion lexicon. 2) building feature dictionary using training data 
o	Feature selection algorithm – Bing Liu Opinion Lexicon. Has link to download. 
o	Other feature selection algorithm – loops over training set to build a dictionary mapping each word to frequency of occurrence in the training set.
	Appends “not_” to every word between negation and following punctuation. 
	Removes stop words (common words) found using the Terrier stop wordlist. 
	Stemming – reducing a word to its stem/root word. Uses Porter Algorithm, integrated in Natural Language Toolkit (NLTK).
	Stopword + Stemming gives the best precision and recall percentages for test error using Naive Bayes. (highest prediction accuracy)
	High bias using an existing lexicon – lot of the features in the lexicon are not from the yelp dataset. Yelp has words spelled wrong, but still has sentiment behind them. 
	Negation handling did not improve results. Potentially overfitting by adding more words. May have generated noise. Had 39,030 weights in the model.
o	Precision and recall percentages much better for ratings 1 and ratings 5
•	Only training on 2 categories for each sentence within a review, positive and negative. Difficult to predict how positive and how negative. 
•	Predictions are predicted to be consistently lower than the actual rating. They scaled their predictions to have the same mean and standard deviation of the actual stary ratings. This did not improve prediction accuracy. 
o	Separated reviews into two groups: 1-3 Starts as Positive, 4-5 stars as negative. 
	Not sure why they did this. 
•	Naive Bayes – in the sciit-learn ML library. Most suitable for text classification. Use laplace smoothing to avoid overfitting. Implemented Binarized Naive bayes using Boolean feature vector (1 or 0 if the word occurred or not). Word occurrences may matter more than frequency. Much better than the perceptron algorithm. Star 4 and 5 are difficult to distinguish from each other. Same for stars 1, 2, 3



### Some more papers I found that were used as references in other papers:

#### Integrating Collaborative Filtering and Sentiment Analysis: A Rating Inference Approach; Cane Wing-ki Leung and Stephen Chi-fai Chan and Fu-lai Chung
* ...

#### The Bag-of-Opinions Method for Review Rating Prediction from Sparse Text Patterns; Qu Et al.
* ...

#### Predicting a Business’ Star in Yelp from Its Reviews’ Text Alone; Fan, Khademi
* ...
