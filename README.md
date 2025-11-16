# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: AMIRTHAA R

*INTERN ID*: CT04DR1837

*DOMAIN*: Machine Learning

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

# DESCRIPTION ABOUT MY PROJECT 

Task 2 focuses on implementing a Sentiment Analysis model using Natural Language Processing (NLP) techniques. Sentiment analysis seeks to determine whether a piece of text expresses positive or negative emotions. This task uses the TF-IDF Vectorization method and Logistic Regression classifier to build a text classification system capable of predicting sentiment from written reviews.

The task begins by preparing a dataset consisting of customer reviews. Since external datasets could not be downloaded due to system limitations, a custom dataset of 20 balanced reviews was created manually, containing an equal number of positive and negative sentiment labels. This dataset simulates real-world review data and allows the model to learn patterns associated with different sentiments. The data is stored in a Pandas DataFrame for easy manipulation and preprocessing.

The next step is to convert textual data into numerical features that machine learning algorithms can understand. This is achieved using TF-IDF (Term Frequency–Inverse Document Frequency), a widely used text vectorization method. TF-IDF assigns importance to words based on how frequently they appear across documents while reducing the weight of commonly occurring words. Using scikit-learn’s TfidfVectorizer, the text is converted into a matrix of numerical values, representing word importance.

To avoid bias in the training process, the dataset is split into training and testing subsets using stratified sampling. The model used for classification is Logistic Regression, which is effective for binary classification tasks such as sentiment analysis. The model is trained on the TF-IDF vectors and tested on unseen data.

Evaluation is carried out using accuracy, classification report, and confusion matrix. These metrics show how effectively the model distinguishes between positive and negative reviews. Despite the small dataset, the model achieves reasonable accuracy, demonstrating its ability to learn meaningful patterns from text. A classification report provides precision, recall, and F1-score, offering deeper insight into performance.

Overall, Task 2 successfully implements a complete NLP pipeline—from text preprocessing to vectorization, model training, and performance evaluation. It demonstrates how machine learning can be applied to linguistic data to extract meaningful insights about sentiment.

# OUTPUT
<img width="440" height="396" alt="Image" src="https://github.com/user-attachments/assets/13f45d70-a7c2-483c-bc16-ec0fc164faa5" />
