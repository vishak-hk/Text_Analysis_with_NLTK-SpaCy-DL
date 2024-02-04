# Natural Language Processing with NLTK, SpaCy, Word2Vec, and TF-IDF

## Objective:
Learn and apply Natural Language Processing (NLP) techniques using popular Python libraries including NLTK, SpaCy, Word2Vec, and TF-IDF. This assignment focuses on tasks such as tokenization, stemming, lemmatization, named entity recognition, Word2Vec, and TF-IDF. The BBC News dataset from Kaggle serves as the basis for practical implementation.

1. **Importing Libraries:**
   Begin by importing the necessary libraries, setting the foundation for comprehensive NLP exploration. NLTK for versatile NLP functionalities, SpaCy for advanced NLP tasks, gensim for Word2Vec implementation, and scikit-learn for TF-IDF vectorization.

2. **Loading the Dataset:**
   Utilize the `read_csv()` function to load the BBC News dataset (`BBC_DATA.csv`) into a pandas DataFrame. With 1,490 rows and 3 columns, the dataset's first column contains the text of news articles, providing rich content for NLP analysis.

3. **Tokenization with NLTK:**
   Dive into NLTK's powerful tokenization capabilities. Use `word_tokenize()` for breaking down text into words and `sent_tokenize()` for segmenting text into sentences. Apply these functions to a sample news article, gaining insights into the underlying structure of the text.

4. **Stemming and Lemmatization with NLTK:**
   Explore NLTK's linguistic tools for reducing words to their base or root form. Implement the Porter Stemmer for stemming, simplifying words to their core, and WordNetLemmatizer for lemmatization, deriving the base form of words. Apply these techniques to a sample news article, understanding how they impact language processing.

5. **Named Entity Recognition with SpaCy:**
   Leverage SpaCy's pre-trained model to identify and classify named entities in text. Perform named entity recognition on a sample news article, revealing entities such as persons, organizations, and locations. Visualize these entities using displaCy, gaining a visual understanding of the identified information.

6. **Word2Vec with gensim:**
   Enter the realm of word embeddings with gensim's Word2Vec. Implement this technique on the entire dataset to create vector representations of words. Train the Word2Vec model and explore the vector representation of a sample word, uncovering the semantic relationships encoded in the vectors.

7. **TF-IDF with scikit-learn:**
   Explore the power of term frequency-inverse document frequency (TF-IDF) with scikit-learn. Utilize the `TfidfVectorizer` to transform the entire dataset's text into numerical vectors, effectively capturing the significance of words in each document. This technique proves valuable in representing textual information quantitatively, enabling a deeper understanding of document content.

   To take it a step further, calculate the cosine similarity between two news articles. This measurement offers insights into the textual similarity between documents, providing a quantitative measure of their content-related closeness.

   Additionally, complementing the TF-IDF analysis, an SVM (Support Vector Machine) model is employed to predict the category of new text. The model is trained on a labeled dataset containing articles categorized into 'sport,' 'business,' 'politics,' 'entertainment,' and 'tech.' By using the TF-IDF-transformed features, the SVM model learns to classify and predict the category of new textual data.

   This combined approach, involving TF-IDF for feature extraction and SVM for category prediction, demonstrates a powerful methodology for text analysis and categorization.

8. **Bonus:**
   Elevate the project by selecting a different dataset and applying various NLP techniques. Unleash your creativity in exploring new challenges and solutions within the vast field of Natural Language Processing.
