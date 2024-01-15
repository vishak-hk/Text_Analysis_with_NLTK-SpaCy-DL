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
   Delve into the world of term frequency-inverse document frequency (TF-IDF) with scikit-learn. Implement `TfidfVectorizer` on the entire dataset to transform text into numerical vectors, capturing the importance of words in each document. Calculate the cosine similarity between two news articles, providing a measure of their textual similarity.

8. **Bonus:**
   Elevate the project by selecting a different dataset and applying various NLP techniques. Unleash your creativity in exploring new challenges and solutions within the vast field of Natural Language Processing.
