# IMDB Movie Review Sentiment Analysis

A machine learning project that classifies movie reviews as **positive** or **negative** using Naive Bayes algorithms.

## Dataset
- IMDB Movie Reviews dataset (50,000 reviews)
- Each review labeled as either "positive" or "negative"

## Approach

**1. Text Cleaning**
- Removed HTML tags (e.g., `<br />`) using regex
- Converted all text to lowercase
- Removed special characters and punctuation
- Removed stopwords (common words like "the", "is", "and" that don't add meaning)
- Applied stemming to reduce words to their root form (e.g., "loved" → "love")

**2. Feature Extraction**
- Used `CountVectorizer` to convert cleaned text into numerical word-count data
- Kept the data in sparse format to handle the full dataset efficiently

**3. Model Training**
- Split data into 80% training, 20% testing
- Trained two models: `MultinomialNB` and `BernoulliNB`
- Chose these algorithms specifically because they work well with word-count/text data

## Results
| Model | Accuracy |
|---|---|
| MultinomialNB | 85.34% |
| BernoulliNB | 84.76% |

## Tools Used
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK (for stopwords and stemming)

## How to Run
1. Install dependencies: `pip install pandas numpy scikit-learn nltk`
2. Run `naive_bayes.py`