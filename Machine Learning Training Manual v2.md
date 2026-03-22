# Machine Learning: The Next Layer
A Plain-English Training Manual for Volume 1 Graduates

## How To Use This Manual
- Audience: students who have completed Volume 1 (or equivalent basics in pandas and scikit-learn).
- Goal: solve a wider range of real-world ML problems, including text, time-series, recommendations, and anomaly detection.
- Prerequisites:
  - You can split data into train/validation/test sets.
  - You can train and evaluate baseline models.
  - You understand classification, regression, and clustering basics.
  - You can read and run simple Python notebooks/scripts.
- Pace:
  - Fast track: 30 days (1 focused session/day)
  - Standard track: 8 weeks (4 sessions/week)
- Study method:
  - Read the concept in plain language
  - Run the mini exercise
  - Answer the checkpoint questions
  - Explain the idea in your own words

What is different from Volume 1:
- Data types become less "clean": text and time sequences are common.
- Leakage risks become trickier (especially in time-series and recommendations).
- Metrics and evaluation design matter even more than model choice.

Pattern reminder for each chapter:
- Concept
- Why it matters
- Quick exercise
- Visual aid

---

## Full Course Structure

### Orientation: From Volume 1 to Volume 2
1. What you already know
2. The four new problem families
3. What changes when data is text or time-based
4. Environment setup refresh
5. Checkpoint + readiness self-test

### Text as Data (Before Modeling)
1. Why text needs preprocessing
2. Tokenization, vocabulary, and bag-of-words
3. TF-IDF in plain language
4. Hands-on: convert text to features
5. Checkpoint + common preprocessing mistakes

### Text Classification
1. Mapping text tasks to classification
2. Naive Bayes intuition for text
3. Logistic regression on TF-IDF features
4. Hands-on: intent/topic classifier mini project
5. Checkpoint + error analysis exercise

### Sentiment Analysis
1. Sentiment tasks and business use cases
2. Lexicon-based sentiment (no training)
3. ML-based sentiment (supervised)
4. Hands-on: review sentiment classifier
5. Checkpoint + threshold and imbalance exercise

### Word Embeddings (Meaning Beyond Word Counts)
1. Why bag-of-words misses context
2. Embedding intuition (words as coordinates)
3. Pretrained embeddings for beginners
4. Hands-on: similarity lookup mini lab
5. Checkpoint + TF-IDF vs embedding comparison

### Time-Series Basics
1. What makes time-series different
2. Trend, seasonality, noise in plain words
3. Lag and rolling features for forecasting
4. Time-aware splitting (no random shuffle)
5. Hands-on: build a clean time-series feature table
6. Checkpoint + leakage traps

### Forecasting Models
1. Baselines first (naive and moving average)
2. ARIMA intuition (no heavy math)
3. Regression-style forecasting with lag features
4. Forecast metrics (MAE, RMSE, MAPE)
5. Hands-on: weekly demand/sales forecast
6. Checkpoint + horizon and uncertainty exercise

### Recommendation Systems
1. Recommendation problem framing
2. Collaborative filtering intuition
3. Content-based recommendation intuition
4. Cold-start and sparsity challenges
5. Hands-on: simple product/movie recommender
6. Checkpoint + quality tradeoff exercise

### Anomaly Detection
1. What anomalies are in real systems
2. Statistical rules (z-score, IQR)
3. Isolation Forest intuition
4. Evaluating rare-event detection
5. Hands-on: fraud/equipment anomaly mini lab
6. Checkpoint + threshold tuning exercise

### Responsible and Real-World ML
1. Bias risks in text systems
2. Drift and retraining in time-series
3. Recommendation feedback loops
4. Privacy in logs, text, and behavioral data
5. Should we deploy? expanded readiness checklist
6. Case studies with trade-offs

### Capstone Project
1. Choose one domain and one new problem type
2. Define target metric and deployment scenario
3. Build baseline + one controlled improvement
4. Evaluate reliability and responsible-ML risks
5. Final presentation template (portfolio-ready)

---

## Chapter 0
### Orientation: From Volume 1 to Volume 2

#### 1) What You Will Learn
This chapter helps you transition smoothly from beginner ML foundations to new, practical ML problem families.

You are not starting from zero anymore.
You already have useful habits from Volume 1.
Now we extend those habits to harder data types and more realistic workflows.

By the end, you should be able to:
1. Explain what changes in Volume 2 compared with Volume 1
2. Identify which new problem family fits a use case
3. Recognize the biggest new leakage/evaluation risks
4. Confirm your setup is ready for the next chapters

Pattern reminder:
- Concept
- Why it matters
- Quick exercise
- Visual aid

#### 2) What You Already Know
If you completed Volume 1, you already know the core ML loop:
1. Define problem
2. Prepare data
3. Split data
4. Train baseline
5. Evaluate honestly
6. Improve carefully
7. Deploy and monitor

You also already know:
- Regression for numeric targets
- Classification for labels
- Clustering for unlabeled grouping
- Core metrics like MAE, RMSE, precision, recall, F1
- Why leakage can fake performance

This foundation is enough to start Volume 2.

Why it matters:
- Many learners underestimate what they already know.
- Confidence helps you learn faster when topics become less familiar.

Quick exercise:
- In two lines, write the ML workflow from memory without checking notes.

#### 3) The Four New Problem Families
Volume 2 focuses on four practical problem families:

1. Text ML
- Use language as input data.
- Examples: intent detection, spam filtering, sentiment.

2. Time-series forecasting
- Predict future values from historical sequences.
- Examples: weekly sales, demand, energy usage.

3. Recommendation systems
- Suggest items users are likely to value.
- Examples: products, videos, songs, articles.

4. Anomaly detection
- Detect unusual or suspicious behavior.
- Examples: fraud, equipment failure, intrusions.

Why it matters:
- This gives you a clear map of what this volume is for.
- You can quickly route new business problems to the right chapter.

Quick exercise:
- Match each use case to a family:
1. Predict next week's call volume
2. Detect suspicious credit-card transactions
3. Suggest similar movies
4. Classify customer support message intent

#### 4) What Changes With Text and Time Data
In Volume 1, most examples used clean tabular rows with independent records.
In Volume 2, that assumption often breaks.

Text differences:
- Raw text is unstructured.
- You must convert words into numeric features before training.
- Preprocessing choices heavily affect results.

Time-series differences:
- Order matters.
- Future data must never leak into training features.
- Random split is often unsafe.

Recommendation differences:
- Data is sparse (most users rate only a tiny fraction of items).
- Feedback loops can reinforce popularity bias.

Anomaly differences:
- Positive class is rare.
- Accuracy can be misleading.
- Threshold selection is usually business-critical.

Why it matters:
- You can keep the same ML workflow, but your data assumptions must adapt.
- Most new mistakes in Volume 2 come from reusing Volume 1 habits without adjusting for data type.

Quick exercise:
- Why is random train/test split often risky for time-series forecasting?

#### 5) Setup Refresh (Practical)
Recommended environment (Python):
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk (or spaCy later)
- statsmodels (for ARIMA chapters)

Quick setup check script:
```python
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import seaborn
import statsmodels

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", seaborn.__version__)
print("statsmodels:", statsmodels.__version__)
print("Environment check passed.")
```

If one import fails, install the missing package before continuing.

#### 6) Readiness Self-Test
Answer quickly without notes:
1. Difference between precision and recall?
2. Why do we keep a final untouched test set?
3. One example of leakage?
4. One reason random split can be wrong for time-series?

If you can answer all four clearly, continue to Chapter 1.

#### 7) Orientation Map

![Volume 2 orientation map](images/manual_visuals_v2/chapter0_orientation_map.png)

What to notice:
- Volume 2 extends Volume 1 workflow, not replaces it.
- Data type choice drives feature engineering and evaluation strategy.

#### 8) Common Transition Mistakes
- Jumping to advanced models before defining the problem clearly
- Reusing random split on time-ordered data
- Treating raw text like ready numeric features
- Evaluating anomalies with accuracy only
- Ignoring deployment constraints when choosing a recommendation approach

#### 9) Chapter Checkpoint
Can you answer these in your own words?
1. What are the four new problem families in Volume 2?
2. What major assumption changes when data is time-ordered?
3. Why does preprocessing matter so much for text ML?
4. Why can anomaly detection metrics be tricky?

If yes, move to the next chapter.

---

## Chapter 1
### Text as Data (Before Modeling)

#### 1) What You Will Learn
This chapter gives you the core mental model for turning raw text into model-ready numeric features.

Text looks simple to humans but is not directly usable by most ML algorithms.
Before you can classify text, predict sentiment, or rank recommendations from language, you need a robust text-to-features pipeline.

By the end, you should be able to:
1. Explain tokenization, vocabulary, and bag-of-words in plain English
2. Explain what TF-IDF does and why it helps
3. Build a basic text feature matrix with scikit-learn
4. Avoid common preprocessing mistakes that hurt model quality

Pattern reminder:
- Concept
- Why it matters
- Quick exercise
- Visual aid

#### 2) Why Text Needs Preprocessing
Most ML models expect numbers, not raw sentences.

Example:
- Raw text: "The delivery was late but support fixed it quickly."
- Model input needed: numeric vector (feature values)

So text preprocessing is the bridge from language to numbers.

Layered view:
- Layer 1 (what): convert words into structured numeric features.
- Layer 2 (why): algorithms operate on numbers and distances/probabilities.
- Layer 3 (goal): preserve useful meaning while reducing noise.

Why it matters:
- Without this step, text ML cannot start.
- Bad preprocessing can hurt a good model more than weak model choice.

Quick exercise:
- In one sentence, explain why "raw text -> numbers" is necessary in classic ML.

#### 3) Tokenization and Vocabulary
Tokenization means splitting text into smaller units, usually words.

Example sentence:
- "Delivery was very fast"

Possible tokens:
- ["delivery", "was", "very", "fast"]

Vocabulary means the full set of unique tokens in your dataset.

Tiny corpus:
1. "fast delivery"
2. "slow support"
3. "fast support"

Vocabulary:
- ["delivery", "fast", "slow", "support"]

Why it matters:
- Tokens are the building blocks of text features.
- Vocabulary design affects feature size, sparsity, and model behavior.

Quick exercise:
- Tokenize: "Great support and quick refund"

#### 4) Bag-of-Words Intuition
Bag-of-words (BoW) counts how many times vocabulary words appear in each document.

Important simplification:
- BoW usually ignores word order.
- It keeps frequency information.

Example vocabulary:
- ["fast", "slow", "support"]

Sentence A: "fast support"
- [1, 0, 1]

Sentence B: "slow support support"
- [0, 1, 2]

Why it matters:
- BoW is simple, fast, and often a strong baseline for text classification.
- It gives you interpretable features before moving to more advanced methods.

Quick exercise:
- With vocabulary ["refund", "late", "great"], encode "late refund late" as counts.

#### 5) Stopwords, Casing, and Basic Cleaning
Common preprocessing choices:
1. Lowercasing
- "Great" and "great" become the same token.

2. Stopword removal
- Very common words like "the", "is", "and" may be removed.

3. Punctuation handling
- Keep or remove punctuation depending on task.

4. Numbers and symbols
- Sometimes useful (e.g., order IDs, prices), sometimes noise.

Beginner rule:
- Start with simple defaults.
- Change one preprocessing choice at a time.

Why it matters:
- Over-cleaning can remove useful signal.
- Under-cleaning can inflate noise and dimensionality.

Quick exercise:
- Give one case where removing numbers might hurt performance.

#### 6) TF-IDF in Plain Language
BoW counts words, but very common words can dominate.
TF-IDF reduces the weight of words that appear in almost every document and boosts words that are distinctive.

Name breakdown:
- TF = term frequency in one document
- IDF = inverse document frequency across all documents

Practical intuition:
- "the" appears everywhere -> low distinguishing power -> lower weight
- "chargeback" appears in fewer docs -> high distinguishing power -> higher weight

Why it matters:
- TF-IDF often improves text classification baselines versus raw counts.
- It keeps models simple while making features more informative.

Quick exercise:
- Which should get higher TF-IDF weight in most corpora: "the" or "chargeback"?

#### 7) Hands-On Mini Exercise
Goal: build BoW and TF-IDF feature matrices from tiny text samples.

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

texts = [
    "Delivery was fast and smooth",
    "Support was slow but helpful",
    "Fast refund and great support",
    "Late delivery and slow response",
]

# 1) Bag-of-words (counts)
count_vec = CountVectorizer(lowercase=True, stop_words="english")
X_count = count_vec.fit_transform(texts)

count_df = pd.DataFrame(
    X_count.toarray(),
    columns=count_vec.get_feature_names_out()
)
print("Bag-of-Words Features:")
print(count_df)

# 2) TF-IDF
tfidf_vec = TfidfVectorizer(lowercase=True, stop_words="english")
X_tfidf = tfidf_vec.fit_transform(texts)

tfidf_df = pd.DataFrame(
    X_tfidf.toarray(),
    columns=tfidf_vec.get_feature_names_out()
)
print("\nTF-IDF Features:")
print(tfidf_df.round(3))
```

What to notice:
- Same vocabulary can produce different feature values under BoW vs TF-IDF.
- TF-IDF downweights very frequent terms.
- Output matrices are sparse and high-dimensional even for small corpora.

#### 8) Common Beginner Mistakes in Text Preprocessing
- Splitting train/test after vectorizing the full dataset (leakage risk)
- Aggressive cleaning that removes useful signal
- Not storing vectorizer settings used in training
- Comparing models with different preprocessing and claiming "model improvement"
- Ignoring class imbalance in text labels

Safe pattern:
1. Split data first
2. Fit vectorizer on training text only
3. Transform validation/test using the training-fitted vectorizer

#### 9) BoW vs TF-IDF Visual

![Chapter 1 text feature map](images/manual_visuals_v2/chapter1_bow_tfidf_map.png)

What to notice:
- BoW highlights frequency.
- TF-IDF highlights discriminative terms.

#### 10) Chapter Checkpoint
Answer in your own words:
1. What is tokenization?
2. How is vocabulary different from one document's tokens?
3. Why can TF-IDF outperform raw counts?
4. What leakage risk appears if you fit vectorizer before splitting?

If you can answer these confidently, move to the next chapter.

---

## Chapter 2
### Text Classification: Turning Messages Into Decisions

#### 1) What You Will Learn
In this chapter, you will learn how to classify text into categories using beginner-friendly models and evaluation habits.

You already know classification from Volume 1.
Now you will apply that same logic to text features such as TF-IDF vectors.
The core upgrade is not only model training, but also error analysis on misclassified messages.

By the end, you should be able to:
1. Frame text tasks as classification problems
2. Build a Naive Bayes and logistic regression text classifier
3. Evaluate text classifiers with precision, recall, F1, and confusion matrix
4. Perform simple error analysis and identify improvement ideas

Pattern reminder:
- Concept
- Why it matters
- Quick exercise
- Visual aid

#### 2) What Is Text Classification?
Text classification means assigning a label to a text input.

Examples:
- Spam vs not spam
- Support ticket intent (billing, technical, account)
- News topic (sports, finance, politics)
- Urgency level (low, medium, high)

Layered view:
- Layer 1 (what): map text to one label from a fixed label set.
- Layer 2 (why): teams need fast, consistent routing and triage decisions.
- Layer 3 (goal): automate repetitive decisions while keeping acceptable error types.

Why it matters:
- Text classification is one of the most common NLP use cases in production.
- It often delivers value quickly because many organizations already have labeled historical messages.

Quick exercise:
- Is "predict support ticket department" a classification or regression task?

#### 3) Pipeline Overview: Text to Label
A simple text classification pipeline:
1. Collect labeled text examples
2. Split into train/validation/test
3. Convert text to numeric features (BoW or TF-IDF)
4. Train classifier
5. Evaluate using task-appropriate metrics
6. Analyze mistakes and improve

Important ordering rule:
- Split first.
- Fit vectorizer on training text only.
- Transform validation/test with the training-fitted vectorizer.

Why it matters:
- This is where many beginners accidentally leak information from test text into training artifacts.
- Correct ordering keeps evaluation honest.

Quick exercise:
- Which step should happen first: vectorizer fitting or train/test split?

#### 4) Naive Bayes Intuition for Text
Naive Bayes is a classic baseline for text classification.

Core idea:
- It estimates how strongly words are associated with each class.
- It assumes word contributions are conditionally independent (the "naive" assumption).

Tiny intuition:
- Words like "refund" and "charge" may increase probability of class "billing".
- Words like "error" and "crash" may increase probability of class "technical".

Why it often works well on text:
- Fast training
- Strong baseline performance
- Robust on sparse high-dimensional features

Why it matters:
- A strong baseline gives you a reliable starting point before trying more complex methods.

Quick exercise:
- Why might Naive Bayes be a good first model for small text datasets?

#### 5) Logistic Regression on TF-IDF
Logistic regression can also perform very well on text vectors.

What it does:
- Learns weights for features to separate classes.
- Produces probabilities that can be thresholded.

Practical differences vs Naive Bayes:
- Often stronger when features are informative and data is sufficient
- Easy to regularize to reduce overfitting
- Good interpretability via feature weights

Binary example:
- Predict spam if `P(spam) >= threshold`.

Why it matters:
- Logistic regression is one of the most reliable "workhorse" models for text classification.

Quick exercise:
- If false positives are very costly, should you usually raise or lower the decision threshold?

#### 6) Hands-On Mini Project: Support Ticket Intent Classifier
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Tiny synthetic dataset
data = pd.DataFrame({
  "text": [
    "I was charged twice for my order",
    "Payment failed and card was declined",
    "App crashes when I open settings",
    "The website shows server error",
    "Please reset my password",
    "I cannot log in to my account",
    "Need invoice copy for last month",
    "Refund has not arrived yet",
    "Video call quality is very poor",
    "My account email needs to be changed",
    "Unable to update payment method",
    "App freezes after update",
  ],
  "label": [
    "billing",
    "billing",
    "technical",
    "technical",
    "account",
    "account",
    "billing",
    "billing",
    "technical",
    "account",
    "billing",
    "technical",
  ]
})

X_train, X_test, y_train, y_test = train_test_split(
  data["text"],
  data["label"],
  test_size=0.25,
  random_state=42,
  stratify=data["label"]
)

# Model 1: Multinomial Naive Bayes
nb_pipeline = Pipeline([
  ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
  ("clf", MultinomialNB())
])
nb_pipeline.fit(X_train, y_train)
nb_preds = nb_pipeline.predict(X_test)

print("Naive Bayes F1 (macro):", round(f1_score(y_test, nb_preds, average="macro"), 3))
print("Naive Bayes Confusion Matrix:\n", confusion_matrix(y_test, nb_preds))
print("Naive Bayes Report:\n", classification_report(y_test, nb_preds))

# Model 2: Logistic Regression
lr_pipeline = Pipeline([
  ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
  ("clf", LogisticRegression(max_iter=1000))
])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

print("Logistic Regression F1 (macro):", round(f1_score(y_test, lr_preds, average="macro"), 3))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, lr_preds))
print("Logistic Regression Report:\n", classification_report(y_test, lr_preds))
```

What to notice:
- The pipeline bundles vectorization + classifier safely.
- Macro-F1 is useful for balanced attention across classes.
- Confusion matrix reveals which labels are most often mixed.

#### 7) Metrics for Text Classification
Use the same core metrics from Volume 1, but watch class distribution closely.

Common choices:
- Accuracy: useful, but can hide minority-class failure.
- Macro-F1: treats all classes equally, good for imbalanced intent classes.
- Weighted-F1: weights classes by frequency.
- Per-class precision/recall: shows which class behavior is weak.

Mini example:
- If "billing" is 70% of data, a model can look strong while failing "technical" and "account".

Why it matters:
- Evaluation must match business cost of errors.
- Routing systems often care about minority classes just as much as majority classes.

Quick exercise:
- Which metric is usually safer when you want equal focus on all classes: accuracy or macro-F1?

#### 8) Error Analysis: Where the Real Learning Happens
After metrics, inspect mistakes manually.

Simple error analysis checklist:
1. Which class pairs are most confused?
2. Are labels inconsistent in training data?
3. Are short/ambiguous texts causing most errors?
4. Are domain-specific words missing from training samples?

Tiny example insight:
- "Unable to update payment method" could be labeled "billing" or "account" depending on team rules.

That means you may need label-guideline cleanup, not just a different model.

Why it matters:
- Many text-model improvements come from data/label quality, not algorithm changes.

Quick exercise:
- Name one non-model fix that could improve text classifier performance.

#### 9) Common Beginner Mistakes in Text Classification
- Fitting TF-IDF on full dataset before splitting
- Using only accuracy and ignoring per-class metrics
- Comparing models with different preprocessing settings unfairly
- Treating inconsistent human labels as if they were perfect ground truth
- Skipping error analysis and jumping straight to model complexity

Practical rule:
- Improve labels and preprocessing before reaching for a larger model.

#### 10) Confusion Matrix and Class Report Visual

![Chapter 2 text confusion matrix](images/manual_visuals_v2/chapter2_text_confusion_matrix.png)

What to notice:
- Diagonal cells show correct routing.
- Off-diagonal cells indicate where category boundaries are weak.

#### 11) Chapter Checkpoint
Answer in your own words:
1. Why is text classification still a classification problem even though input is unstructured?
2. Why is split order critical before vectorization?
3. When can macro-F1 be more useful than accuracy?
4. Why should you inspect misclassified samples manually?

If you can answer these confidently, move to the next chapter.

---

## Chapter 3
### Sentiment Analysis: Measuring Tone From Text

#### 1) What You Will Learn
In this chapter, you will learn how to classify the emotional tone of text using two beginner-friendly approaches.

Sentiment analysis is one of the fastest ways to turn customer text into actionable signals.
You will learn both a rule-based approach and a supervised ML approach.
This helps you choose the right method based on data availability and business constraints.

By the end, you should be able to:
1. Explain sentiment analysis in plain language
2. Compare lexicon-based and ML-based sentiment methods
3. Build a basic sentiment classifier workflow
4. Evaluate results and understand common failure modes

Pattern reminder:
- Concept
- Why it matters
- Quick exercise
- Visual aid

#### 2) What Is Sentiment Analysis?
Sentiment analysis is the task of labeling text tone, usually as positive, negative, or neutral.

Examples:
- Product reviews
- App store comments
- Support chat transcripts
- Social media mentions

Layered view:
- Layer 1 (what): detect tone category from text.
- Layer 2 (why): large text volumes are too big for manual review.
- Layer 3 (goal): summarize customer attitude trends and prioritize action.

Why it matters:
- Teams use sentiment trends for product quality monitoring, brand tracking, and support escalation.
- It creates a compact signal from noisy language data.

Quick exercise:
- Is "The app keeps crashing after login" likely positive, negative, or neutral?

#### 3) Method 1: Lexicon-Based Sentiment
Lexicon-based sentiment uses predefined word dictionaries with positive/negative scores.

Simple idea:
- Positive words (great, smooth, love) increase positive score.
- Negative words (slow, broken, hate) increase negative score.

How it typically works:
1. Tokenize text
2. Look up words in sentiment lexicon
3. Sum scores
4. Convert final score to label

Strengths:
- No labeled training data needed
- Quick baseline
- Easy to explain

Limitations:
- Struggles with sarcasm and context
- Domain words may be missing from generic lexicons

Why it matters:
- Useful when you need a quick start and do not have labeled sentiment data.

Quick exercise:
- Why might the sentence "This is sick" be hard for lexicon methods?

#### 4) Method 2: ML-Based Sentiment
ML-based sentiment trains a classifier on labeled sentiment examples.

Typical workflow:
1. Collect labeled reviews/messages
2. Split data
3. Vectorize text (TF-IDF)
4. Train classifier (Naive Bayes or logistic regression)
5. Evaluate metrics and confusion matrix

Strengths:
- Learns domain language patterns
- Usually better than generic lexicons with enough quality labels

Limitations:
- Requires labeled data
- Label consistency strongly affects quality

Why it matters:
- In production settings with domain data, ML-based sentiment often gives better reliability.

Quick exercise:
- If a business has 100,000 labeled reviews, which approach is usually stronger: lexicon-only or ML-based?

#### 5) Lexicon vs ML: When to Use Which
Use lexicon-first when:
- You need fast prototype results
- No labeled data is available
- Explainability and setup speed are top priorities

Use ML-first when:
- You have enough labeled examples
- Domain language is specialized
- You need better long-term performance

Practical beginner strategy:
1. Start with lexicon baseline
2. Train ML baseline
3. Compare on the same test split
4. Keep the method that is more reliable for your use case

Why it matters:
- Method choice should be evidence-based, not trend-based.

Quick exercise:
- In one sentence, describe a case where lexicon baseline is the right first step.

#### 6) Hands-On Mini Project: Review Sentiment Baseline
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Tiny synthetic sentiment dataset
data = pd.DataFrame({
  "text": [
    "The update is fantastic and very smooth",
    "Great value and excellent customer support",
    "I love this app and use it daily",
    "The service is okay, nothing special",
    "Average experience, could be better",
    "It works, but setup was confusing",
    "Terrible update, app crashes constantly",
    "Very slow and frustrating experience",
    "Support never replied, extremely disappointed",
    "Billing issue still unresolved",
    "Neutral experience overall",
    "Not bad, not great",
  ],
  "sentiment": [
    "positive",
    "positive",
    "positive",
    "neutral",
    "neutral",
    "neutral",
    "negative",
    "negative",
    "negative",
    "negative",
    "neutral",
    "neutral",
  ]
})

X_train, X_test, y_train, y_test = train_test_split(
  data["text"],
  data["sentiment"],
  test_size=0.25,
  random_state=42,
  stratify=data["sentiment"]
)

pipeline = Pipeline([
  ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
  ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print("Macro F1:", round(f1_score(y_test, preds, average="macro"), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
print("Classification Report:\n", classification_report(y_test, preds))
```

What to notice:
- Neutral class often causes confusion.
- Macro-F1 is useful for balanced attention across positive/neutral/negative labels.
- Manual review of mistakes is critical before claiming success.

#### 7) Sentiment Pitfalls You Should Expect
Common challenges:
- Sarcasm ("Amazing, it crashed again")
- Mixed sentiment in one sentence
- Domain-specific words ("hot" in gaming can be positive)
- Negation handling ("not good")
- Label inconsistency between annotators

Why this happens:
- Sentiment is contextual and sometimes subjective.

Practical mitigation:
1. Define clear labeling rules
2. Keep ambiguous examples in a review bucket
3. Track confusion between neutral and other classes

Why it matters:
- Many sentiment failures are data-definition problems, not model problems.

Quick exercise:
- Write one ambiguous sentence that could be labeled neutral or negative depending on policy.

#### 8) Thresholds and Class Imbalance in Sentiment
Even in sentiment tasks, class imbalance is common.

Example:
- 80% positive reviews
- 15% neutral
- 5% negative

A model can look strong overall while missing the minority class you care about most.

If the business cares about catching negative feedback early:
- Prioritize recall for negative class
- Consider threshold tuning and class weights

Why it matters:
- Business cost is often tied to minority sentiment classes (critical complaints, churn risk).

Quick exercise:
- If missing negative reviews is costly, should you optimize mostly for negative recall or overall accuracy?

#### 9) Common Beginner Mistakes in Sentiment Projects
- Treating sentiment labels as objective truth without annotation rules
- Evaluating only accuracy in imbalanced datasets
- Ignoring neutral class because it is hard
- Using a generic lexicon in highly domain-specific language without adaptation
- Skipping manual error inspection

Practical rule:
- Use both metrics and sample-level review before deployment decisions.

#### 10) Sentiment Distribution + Error Breakdown Visual

![Chapter 3 sentiment overview](images/manual_visuals_v2/chapter3_sentiment_overview.png)

What to notice:
- Class distribution affects metric interpretation.
- Neutral-vs-negative confusion is usually a high-priority review area.

#### 11) Chapter Checkpoint
Answer in your own words:
1. What is the difference between lexicon-based and ML-based sentiment?
2. Why can neutral class be difficult?
3. Why might macro-F1 be preferred over accuracy in multi-class sentiment?
4. Name one non-model change that can improve sentiment quality.

If you can answer these confidently, move to the next chapter.

---

## Chapter 4
### Word Embeddings: Meaning Beyond Word Counts

#### 1) What You Will Learn
In this chapter, you will learn why simple word counts are sometimes not enough, and how embeddings provide a richer representation of language.

In Chapter 1 and Chapter 2, you used BoW and TF-IDF.
Those methods are still valuable, but they treat words mostly as separate tokens.
Embeddings add a new idea: represent words as numeric coordinates where similar meanings are close together.

By the end, you should be able to:
1. Explain what a word embedding is in plain language
2. Understand why embeddings can capture semantic similarity better than BoW
3. Use a pretrained embedding model for simple similarity tasks
4. Decide when TF-IDF is enough and when embeddings are worth trying

Pattern reminder:
- Concept
- Why it matters
- Quick exercise
- Visual aid

#### 2) Why Bag-of-Words Has Limits
BoW and TF-IDF are strong baselines, but they have limits:
- They mostly ignore word order and context
- They treat words as independent tokens
- They do not naturally understand that "refund" and "reimbursement" are related

Tiny example:
- Sentence A: "The app is excellent"
- Sentence B: "The app is great"

BoW sees different tokens (`excellent` vs `great`).
A semantic method should recognize these are similar in meaning.

Why it matters:
- If your task depends on synonym or semantic similarity, token counts may miss important signal.

Quick exercise:
- Give two different words that often mean similar things in customer feedback.

#### 3) What Is a Word Embedding?
A word embedding maps each word to a numeric vector.

Simple intuition:
- Each word becomes a point in a high-dimensional space.
- Words used in similar contexts appear near each other.

So instead of:
- "refund" = just a token ID

You have:
- "refund" = [0.12, -0.44, 0.81, ...]

Important idea:
- Distance between vectors roughly reflects semantic similarity.

Why it matters:
- Embeddings let models use similarity structure rather than only exact token matches.

Quick exercise:
- In embedding space, should "good" and "great" usually be closer or farther apart than "good" and "server"?

#### 4) How Embeddings Are Learned
Classic embedding methods (like Word2Vec) learn vectors by predicting nearby words.

Plain-language process:
1. Read many sentences
2. Learn which words appear in similar neighborhoods
3. Adjust vectors so context-similar words become closer

You do not need to train embeddings from scratch to benefit.
Beginners usually start with pretrained vectors learned on large corpora.

Why it matters:
- Pretrained embeddings save time and often work well with limited project data.

Quick exercise:
- Why might pretrained embeddings help when your labeled dataset is small?

#### 5) Word-Level vs Document-Level Representations
Embeddings are often word-level vectors, but many tasks need a vector for a whole sentence or document.

Simple beginner approach:
- Average word vectors in the sentence to get one document vector.

Tradeoff:
- Fast and simple
- Loses some word-order nuance

Practical note:
- TF-IDF + logistic regression is often still a very strong baseline.
- Embeddings are an additional tool, not an automatic replacement.

Why it matters:
- Choosing representation affects both quality and complexity.

Quick exercise:
- If you need a quick baseline for text classification, which should you usually try first: TF-IDF or custom embedding training?

#### 6) Hands-On Mini Lab: Similar Words With Pretrained Vectors
```python
import gensim.downloader as api

# Load a small pretrained embedding model
# First run may download files from the internet.
model = api.load("glove-wiki-gigaword-50")

queries = ["good", "refund", "slow", "account"]

for q in queries:
  if q in model.key_to_index:
    print(f"\nTop similar words for '{q}':")
    for word, score in model.most_similar(q, topn=5):
      print(f"  {word:15s} similarity={score:.3f}")
  else:
    print(f"Word not found in vocabulary: {q}")
```

What to notice:
- Similar words are grouped by usage patterns, not strict dictionary rules.
- Results are useful but imperfect.
- Domain-specific language may still need domain-tuned representations.

#### 7) Embeddings for Downstream Tasks
Common uses:
- Text similarity search
- Query expansion
- Better features for sentiment/topic tasks

Simple workflow for classification with averaged embeddings:
1. Convert each sentence to a vector (mean of token vectors)
2. Train a classifier on these vectors
3. Compare against TF-IDF baseline

Evaluation rule:
- Always compare on the same split and same metric.

Why it matters:
- Embeddings should be adopted when they improve measurable outcomes, not because they sound more advanced.

Quick exercise:
- Why is it important to compare TF-IDF and embedding-based approaches on the same test set?

#### 8) Common Beginner Mistakes With Embeddings
- Assuming embeddings always beat TF-IDF
- Ignoring out-of-vocabulary words
- Skipping baseline comparison
- Mixing datasets/splits between experiments
- Treating nearest-neighbor outputs as perfect semantic truth

Practical rule:
- Keep TF-IDF baseline results documented before adding embedding complexity.

#### 9) TF-IDF vs Embeddings: Practical Decision Checklist
Start with TF-IDF when:
- You need fast baseline performance
- Dataset is small/medium
- Interpretability and simplicity matter

Try embeddings when:
- Synonyms/context differences matter a lot
- You need semantic search/similarity behavior
- Baseline misses meaning-level relationships

Why it matters:
- Method choice should match task constraints, not trends.

#### 10) Embedding Space Visual

![Chapter 4 embedding space map](images/manual_visuals_v2/chapter4_embedding_space.png)

What to notice:
- Semantically similar words form local neighborhoods.
- Distances are informative, but not perfect.

#### 11) Chapter Checkpoint
Answer in your own words:
1. What problem do embeddings solve that BoW/TF-IDF can miss?
2. Why are pretrained embeddings useful for beginners?
3. Why should TF-IDF still be kept as baseline?
4. Name one case where embeddings are especially helpful.

If you can answer these confidently, move to the next chapter.

---

---

## Chapter 5: Time-Series Basics

### Time-Ordered Data and Why It Is Different

#### 1) What You Will Learn

By the end of this chapter you will be able to:

1. Explain what makes time-series data structurally different from ordinary tabular rows.
2. Decompose a series into trend, seasonality, and noise.
3. Engineer lag features and rolling statistics so a standard ML model can consume time data.
4. Perform a time-aware train/test split that prevents look-ahead leakage.

---

### 5.1 What Is a Time Series?
A **time series** is a sequence of measurements recorded at regular intervals — daily sales, hourly temperature, monthly churn counts.

The critical difference from the datasets in Volume 1: **the order of rows matters**. In a customer table you can shuffle the rows and the model still learns the same patterns. Shuffle a time series and you destroy the information.

**Layered view:**

- **Layer 1 (plain English):** Numbers listed in time order, like a diary with one entry per day.
- **Layer 2 (practitioner):** An indexed sequence $y_1, y_2, \dots, y_T$ where the index $t$ carries meaning — past values can be used to predict future values.
- **Layer 3 (math hint):** The goal is often to model $\hat{y}_{t+h} = f(y_t, y_{t-1}, \dots)$ for some forecast horizon $h$.

**Why it matters:** Retail, finance, healthcare, manufacturing, and operations all have time-series problems. Predicting demand, detecting equipment failures, and forecasting traffic are everyday tasks that assume you understand time order.

**Quick exercise:** Think of three numbers you track over time in your own life (steps per day, monthly spending, weekly hours worked). Which one changes smoothest? Which one looks most random?

---

### 5.2 Trend, Seasonality, and Noise
Every time series can be thought of as the sum of three components:

| Component | Plain-English Definition | Example |
|---|---|---|
| **Trend** | Long-term rise or fall | Revenue growing 5 % each year |
| **Seasonality** | Repeating pattern at fixed intervals | Toy sales spike every December |
| **Noise** | Random day-to-day fluctuation | A Tuesday that was 2 % above average for no clear reason |

Classical decomposition splits the observed series $y_t$ into:

$$y_t = \text{Trend}_t + \text{Seasonal}_t + \text{Noise}_t$$

**Layered view:**

- **Layer 1:** Your series is a mix of a general direction, a repeating rhythm, and random wobble.
- **Layer 2:** Separating these components helps you choose the right model and diagnose problems. A series with strong seasonality needs seasonal-aware methods.
- **Layer 3:** Statisticians use STL decomposition (Seasonal-Trend decomposition using LOESS) to estimate each component robustly.

**Why it matters:** Misreading noise as trend is one of the most common business mistakes — "sales went up three days in a row, we're on a roll!" Decomposition keeps you honest.

**Quick exercise:** Draw or imagine a line chart of weekly coffee shop sales. Sketch where trend, weekly seasonality (busy weekends), and noise would appear in that chart.

---

### 5.3 Lag Features and Rolling Statistics
Standard ML models like `RandomForestClassifier` or `LinearRegression` expect a flat table of features. A raw time series is just one column of numbers — we need to engineer features that capture recent history.

**Lag features** copy past values into new columns:

| date | sales | lag_1 | lag_7 |
|---|---|---|---|
| 2024-01-08 | 120 | 115 | 108 |
| 2024-01-09 | 118 | 120 | 110 |

`lag_1` = yesterday's sales. `lag_7` = same day last week.

**Rolling statistics** summarise a sliding window:

| date | sales | rolling_mean_7 | rolling_std_7 |
|---|---|---|---|
| 2024-01-08 | 120 | 112.4 | 5.2 |

**Layered view:**

- **Layer 1:** Instead of "today's number," give the model "today's plus the last six days' numbers."
- **Layer 2:** Lag and window features are how you let a regression model learn temporal patterns without writing a custom time-series algorithm.
- **Layer 3:** Choosing which lags to include requires domain knowledge or a correlation analysis using the autocorrelation function (ACF plot).

**Why it matters:** This technique lets you use every model from Volume 1 — linear regression, random forests, gradient boosting — on time data with minimal changes. It is the most practical bridge between "regular ML" and "time-series ML."

**Quick exercise:** If you wanted to predict tomorrow's ridership on a bus route, which lag features would you add? (Hint: think about weekly patterns.)

---

### 5.4 Time-Aware Train/Test Split
**The single most important rule of time-series ML: never shuffle.**

In Volume 1, you used `train_test_split(X, y, random_state=42)`. That randomly picks rows for train and test. With time-series data, this creates **look-ahead leakage** — your model trains on data from the future and tests on data from the past, making your validation numbers meaninglessly optimistic.

The correct approach: pick a **cutoff date** and put everything before it in the training set, everything after in the test set.

```
Timeline:  ─────────────────────────────────────────────────────▶
Training              [Jan 2022 ──────────── Dec 2023]
Test                                                  [Jan 2024 ── Jun 2024]
```

**Layered view:**

- **Layer 1:** Train on old data, test on newer data. Respect chronological order.
- **Layer 2:** This mimics the real deployment scenario — your model only ever sees past data when making predictions.
- **Layer 3:** For hyperparameter tuning, use **walk-forward (rolling) cross-validation** rather than standard k-fold. `TimeSeriesSplit` in scikit-learn implements this.

**Why it matters:** Using random split on time-series data is one of the most dangerous common mistakes in applied ML. Models appear to work brilliantly in development and fail immediately in production.

**Quick exercise:** You have two years of daily sales data (730 rows). If you want a 20 % test set, what date does the cutoff fall on?

---

### 5.5 Hands-On: Building a Time-Series Feature Table

Let us create a realistic scenario: a fictional coffee shop tracking daily sales. We will build lag and rolling features, perform a proper train/test split, and fit a simple linear regression.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ── 1. Generate synthetic daily sales data ──────────────────────────────────
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=365, freq="D")

# Base trend + weekly seasonality + noise
trend     = np.linspace(100, 130, 365)
weekly    = 15 * np.sin(2 * np.pi * np.arange(365) / 7)  # 7-day cycle
noise     = np.random.normal(0, 5, 365)
sales     = trend + weekly + noise

df = pd.DataFrame({"date": dates, "sales": sales})
df = df.set_index("date")

# ── 2. Engineer lag and rolling features ────────────────────────────────────
df["lag_1"]  = df["sales"].shift(1)   # yesterday
df["lag_7"]  = df["sales"].shift(7)   # same day last week
df["roll_7_mean"] = df["sales"].shift(1).rolling(7).mean()  # shift FIRST to avoid leakage
df["roll_7_std"]  = df["sales"].shift(1).rolling(7).std()

# Drop rows where features are NaN (first 7 days have incomplete windows)
df = df.dropna()

print("Feature table (first 5 rows):")
print(df.head())
print(f"\nShape after feature engineering: {df.shape}")

# ── 3. Time-aware train/test split ──────────────────────────────────────────
cutoff = "2023-10-01"
train = df.loc[:cutoff]
test  = df.loc[cutoff:]

feature_cols = ["lag_1", "lag_7", "roll_7_mean", "roll_7_std"]
X_train, y_train = train[feature_cols], train["sales"]
X_test,  y_test  = test[feature_cols],  test["sales"]

print(f"\nTraining rows : {len(train)}")
print(f"Test rows     : {len(test)}")

# ── 4. Fit linear regression ─────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nTest MAE: {mae:.2f} units")

# ── 5. Plot actual vs predicted ──────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.plot(y_test.index, y_test.values, label="Actual",    linewidth=1.5)
plt.plot(y_test.index, y_pred,        label="Predicted", linewidth=1.5, linestyle="--")
plt.title("Daily Sales: Actual vs Predicted (Test Period)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("images/manual_visuals_v2/ch05_actual_vs_predicted.png", dpi=150)
plt.show()
```

**What the code does, step by step:**

1. **Generates synthetic data** — trend + seasonality + noise closely mirrors real retail data.
2. **Engineers lag features** — each row now tells the model "what did sales look like yesterday and last week?"
3. **Applies `.shift(1)` before `.rolling()`** — this is the key leakage-prevention pattern; you must shift first so the rolling window never includes the current day's sales.
4. **Splits on a cutoff date** — training months precede test months with no overlap.
5. **Fits plain `LinearRegression`** — proof that Volume 1 models can handle time data with good feature engineering.
6. **Reports MAE** — a natural error metric for forecasting: "on average, my prediction is off by X units."

> **Visual placeholder — Figure 5.1:** Line chart with two traces (blue = actual, orange dashed = predicted) over the test period. X-axis: date. Y-axis: sales. Caption: "Linear regression with lag features tracks the weekly rhythm reasonably well."

---

### 5.6 Common Beginner Mistakes

| Mistake | Why it is dangerous | Fix |
|---|---|---|
| Random shuffle before split | Creates look-ahead leakage; inflated test scores | Always split on a cutoff date |
| Forgetting to `.shift()` before rolling | Today's value leaks into the rolling mean | Shift the series before computing windows |
| Using future column values as features | Classic data leakage | Only use columns available before prediction time |
| Treating all seasonality as noise | Removes useful signal | Plot the series first; check for weekly/monthly patterns |
| Expecting perfect smoothness | Noise is real — overfitting to it hurts generalisation | Keep models simple; use MAE not just R² |

---

### Chapter 5 Checkpoint

Answer these questions before moving on:

1. Why is it wrong to use `train_test_split` with `shuffle=True` on time-series data?
2. What is the difference between a lag feature and a rolling-mean feature?
3. Your series has a clear dip every Sunday. Which component (trend/seasonality/noise) does this represent?
4. You forgot to call `.shift(1)` before computing `rolling(7).mean()`. What problem does this introduce?

If you can answer these confidently, move to the next chapter.

---

## Next Writing Step
- Chapter 6: Forecasting Models

---

## Chapter 6: Forecasting Models

### From Time-Series Features to Practical Forecasts

#### 1) What You Will Learn

By the end of this chapter you will be able to:

1. Build and evaluate strong baseline forecasting models before trying complex ones.
2. Choose forecasting metrics (MAE, RMSE, MAPE) based on business needs.
3. Explain when to use classical forecasting models vs feature-based ML models.
4. Compare multiple forecasting approaches with a fair, time-aware validation setup.

---

### 6.1 Start with Baselines
A **baseline forecast** is a simple prediction rule that your advanced model must beat.

Common baselines:

| Baseline | Rule | Good for |
|---|---|---|
| Last value (naive) | Predict tomorrow as today's value | Stable series, quick benchmark |
| Seasonal naive | Predict this Monday as last Monday | Strong weekly/monthly cycles |
| Moving average | Predict as average of last k points | Noisy series smoothing |

**Layered view:**

- **Layer 1 (plain English):** Before building a fancy model, ask: "Can a trivial rule already do well?"
- **Layer 2 (practitioner):** Baselines expose whether your problem is actually hard. If your advanced model barely beats naive, you may not need complexity.
- **Layer 3 (math hint):** For naive forecast, $\hat{y}_{t+1} = y_t$. For seasonal naive with period $s$, $\hat{y}_{t+h} = y_{t+h-s}$.

**Why it matters:** Teams often celebrate an MAE of 8.2 without context. If naive MAE is 8.0, the "improvement" is negative.

**Quick exercise:** Your daily demand has strong weekly seasonality. Which baseline is more appropriate: last value or seasonal naive with period 7?

---

### 6.2 Forecast Error Metrics
Forecasting is judged by how far predictions are from reality. The three core metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | $\frac{1}{n}\sum |y_i-\hat{y}_i|$ | Average absolute error in original units |
| RMSE | $\sqrt{\frac{1}{n}\sum (y_i-\hat{y}_i)^2}$ | Penalises large misses more strongly |
| MAPE | $\frac{100}{n}\sum \left|\frac{y_i-\hat{y}_i}{y_i}\right|$ | Average percent error |

**Layered view:**

- **Layer 1:** MAE says "on average we miss by X units." RMSE says "big misses hurt extra." MAPE says "error as percent."
- **Layer 2:** Use MAE for business communication, RMSE when outliers are costly, MAPE when scale changes across products or stores.
- **Layer 3:** MAPE is unstable when actual values are near zero. Use sMAPE or MAE in that case.

**Why it matters:** Picking the wrong metric can optimize the wrong behavior. A model with lower RMSE may still produce worse median day-to-day planning quality than one with lower MAE.

**Quick exercise:** If stock-outs are extremely expensive on a few peak days, which metric should you pay extra attention to: MAE or RMSE?

---

### 6.3 Forecasting Model Families
There are two practical families you should know now:

1. **Classical time-series models** (Exponential Smoothing, ARIMA)
2. **Feature-based ML models** (Linear Regression, Random Forest, Gradient Boosting on lag features)

#### Classical models

- **Exponential Smoothing (ETS):** Works well for level/trend/seasonality patterns and is strong on many business series.
- **ARIMA/SARIMA:** Models autocorrelation directly and supports seasonal components.

#### Feature-based ML models

- Build lag and rolling features (Chapter 5).
- Add calendar features (day of week, month, holiday flags).
- Train standard regressors from Volume 1.

**Layered view:**

- **Layer 1:** Classical models learn from the series structure directly; ML models learn from engineered feature columns.
- **Layer 2:** Classical methods are often strong with less feature engineering. ML methods are flexible when you have extra predictors (price, promo, weather, traffic).
- **Layer 3:** In practice, hybrid strategies often win: classical baseline + ML residual modeling.

**Why it matters:** The right family depends on your data and operations. If you only have one clean series, classical methods can be excellent. If you have rich covariates, ML models usually gain an edge.

**Quick exercise:** You are forecasting sales and you also know future planned discounts. Which family likely benefits more from that extra predictor?

---

### 6.4 Validation That Matches Reality
Forecasting validation should mimic production use.

Two common setups:

| Setup | How it works | Use case |
|---|---|---|
| Holdout split | Train on old period, test on newest block | Quick baseline comparison |
| Walk-forward validation | Re-train as window moves forward through time | More realistic model selection |

Walk-forward example with monthly data:

1. Train Jan-Dec, predict Jan next year
2. Train Jan-Jan, predict Feb
3. Train Jan-Feb, predict Mar

This yields multiple out-of-sample errors and a more reliable estimate.

**Layered view:**

- **Layer 1:** Keep stepping forward in time, always predicting the future from the past.
- **Layer 2:** This estimates how model quality changes as conditions drift.
- **Layer 3:** Use `TimeSeriesSplit` or custom expanding-window loops to reproduce deployment.

**Why it matters:** One lucky test period can mislead you. Walk-forward validation reduces luck and gives a distribution of errors, not just one number.

**Quick exercise:** Why is random k-fold cross-validation usually inappropriate for forecasting tasks?

---

### 6.5 Hands-On: Compare Three Forecasting Approaches

In this lab we compare:

1. Naive baseline
2. Linear regression on lag features
3. Holt-Winters Exponential Smoothing

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# 1) Synthetic daily demand with trend + weekly seasonality
np.random.seed(7)
n = 420
dates = pd.date_range("2023-01-01", periods=n, freq="D")
trend = np.linspace(200, 250, n)
season = 20 * np.sin(2 * np.pi * np.arange(n) / 7)
noise = np.random.normal(0, 6, n)
demand = trend + season + noise

df = pd.DataFrame({"date": dates, "demand": demand}).set_index("date")

# 2) Holdout split (last 90 days as test)
train = df.iloc[:-90].copy()
test = df.iloc[-90:].copy()

# 3) Naive forecast: y_hat_t = y_{t-1}
naive_pred = test["demand"].shift(1)
naive_pred.iloc[0] = train["demand"].iloc[-1]

# 4) Linear regression using lag features
full = df.copy()
full["lag_1"] = full["demand"].shift(1)
full["lag_7"] = full["demand"].shift(7)
full["roll_7_mean"] = full["demand"].shift(1).rolling(7).mean()
full = full.dropna()

train_ml = full.iloc[:-90]
test_ml = full.iloc[-90:]

X_train = train_ml[["lag_1", "lag_7", "roll_7_mean"]]
y_train = train_ml["demand"]
X_test = test_ml[["lag_1", "lag_7", "roll_7_mean"]]
y_test = test_ml["demand"]

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 5) Holt-Winters (additive trend + additive weekly seasonality)
hw = ExponentialSmoothing(
  train["demand"],
  trend="add",
  seasonal="add",
  seasonal_periods=7
).fit()
hw_pred = hw.forecast(len(test))

# 6) Evaluation helper
def report(name, y_true, y_pred):
  mae = mean_absolute_error(y_true, y_pred)
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  print(f"{name:20s} MAE={mae:6.2f} RMSE={rmse:6.2f}")

print("Forecast comparison on test period:")
report("Naive", test["demand"], naive_pred)
report("LinearRegression", y_test, lr_pred)
report("HoltWinters", test["demand"], hw_pred)

# 7) Plot test forecasts
plt.figure(figsize=(12, 4))
plt.plot(test.index, test["demand"], label="Actual", linewidth=1.6)
plt.plot(test.index, naive_pred, label="Naive", linestyle="--")
plt.plot(test_ml.index, lr_pred, label="LinearRegression", linestyle=":")
plt.plot(test.index, hw_pred, label="HoltWinters", linestyle="-.")
plt.title("Forecast Model Comparison (Test Window)")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.tight_layout()
plt.savefig("images/manual_visuals_v2/ch06_model_comparison.png", dpi=150)
plt.show()
```

**What the code does, step by step:**

1. Creates a realistic series with weekly seasonality.
2. Reserves the most recent 90 days as test data.
3. Builds a naive baseline for context.
4. Trains a lag-feature linear regression model.
5. Trains Holt-Winters as a classical seasonal model.
6. Compares MAE and RMSE side by side.
7. Plots all predictions on the same chart for visual inspection.

> **Visual placeholder - Figure 6.1:** Four lines over the same test window (actual, naive, linear regression, Holt-Winters). Caption: "Different forecasting families can be compared fairly when evaluated on the same future period."

---

### 6.6 Common Beginner Mistakes

| Mistake | Why it is dangerous | Fix |
|---|---|---|
| Skipping baselines | You cannot tell if your model is truly useful | Always report naive and seasonal naive first |
| Reporting one metric only | Hides trade-offs (small typical errors vs big misses) | Report MAE and RMSE together |
| Tuning on test period repeatedly | Test set becomes training-by-feedback | Keep a final untouched test block |
| Ignoring seasonality in model choice | Forecasts miss repeating patterns | Include seasonal lags or seasonal models |
| Comparing models on different time windows | Unfair comparison | Use the exact same test dates for all models |

---

### Chapter 6 Checkpoint

Answer these questions before moving on:

1. Why should every forecasting project begin with at least one baseline?
2. When would RMSE be preferred over MAE?
3. What is one major advantage of walk-forward validation over a single holdout split?
4. You have known future promotional events as inputs. Which model family often benefits most and why?

If you can answer these confidently, move to the next chapter.

---

## Next Writing Step
- Chapter 7: Recommendation Systems

---

## Chapter 7: Recommendation Systems

### Recommending the Right Thing to the Right User

#### 1) What You Will Learn

By the end of this chapter you will be able to:

1. Explain the difference between content-based and collaborative filtering recommenders.
2. Build a simple user-item matrix and generate item recommendations from interaction data.
3. Understand the cold-start problem and practical mitigation strategies.
4. Evaluate recommendation quality with ranking-focused metrics.

---

### 7.1 What Is a Recommendation System?
A **recommendation system** predicts what a user is likely to click, buy, watch, or read next.

Unlike standard classification in Volume 1 ("spam or not spam"), recommendation is usually a **ranking** problem: out of many candidate items, which few should appear at the top?

**Layered view:**

- **Layer 1 (plain English):** You are building a smart shortlist generator.
- **Layer 2 (practitioner):** Given user history and item information, estimate preference scores and rank items by score.
- **Layer 3 (math hint):** Recommendation often optimizes top-k ranking quality, where you care about $P(i \mid u)$ for each item $i$ and user $u$.

**Why it matters:** Recommendations drive major business outcomes: conversion, retention, session time, and basket size.

**Quick exercise:** Name two products you use where recommendations clearly influence what you consume next.

---

### 7.2 Content-Based vs Collaborative Filtering
Two core approaches dominate practical recommender systems:

| Approach | Uses | Strength | Weakness |
|---|---|---|---|
| **Content-based** | Item features (genre, tags, text, price) + user profile | Works even with few users | Can over-specialize and miss discovery |
| **Collaborative filtering** | User-item interaction patterns | Captures hidden taste patterns without manual features | Suffers with new users/items (cold start) |

#### Content-based intuition

If a user liked sci-fi books, recommend other items with similar content vectors.

#### Collaborative filtering intuition

If users A and B liked many of the same items, items liked by A but unseen by B are good candidates for B.

**Layered view:**

- **Layer 1:** Content-based matches item similarity; collaborative filtering matches behavior similarity.
- **Layer 2:** Content-based needs useful item metadata. Collaborative filtering needs enough interaction volume.
- **Layer 3:** Matrix factorization approximates user-item preference matrix $R$ with low-rank factors $U V^T$.

**Why it matters:** Choosing the wrong recommendation family for your data is one of the most expensive architecture mistakes in product teams.

**Quick exercise:** A new movie platform has rich movie metadata but very few user interactions. Which approach should it start with?

---

### 7.3 User-Item Matrix and Implicit Feedback
A simple and powerful representation is the **user-item matrix**:

- Rows: users
- Columns: items
- Values: interactions (view, click, purchase, listen, watch time)

In many real systems, you do not have explicit ratings (1-5 stars). You mostly have **implicit feedback** (click/no click, bought/not bought).

| user_id | item_id | interaction |
|---|---|---|
| U1 | I4 | 1 |
| U1 | I9 | 1 |
| U2 | I4 | 1 |
| U3 | I1 | 1 |

**Layered view:**

- **Layer 1:** Keep track of what each user touched.
- **Layer 2:** Build similarity between users or items from this matrix.
- **Layer 3:** For implicit feedback, confidence weighting and negative sampling are common in production-scale methods.

**Why it matters:** Without a clean interaction table, recommendation quality collapses before modeling even starts.

**Quick exercise:** Why is "no click" not always the same as "dislike" in recommendation data?

---

### 7.4 Cold Start and Practical Fixes
The **cold-start problem** appears when:

1. New users have no history.
2. New items have no interactions.

Practical strategies:

| Problem | Practical Fix |
|---|---|
| New user | Ask onboarding preferences, show popular-in-segment items |
| New item | Use content features (tags, text embeddings) to place item near similar known items |
| Sparse data | Hybrid model combining collaborative + content signals |

**Layered view:**

- **Layer 1:** If there is no history, use metadata and popularity.
- **Layer 2:** Hybrid systems reduce failure modes of either method alone.
- **Layer 3:** Contextual bandit strategies can improve exploration vs exploitation in live systems.

**Why it matters:** Cold-start failures are visible to every new user, which can hurt first-week retention.

**Quick exercise:** You launch 500 brand-new products today. Which signal can you rely on immediately: collaborative interactions or item metadata?

---

### 7.5 Hands-On: Item-Based Collaborative Filtering

This lab builds a simple item-item recommender using cosine similarity over a user-item matrix.

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1) Example implicit interaction data
interactions = pd.DataFrame({
  "user": ["U1","U1","U1","U2","U2","U3","U3","U3","U4","U4","U5","U5"],
  "item": ["A","B","D","A","C","B","C","E","A","E","C","D"],
  "interaction": [1,1,1,1,1,1,1,1,1,1,1,1]
})

# 2) Build user-item matrix
user_item = interactions.pivot_table(
  index="user", columns="item", values="interaction", fill_value=0
)

print("User-item matrix:")
print(user_item)

# 3) Compute item-item cosine similarity
item_vectors = user_item.T  # rows become items
sim = cosine_similarity(item_vectors)
sim_df = pd.DataFrame(sim, index=item_vectors.index, columns=item_vectors.index)

print("\nItem similarity matrix:")
print(sim_df.round(2))

# 4) Recommend items similar to a target item
def similar_items(target_item, top_n=3):
  scores = sim_df[target_item].drop(target_item).sort_values(ascending=False)
  return scores.head(top_n)

print("\nTop items similar to 'A':")
print(similar_items("A", top_n=3))

# 5) Recommend for a specific user based on items they already interacted with
def recommend_for_user(user_id, top_n=3):
  seen_items = user_item.columns[user_item.loc[user_id] > 0].tolist()
  candidate_scores = {}

  for seen in seen_items:
    for candidate, score in sim_df[seen].items():
      if candidate in seen_items or candidate == seen:
        continue
      candidate_scores[candidate] = candidate_scores.get(candidate, 0) + score

  ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
  return ranked[:top_n]

user_to_test = "U2"
recs = recommend_for_user(user_to_test, top_n=3)
print(f"\nRecommendations for {user_to_test}:")
for item, score in recs:
  print(f"  item={item}, score={score:.3f}")
```

**What the code does, step by step:**

1. Builds a small implicit-feedback dataset.
2. Converts interactions to a user-item matrix.
3. Computes cosine similarity between items.
4. Retrieves nearest-neighbor items for any target item.
5. Creates user recommendations by aggregating similarities from items the user already engaged with.

> **Visual placeholder - Figure 7.1:** Heatmap of the item-item similarity matrix. Darker cells indicate stronger similarity. Caption: "Item-based collaborative filtering recommends items with similar interaction footprints."

---

### 7.6 Evaluating Recommenders
Recommendation quality is usually judged by top-k ranking metrics:

| Metric | What it asks |
|---|---|
| Precision@k | Of top-k recommended items, how many were relevant? |
| Recall@k | Of all relevant items, how many appeared in top-k? |
| MAP / NDCG | Did relevant items appear near the top, and in what order? |

**Layered view:**

- **Layer 1:** Not just "was it right?" but "was it near the top?"
- **Layer 2:** Offline ranking metrics are useful, but online A/B testing is still required for product impact.
- **Layer 3:** Counterfactual bias and position bias can distort evaluation if logs are not handled carefully.

**Why it matters:** A recommender with slightly better ranking can produce substantial revenue and retention gains at scale.

**Quick exercise:** If users only view the first 5 recommended items, which metric should be emphasized most: Precision@5 or Recall@100?

---

### 7.7 Common Beginner Mistakes

| Mistake | Why it is dangerous | Fix |
|---|---|---|
| Treating recommendation as plain classification | Ignores ranking objective | Optimize top-k ranking quality |
| Ignoring cold start | New users/items get poor experiences | Add hybrid and popularity fallbacks |
| Recommending already-seen items | Wastes recommendation slots | Filter out seen items before ranking |
| Using only popularity | Creates filter bubbles and low personalization | Blend popularity with personalized signals |
| Evaluating only offline | May not reflect user behavior in production | Run controlled online experiments |

---

### Chapter 7 Checkpoint

Answer these questions before moving on:

1. What is the main difference between content-based and collaborative filtering?
2. Why is cold start especially challenging for collaborative methods?
3. In an item-based recommender, what does cosine similarity measure?
4. Why are Precision@k and Recall@k more suitable than simple accuracy for recommendations?

If you can answer these confidently, move to the next chapter.

---

## Next Writing Step
- Chapter 8: Anomaly Detection

---

## Chapter 8: Anomaly Detection

### Finding Rare, Important, Unusual Events

#### 1) What You Will Learn

By the end of this chapter you will be able to:

1. Explain what anomaly detection is and when to use it.
2. Distinguish point anomalies, contextual anomalies, and collective anomalies.
3. Apply practical anomaly methods: z-score, IQR rule, and Isolation Forest.
4. Evaluate anomaly systems under severe class imbalance.

---

### 8.1 What Is Anomaly Detection?
**Anomaly detection** is the task of identifying observations that deviate strongly from normal behavior.

Examples:

- Fraudulent transactions in payment data
- Faulty sensor readings from industrial equipment
- Unusual login behavior in cybersecurity logs
- Sudden spikes or crashes in web traffic

Unlike many Volume 1 tasks, anomalies are usually rare, and labels are often incomplete.

**Layered view:**

- **Layer 1 (plain English):** Find the weird stuff quickly.
- **Layer 2 (practitioner):** Learn normal patterns, then score each new record by how "different" it is.
- **Layer 3 (math hint):** Many methods produce an anomaly score $s(x)$; flag as anomaly when $s(x)$ crosses a threshold $\tau$.

**Why it matters:** Rare events can have disproportionate business impact. Missing one critical anomaly can be far more costly than many false alarms.

**Quick exercise:** Name one domain where false negatives (missed anomalies) are more dangerous than false positives.

---

### 8.2 Types of Anomalies
Not all anomalies look the same.

| Type | Definition | Example |
|---|---|---|
| **Point anomaly** | One value is individually unusual | A single $10,000 charge in a normally $20-$80 spending pattern |
| **Contextual anomaly** | Value is unusual given context | High electricity usage at 3 AM but normal at 3 PM |
| **Collective anomaly** | A sequence/pattern is unusual as a group | Many small transactions split over minutes to evade fraud checks |

**Layered view:**

- **Layer 1:** Some odd values are obvious; others are only odd in context or sequence.
- **Layer 2:** Method choice depends on anomaly type. Simple thresholding helps point anomalies, while time-window or sequence methods are needed for collective anomalies.
- **Layer 3:** Contextual detection can be framed as conditional density estimation $p(x \mid c)$ where $c$ is context.

**Why it matters:** Teams often deploy a point-anomaly method to a contextual problem and wonder why detection quality is poor.

**Quick exercise:** Is a high website traffic spike always an anomaly? What context would you check first?

---

### 8.3 Practical Methods: z-score, IQR, Isolation Forest
#### z-score thresholding

For approximately bell-shaped distributions, compute:

$$z = \frac{x-\mu}{\sigma}$$

Values with $|z| > 3$ are often treated as outliers.

#### IQR rule

Use quartiles to avoid sensitivity to extreme values:

- $IQR = Q3 - Q1$
- Lower bound: $Q1 - 1.5\times IQR$
- Upper bound: $Q3 + 1.5\times IQR$

Values outside bounds are flagged.

#### Isolation Forest

An unsupervised tree-based method that isolates rare points quickly.

- Normal points require more random splits to isolate.
- Anomalies need fewer splits, so they get higher anomaly scores.

**Layered view:**

- **Layer 1:** z-score and IQR are simple rules; Isolation Forest is a flexible ML method.
- **Layer 2:** Start with simple baselines, then move to Isolation Forest when relationships are multivariate and nonlinear.
- **Layer 3:** Isolation Forest's expected path length is shorter for anomalies in randomly partitioned trees.

**Why it matters:** Strong simple baselines catch many issues fast and give a reference point before deploying more complex systems.

**Quick exercise:** When data has heavy tails and extreme skew, which is usually safer: z-score or IQR?

---

### 8.4 Evaluation Under Imbalance
Anomaly datasets are highly imbalanced. Accuracy becomes misleading.

Example:

- 10,000 events
- 50 true anomalies (0.5%)
- Predict "normal" for everything -> 99.5% accuracy, but completely useless

Prefer metrics such as:

| Metric | Why it helps |
|---|---|
| Precision | Of flagged events, how many are truly anomalous? |
| Recall | Of true anomalies, how many did we catch? |
| F1-score | Balances precision and recall |
| PR-AUC | Better summary than ROC-AUC for rare positives |

Threshold selection is a business decision:

- Low threshold -> higher recall, more alerts
- High threshold -> fewer alerts, lower noise

**Layered view:**

- **Layer 1:** Accuracy lies when anomalies are rare.
- **Layer 2:** Pick metrics that reflect alert quality and missed-risk costs.
- **Layer 3:** Tune thresholds on validation sets and align to operational alert capacity.

**Why it matters:** A model can be statistically strong but operationally unusable if it floods analysts with false alarms.

**Quick exercise:** Your fraud team can manually review only 100 alerts/day. Which part of the precision-recall tradeoff becomes most important?

---

### 8.5 Hands-On: Detecting Anomalies in Transaction Data

This lab simulates transaction amounts, injects rare anomalies, and compares IQR with Isolation Forest.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_curve, auc

# 1) Create synthetic transaction data
np.random.seed(123)
n_normal = 3000
n_anom = 45

# Normal transactions around $50 with moderate spread
normal_amounts = np.random.normal(loc=50, scale=10, size=n_normal)
normal_amounts = np.clip(normal_amounts, 1, None)

# Anomalies: unusually large and unusually tiny transactions
anom_high = np.random.normal(loc=260, scale=25, size=n_anom // 2)
anom_low = np.random.normal(loc=2, scale=0.8, size=n_anom - len(anom_high))
anomaly_amounts = np.concatenate([anom_high, anom_low])

amounts = np.concatenate([normal_amounts, anomaly_amounts])
labels = np.concatenate([
  np.zeros(len(normal_amounts), dtype=int),  # 0 = normal
  np.ones(len(anomaly_amounts), dtype=int)   # 1 = anomaly
])

df = pd.DataFrame({"amount": amounts, "is_anomaly": labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Anomaly rate:", df["is_anomaly"].mean().round(4))

# 2) IQR baseline
q1 = df["amount"].quantile(0.25)
q3 = df["amount"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df["pred_iqr"] = ((df["amount"] < lower) | (df["amount"] > upper)).astype(int)

print("\nIQR baseline report:")
print(classification_report(df["is_anomaly"], df["pred_iqr"], digits=3))

# 3) Isolation Forest
iso = IsolationForest(
  n_estimators=250,
  contamination=len(anomaly_amounts) / len(df),
  random_state=42
)
iso.fit(df[["amount"]])

# In sklearn: prediction -1 means anomaly, 1 means normal
pred_iso_raw = iso.predict(df[["amount"]])
df["pred_iso"] = (pred_iso_raw == -1).astype(int)

print("Isolation Forest report:")
print(classification_report(df["is_anomaly"], df["pred_iso"], digits=3))

# 4) PR-AUC using anomaly score
# Higher anomaly likelihood should map to larger score
scores = -iso.score_samples(df[["amount"]])
precision, recall, _ = precision_recall_curve(df["is_anomaly"], scores)
pr_auc = auc(recall, precision)
print(f"Isolation Forest PR-AUC: {pr_auc:.3f}")
```

**What the code does, step by step:**

1. Simulates mostly normal transactions and injects rare anomalous amounts.
2. Applies a transparent IQR rule as a baseline detector.
3. Trains Isolation Forest without needing class labels during fitting.
4. Compares both methods with precision/recall/F1.
5. Computes PR-AUC from model scores for threshold-aware evaluation.

> **Visual placeholder - Figure 8.1:** Histogram of transaction amounts with anomaly points highlighted in red and IQR bounds marked as vertical lines. Caption: "IQR and Isolation Forest both aim to separate rare behavior from normal activity."

---

### 8.6 Common Beginner Mistakes

| Mistake | Why it is dangerous | Fix |
|---|---|---|
| Using accuracy as the main metric | Hides failure on rare anomalies | Track precision, recall, F1, PR-AUC |
| Treating every outlier as fraud/failure | Outliers can be valid edge cases | Add human review and root-cause checks |
| No threshold tuning | Too many or too few alerts | Set thresholds based on operational capacity |
| Ignoring context/time windows | Misses contextual or collective anomalies | Engineer contextual and sequence features |
| Skipping baseline rules | No reference point for model gains | Start with z-score/IQR before advanced methods |

---

### Chapter 8 Checkpoint

Answer these questions before moving on:

1. Why can a model with 99%+ accuracy still be poor for anomaly detection?
2. What is one difference between point anomalies and contextual anomalies?
3. When is IQR usually more robust than z-score?
4. Why is PR-AUC often preferred over ROC-AUC for rare-event detection?

If you can answer these confidently, move to the next chapter.

---

## Next Writing Step
- Chapter 9: Responsible ML (Volume 2)

---

## Chapter 9: Responsible ML

### Operating NLP, Forecasting, and Recommenders Safely

#### 1) What You Will Learn

By the end of this chapter you will be able to:

1. Identify fairness and bias risks specific to NLP systems.
2. Recognize data drift and concept drift in time-series and forecasting workloads.
3. Explain recommendation feedback loops and why they can amplify bias.
4. Design a practical monitoring checklist for post-deployment ML systems.

---

### 9.1 Responsible ML Beyond Tabular Data
In Volume 1, you learned core ethics ideas: fairness, transparency, and accountability. In Volume 2 domains, the risks get subtler:

- Text models can absorb harmful stereotypes from training data.
- Forecasting models can degrade quietly as behavior shifts over time.
- Recommenders can shape user behavior, not just predict it.

**Layered view:**

- **Layer 1 (plain English):** Advanced models can fail in advanced ways.
- **Layer 2 (practitioner):** Risk management must be domain-aware: NLP risk checks differ from forecasting risk checks.
- **Layer 3 (math hint):** Risk is often measured through group-level metric gaps, drift statistics, and calibration error.

**Why it matters:** A model can be technically accurate and still harmful, unstable, or unfair in production.

**Quick exercise:** Think of one model from Chapters 2-8. What is one harm it could cause if deployed without monitoring?

---

### 9.2 NLP-Specific Bias Risks
Text data reflects real-world language patterns, including social bias and historical inequity. NLP models can inherit and amplify these patterns.

Common NLP risk areas:

| Risk | Example |
|---|---|
| Toxicity false positives | Dialect or reclaimed terms incorrectly flagged as abusive |
| Sentiment skew | Neutral statements about one group scored as negative more often |
| Representation bias | Underrepresented groups have fewer examples, causing weaker performance |
| Label bias | Human annotators apply standards inconsistently across groups |

**Layered view:**

- **Layer 1:** The model learns from language, and language contains bias.
- **Layer 2:** Audit performance across slices (region, dialect, channel, demographic proxies where legally appropriate).
- **Layer 3:** Evaluate disaggregated metrics: precision/recall/F1 per slice and gap analysis between groups.

**Why it matters:** If one user group receives systematically lower-quality predictions, trust and product value both decline.

**Quick exercise:** Your sentiment model has overall F1 = 0.89. What extra check would you run before deployment to ensure this is fair?

---

### 9.3 Drift in Time-Series Forecasting
Forecasting systems face a constant reality: the future changes.

Two key drift types:

| Drift Type | What changes | Example |
|---|---|---|
| **Data drift** | Input distribution changes | Average order size increases after a pricing change |
| **Concept drift** | Relationship between inputs and target changes | Promotions no longer drive demand like they did last quarter |

**Layered view:**

- **Layer 1:** Yesterday's patterns can stop working.
- **Layer 2:** Track feature distributions and forecast error over time windows; retrain when thresholds are crossed.
- **Layer 3:** Use tests such as PSI (Population Stability Index) or KS-based checks for feature shift, plus rolling MAE/RMSE for model degradation.

**Why it matters:** Forecast models often fail gradually, so teams miss the warning signs unless they monitor trends, not just one-off scores.

**Quick exercise:** Your rolling 30-day MAE doubles over two months while model code is unchanged. What does this suggest?

---

### 9.4 Recommendation Feedback Loops
Recommenders do more than observe behavior; they influence it.

If the model repeatedly recommends one narrow content type, users click what they see, and the system interprets that as "user preference," reinforcing itself.

This can cause:

- Popularity bias (already-popular items become even more dominant)
- Reduced content diversity
- Filter bubbles and weaker discovery

**Layered view:**

- **Layer 1:** Recommendation changes behavior, then behavior retrains recommendation.
- **Layer 2:** Add diversity, novelty, and exploration constraints to avoid narrow loops.
- **Layer 3:** Multi-objective optimization can balance relevance with fairness/diversity objectives.

**Why it matters:** Long-term ecosystem health (creator diversity, catalog coverage, user satisfaction) can deteriorate even when short-term CTR rises.

**Quick exercise:** Which offline metric might improve while user satisfaction still declines due to repetitive recommendations?

---

### 9.5 Hands-On: Build a Responsible ML Monitoring Snapshot

This lab creates a compact monitoring report combining fairness slices, forecast drift, and recommendation diversity.

```python
import pandas as pd
import numpy as np

np.random.seed(42)

# 1) NLP fairness slice example (simulated)
nlp_eval = pd.DataFrame({
  "group": ["Group_A", "Group_B", "Group_C"],
  "precision": [0.91, 0.84, 0.89],
  "recall": [0.88, 0.76, 0.87],
  "f1": [0.895, 0.798, 0.880]
})
nlp_eval["f1_gap_vs_best"] = nlp_eval["f1"].max() - nlp_eval["f1"]

# 2) Forecast drift example: rolling MAE by month
months = pd.period_range("2025-01", periods=8, freq="M").astype(str)
rolling_mae = [7.1, 7.3, 7.4, 7.8, 8.4, 9.2, 10.3, 11.0]
forecast_monitor = pd.DataFrame({"month": months, "rolling_mae": rolling_mae})
forecast_monitor["pct_change_from_start"] = (
  (forecast_monitor["rolling_mae"] - forecast_monitor["rolling_mae"].iloc[0])
  / forecast_monitor["rolling_mae"].iloc[0]
  * 100
)

# 3) Recommendation diversity example
recommended_items = [
  "item_01", "item_02", "item_01", "item_03", "item_01",
  "item_02", "item_04", "item_01", "item_05", "item_01"
]
rec_series = pd.Series(recommended_items)
catalog_coverage = rec_series.nunique() / 50  # assume 50 items available
top1_share = rec_series.value_counts(normalize=True).iloc[0]

# 4) Consolidated snapshot
print("=== NLP Fairness Slice Report ===")
print(nlp_eval)

print("\n=== Forecast Drift Report ===")
print(forecast_monitor)

print("\n=== Recommendation Diversity Report ===")
print(f"Catalog coverage: {catalog_coverage:.2%}")
print(f"Top-1 recommendation share: {top1_share:.2%}")

# 5) Basic alert rules
alerts = []

if nlp_eval["f1_gap_vs_best"].max() > 0.08:
  alerts.append("NLP fairness gap alert: one group lags by > 0.08 F1")

if forecast_monitor["pct_change_from_start"].iloc[-1] > 30:
  alerts.append("Forecast drift alert: rolling MAE increased by > 30%")

if top1_share > 0.40:
  alerts.append("Recommender concentration alert: top item share > 40%")

print("\n=== Alerts ===")
if alerts:
  for a in alerts:
    print("-", a)
else:
  print("No alerts triggered.")
```

**What the code does, step by step:**

1. Simulates NLP performance across user groups to check metric gaps.
2. Tracks a rolling forecast error trend to detect degradation.
3. Measures recommendation concentration and catalog coverage.
4. Applies clear operational alert thresholds.
5. Produces one compact, human-readable risk summary.

> **Visual placeholder - Figure 9.1:** A three-panel dashboard mockup: (left) group-wise F1 bars, (middle) rolling MAE trend line, (right) recommendation item frequency bar chart. Caption: "Responsible ML monitoring combines fairness, stability, and ecosystem health signals."

---

### 9.6 Operational Checklist for Responsible ML

Before deployment:

1. Define unacceptable failure modes (harm, bias, severe misses).
2. Set group-slice evaluation criteria and acceptable metric-gap limits.
3. Define drift metrics and retraining triggers.
4. Add recommendation diversity and concentration constraints (if recommender).
5. Write rollback and incident-response procedures.

After deployment:

1. Monitor fairness slices regularly, not once.
2. Track model and data drift trends over time.
3. Review high-impact errors with human oversight.
4. Recalibrate thresholds when business conditions change.
5. Keep model cards/change logs up to date.

**Why it matters:** Responsible ML is not one checklist at launch; it is an ongoing operations practice.

---

### 9.7 Common Beginner Mistakes

| Mistake | Why it is dangerous | Fix |
|---|---|---|
| Trusting only aggregate metrics | Hides group-level failures | Track disaggregated metrics by slice |
| Evaluating fairness once, then stopping | Performance can drift after deployment | Add continuous fairness monitoring |
| Ignoring recommendation concentration | Over-focuses on few items and harms discovery | Track catalog coverage and top-item share |
| No drift thresholds | Teams notice failures too late | Define trigger-based retraining/rollback rules |
| No human review process | Critical edge cases go unchecked | Add escalation path for high-risk predictions |

---

### Chapter 9 Checkpoint

Answer these questions before moving on:

1. Why can strong overall F1 still hide fairness problems in NLP?
2. What is the difference between data drift and concept drift?
3. How can recommendation feedback loops reduce long-term system quality?
4. Name two metrics you would include in a production Responsible ML dashboard.

If you can answer these confidently, move to the next chapter.

---

## Next Writing Step
- Chapter 10: Capstone Project (Volume 2)

---

## Chapter 10: Capstone Project

### Bringing NLP, Forecasting, Recommendation, and Anomaly Detection Together

#### 1) What You Will Learn

By the end of this chapter you will be able to:

1. Scope and execute an end-to-end intermediate ML project with realistic constraints.
2. Select methods from Chapters 1-9 for a business-style problem.
3. Evaluate model performance with task-appropriate metrics.
4. Present a complete, decision-ready ML project report.

---

### 10.1 Capstone Brief
You are now the ML analyst for a fictional e-commerce company called **Northstar Market**.

The company asks for a practical analytics bundle that supports customer experience and operational planning.

You must deliver **one integrated capstone** with four components:

1. **NLP module:** Classify incoming support tickets by topic.
2. **Sentiment module:** Score customer reviews as positive/neutral/negative.
3. **Forecast module:** Predict next-week daily order volume.
4. **Risk module:** Flag anomalous transactions for manual review.

Optional stretch goal:

5. **Recommendation module:** Recommend related products based on interaction patterns.

**Layered view:**

- **Layer 1 (plain English):** Solve multiple practical ML tasks that work together.
- **Layer 2 (practitioner):** Build separate pipelines per task, then combine outputs into one decision dashboard.
- **Layer 3 (system design hint):** A modular architecture lets each model retrain independently while sharing common data quality checks.

**Why it matters:** Real organizations rarely ask for one isolated model. They need several models that connect to business decisions.

**Quick exercise:** Which of the four required modules would you implement first, and why?

---

### 10.2 Capstone Data Pack
Use or simulate datasets with this structure:

| Table | Example Columns | Main Task |
|---|---|---|
| `support_tickets.csv` | `ticket_id`, `created_at`, `text`, `topic_label` | Text classification |
| `reviews.csv` | `review_id`, `product_id`, `review_text`, `sentiment_label` | Sentiment analysis |
| `daily_orders.csv` | `date`, `orders`, `promo_flag`, `holiday_flag` | Forecasting |
| `transactions.csv` | `txn_id`, `timestamp`, `amount`, `country`, `device` | Anomaly detection |
| `interactions.csv` (optional) | `user_id`, `item_id`, `event_type` | Recommendation |

Minimum quality checks before modeling:

1. Missing values handled with explicit rules.
2. Duplicates removed.
3. Time columns parsed and sorted.
4. Label distributions inspected for imbalance.

**Why it matters:** Most capstone failures come from data readiness issues, not algorithm choice.

---

### 10.3 Suggested Build Order
Follow this order to reduce project risk:

1. **Text classification baseline** (TF-IDF + Logistic Regression)
2. **Sentiment model** (same base pipeline, then error analysis)
3. **Forecast baseline** (naive + lag-feature regression)
4. **Anomaly baseline** (IQR) then **Isolation Forest**
5. Optional recommender (item-item cosine)

At each stage:

- Save metrics
- Save confusion/error examples
- Save one figure for your final report

**Layered view:**

- **Layer 1:** Build simple first, then improve.
- **Layer 2:** A staged plan gives early wins and prevents dead ends.
- **Layer 3:** Baseline-first methodology supports defensible model comparisons.

**Quick exercise:** Why is it useful to reuse TF-IDF + Logistic Regression as an early baseline for both topic classification and sentiment?

---

### 10.4 Evaluation Framework
Use metrics matched to each module:

| Module | Primary Metrics | Secondary Checks |
|---|---|---|
| Topic classification | Macro-F1, per-class precision/recall | Confusion matrix |
| Sentiment | Macro-F1, class balance | Error slices by product/category |
| Forecasting | MAE, RMSE | Rolling error trend by week |
| Anomaly detection | Precision, Recall, PR-AUC | Alert volume at threshold |
| Recommendation (optional) | Precision@k, Recall@k | Catalog coverage/diversity |

Set explicit acceptance criteria before training.

Example:

- Topic classifier macro-F1 >= 0.82
- Forecast MAE <= 12 orders/day
- Anomaly precision >= 0.35 at recall >= 0.70

**Why it matters:** Predefined criteria keep evaluation objective and reduce confirmation bias.

---

### 10.5 Hands-On: Capstone Skeleton Pipeline

The code below is a compact scaffold you can expand into your final project.

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# 1) NLP topic classification (toy example)
# ---------------------------------------------------------------------------
tickets = pd.DataFrame({
  "text": [
    "My order is late and tracking is stuck",
    "How do I reset my account password?",
    "The product arrived damaged",
    "Please cancel my subscription",
    "I was charged twice for one item",
    "Where can I update shipping address?"
  ],
  "topic": ["delivery", "account", "returns", "billing", "billing", "account"]
})

topic_clf = Pipeline([
  ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2))),
  ("logreg", LogisticRegression(max_iter=200, random_state=42))
])
topic_clf.fit(tickets["text"], tickets["topic"])
topic_pred = topic_clf.predict(tickets["text"])
topic_f1 = f1_score(tickets["topic"], topic_pred, average="macro")

# ---------------------------------------------------------------------------
# 2) Forecasting orders with lag features (toy example)
# ---------------------------------------------------------------------------
np.random.seed(0)
n = 180
orders = 120 + 0.1*np.arange(n) + 10*np.sin(2*np.pi*np.arange(n)/7) + np.random.normal(0, 3, n)
df_orders = pd.DataFrame({"orders": orders})
df_orders["lag_1"] = df_orders["orders"].shift(1)
df_orders["lag_7"] = df_orders["orders"].shift(7)
df_orders = df_orders.dropna()

split = int(len(df_orders)*0.8)
train_o = df_orders.iloc[:split]
test_o = df_orders.iloc[split:]

reg = LinearRegression()
reg.fit(train_o[["lag_1", "lag_7"]], train_o["orders"])
pred_o = reg.predict(test_o[["lag_1", "lag_7"]])
orders_mae = mean_absolute_error(test_o["orders"], pred_o)

# ---------------------------------------------------------------------------
# 3) Anomaly detection on transaction amounts (toy example)
# ---------------------------------------------------------------------------
txn = pd.DataFrame({
  "amount": np.concatenate([
    np.random.normal(50, 8, 1200),
    np.random.normal(220, 20, 20)
  ])
})

iso = IsolationForest(contamination=20/1220, random_state=42)
iso.fit(txn[["amount"]])
txn["anomaly_flag"] = (iso.predict(txn[["amount"]]) == -1).astype(int)
alert_rate = txn["anomaly_flag"].mean()

# ---------------------------------------------------------------------------
# 4) Consolidated summary
# ---------------------------------------------------------------------------
summary = pd.DataFrame({
  "module": ["topic_classification", "forecasting", "anomaly_detection"],
  "metric": ["macro_f1", "mae", "alert_rate"],
  "value": [topic_f1, orders_mae, alert_rate]
})

print("Capstone mini-dashboard")
print(summary)
```

**What the code does, step by step:**

1. Trains a topic classifier for ticket routing.
2. Builds a lag-based forecast model for order planning.
3. Trains an anomaly detector for transaction monitoring.
4. Produces one mini-dashboard table with core metrics.

> **Visual placeholder - Figure 10.1:** A single-page capstone dashboard mockup with three cards: (A) NLP macro-F1, (B) forecast MAE trend, (C) anomaly alert rate. Caption: "Integrated ML reporting turns model outputs into operational decisions."

---

### 10.6 Capstone Deliverables Checklist

Your final capstone package should include:

1. **Problem statement** (business context and objectives)
2. **Data description** (sources, schema, quality checks)
3. **Modeling approach per module** (baseline and final)
4. **Evaluation results** (tables + visuals)
5. **Error analysis** (what fails and why)
6. **Responsible ML section** (fairness, drift, monitoring)
7. **Deployment proposal** (how this would run weekly/daily)
8. **Executive summary** (non-technical, 1 page)

Suggested presentation flow:

1. Business problem
2. Method overview
3. Results by module
4. Risk and monitoring plan
5. Next-step roadmap

**Why it matters:** Strong communication is part of ML practice. A technically good model that stakeholders cannot trust or understand will not be adopted.

---

### 10.7 Common Beginner Mistakes

| Mistake | Why it is dangerous | Fix |
|---|---|---|
| Building all modules at once | Hard to debug and compare | Implement in staged milestones |
| No baseline metrics recorded | Cannot prove improvement | Save baseline and final metrics side by side |
| Inconsistent data splits | Invalid comparisons across modules | Define split policy early and reuse it |
| Ignoring operations constraints | Great model, unusable workflow | Include latency, alert capacity, and retraining plans |
| Weak final narrative | Stakeholders do not act on outputs | Add concise executive summary and action recommendations |

---

### Chapter 10 Checkpoint

Answer these questions before moving on:

1. Why should a capstone begin with baseline models for each module?
2. Which metric pair is most suitable for anomaly detection and why?
3. What is one sign that your forecasting evaluation setup is unrealistic?
4. Name two items required in a strong executive-facing capstone report.

If you can answer these confidently, you have completed the core Volume 2 chapters.

---

## Next Writing Step
- Appendix A: Formula and Metrics Quick Reference (Vol. 2)

---

## Appendix A: Formula and Metrics Quick Reference

### Text and Classification Metrics

| Name | Formula / Definition | Use |
|---|---|---|
| TF | Term frequency in a document | Basic token weighting |
| IDF | $\log\left(\frac{N}{df_t}\right)$ | Down-weight common terms |
| TF-IDF | TF x IDF | Core text feature baseline |
| Precision | $\frac{TP}{TP+FP}$ | Flag quality |
| Recall | $\frac{TP}{TP+FN}$ | Coverage of true positives |
| F1 | Harmonic mean of precision and recall | Balanced classification metric |
| Macro-F1 | Mean F1 across classes | Good for class imbalance |

### Forecasting Metrics

| Name | Formula / Definition | Use |
|---|---|---|
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | Average absolute miss in original units |
| RMSE | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | Penalizes large misses |
| MAPE | $\frac{100}{n}\sum\left|\frac{y-\hat{y}}{y}\right|$ | Percent error, avoid near-zero denominator |

### Recommendation and Anomaly Metrics

| Name | Formula / Definition | Use |
|---|---|---|
| Precision@k | Relevant items in top-k / k | Top-list quality |
| Recall@k | Relevant items in top-k / all relevant items | Coverage in top-k |
| PR-AUC | Area under precision-recall curve | Rare event evaluation |

---

## Appendix B: Python Library Quick Guide

| Library | Main Role in Volume 2 | Example Use |
|---|---|---|
| pandas | Data loading, cleaning, feature engineering | Lag features, rolling windows, pivot tables |
| numpy | Numeric operations and synthetic data generation | Noise simulation, vectorized math |
| scikit-learn | ML models and evaluation | TF-IDF, LogisticRegression, IsolationForest |
| statsmodels | Classical forecasting | ExponentialSmoothing |
| gensim | Embeddings and similarity | Pretrained GloVe vectors |
| matplotlib / seaborn | Visualization | Error trends, heatmaps, forecast plots |

Minimum install set for this manual:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels gensim
```

---

## Appendix C: Data Split and Leakage Safety Checklist

Use this checklist before every modeling run:

1. Confirm train/test split reflects real deployment timing.
2. Ensure no future information appears in training features.
3. Apply `.shift(1)` before rolling statistics in time-series features.
4. Remove target leakage columns that encode outcomes.
5. Keep validation and test sets untouched during tuning.
6. Version your split logic so all modules use consistent boundaries.

Quick leak test question:

- "Could this feature exist at prediction time in the real world?"

If the answer is no, remove it.

---

## Appendix D: Error Analysis Templates

### NLP / Text Classification

Create a table with columns:

- `sample_id`
- `text`
- `true_label`
- `predicted_label`
- `confidence`
- `error_type` (ambiguous wording, short text, sarcasm, spelling, etc.)

### Forecasting

Track errors over time with:

- `date`
- `actual`
- `predicted`
- `absolute_error`
- `promo_flag`
- `holiday_flag`

Then inspect top 20 worst-error dates for recurring patterns.

### Recommendations

For missed relevance, log:

- `user_id`
- `recommended_items`
- `clicked_items`
- `missed_relevant_items`
- `cold_start_flag`

### Anomaly Detection

For alert review:

- `event_id`
- `anomaly_score`
- `threshold`
- `alerted`
- `human_verdict`
- `root_cause`

---

## Appendix E: Study Tracks After Volume 2

Choose one direction based on your goals.

### Track 1: Applied NLP Engineer

1. Learn transformer basics (attention, tokenization details).
2. Practice fine-tuning BERT-style models.
3. Add retrieval and semantic search fundamentals.
4. Study evaluation for safety and bias in language systems.

### Track 2: Forecasting Specialist

1. Deepen ARIMA/SARIMA and ETS practice.
2. Learn probabilistic forecasting (prediction intervals).
3. Study hierarchical forecasting for multi-store/multi-SKU setups.
4. Build monitoring for drift and recalibration.

### Track 3: Recommender Systems Engineer

1. Learn matrix factorization and embedding-based retrieval.
2. Study candidate generation vs ranking architectures.
3. Practice online experimentation (A/B tests, guardrail metrics).
4. Learn diversity, fairness, and exploration strategies.

### Track 4: Risk and Trust ML

1. Build anomaly systems with richer features and sequences.
2. Learn graph-based fraud detection basics.
3. Study responsible ML operations and governance.
4. Design human-in-the-loop decision workflows.

---

## Appendix F: Capstone Grading Rubric (Self-Assessment)

Score each category from 1 (needs work) to 5 (excellent).

| Category | What to look for |
|---|---|
| Problem framing | Clear business objective and constraints |
| Data quality | Cleaning, schema clarity, leakage prevention |
| Baselines | Strong baseline selection and reporting |
| Model quality | Appropriate methods and metrics per module |
| Error analysis | Concrete examples and failure insights |
| Responsible ML | Fairness, drift, and monitoring plan |
| Communication | Clear visuals and executive summary |
| Reproducibility | Organized code, fixed seeds, documented steps |

Interpretation:

- 34-40: Deployment-ready project quality
- 26-33: Strong project with clear next improvements
- 18-25: Functional but needs deeper evaluation and communication
- Below 18: Rebuild with tighter scope and baseline discipline

---

## Appendix G: Final Reference Sheets

### Quick Model Selection Guide

| Problem | First Baseline | Next Upgrade |
|---|---|---|
| Text topic classification | TF-IDF + Logistic Regression | Linear SVM or transformer fine-tuning |
| Sentiment | TF-IDF + Logistic Regression | Domain-adapted embeddings |
| Forecasting | Naive / seasonal naive | Lag-feature ML and ETS/ARIMA |
| Recommendation | Popularity + item similarity | Matrix factorization or hybrid model |
| Anomaly detection | IQR / z-score | Isolation Forest and threshold tuning |

### Deployment Readiness Checklist

1. Baseline outperformed on a fair validation setup.
2. Metrics mapped to business goals and alert capacity.
3. Responsible ML checks completed (fairness/drift/diversity).
4. Monitoring dashboard and thresholds defined.
5. Retraining and rollback process documented.

### Closing Note

You have completed Volume 2.

You can now:

1. Build practical intermediate ML systems across multiple data types.
2. Evaluate them with the right metrics and validation logic.
3. Communicate results in a way that supports real decisions.
4. Operate models responsibly after deployment.

Keep this manual as a reusable project playbook, not just a one-time tutorial.

---

## References

The sources below are reliable starting points for the concepts used throughout this guide.

### Core Machine Learning and Evaluation

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. https://hastie.su.domains/ElemStatLearn/
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. https://www.statlearning.com/
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
4. scikit-learn User Guide (official documentation). https://scikit-learn.org/stable/user_guide.html

### Natural Language Processing, Text Features, and Sentiment

1. Manning, C. D., Raghavan, P., & Schutze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/
2. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (draft 3rd ed.). https://web.stanford.edu/~jurafsky/slp3/
3. NLTK Book (Bird, Klein, Loper): *Natural Language Processing with Python*. https://www.nltk.org/book/
4. scikit-learn text feature extraction documentation. https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

### Word Embeddings

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781. https://arxiv.org/abs/1301.3781
2. Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. https://aclanthology.org/D14-1162/
3. Gensim documentation (official). https://radimrehurek.com/gensim/

### Time-Series and Forecasting

1. Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice* (open textbook). https://otexts.com/fpp3/
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
3. statsmodels documentation (official). https://www.statsmodels.org/stable/index.html
4. scikit-learn TimeSeriesSplit documentation. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

### Recommendation Systems

1. Ricci, F., Rokach, L., & Shapira, B. (Eds.). (2022). *Recommender Systems Handbook* (3rd ed.). Springer.
2. Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.
3. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8), 30-37. https://ieeexplore.ieee.org/document/5197422

### Anomaly Detection

1. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3), 1-58. https://dl.acm.org/doi/10.1145/1541880.1541882
2. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *ICDM 2008*. https://ieeexplore.ieee.org/document/4781136
3. scikit-learn IsolationForest documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

### Responsible AI and ML Governance

1. NIST AI Risk Management Framework (AI RMF 1.0). https://www.nist.gov/itl/ai-risk-management-framework
2. OECD AI Principles. https://oecd.ai/en/ai-principles
3. EU Ethics Guidelines for Trustworthy AI (High-Level Expert Group on AI). https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai
4. Google ML Crash Course: Fairness module. https://developers.google.com/machine-learning/crash-course/fairness/video-lecture
5. IBM AI Fairness 360 Toolkit documentation. https://aif360.readthedocs.io/

### Practical Data Science and Documentation References

1. pandas documentation (official). https://pandas.pydata.org/docs/
2. NumPy documentation (official). https://numpy.org/doc/
3. Matplotlib documentation (official). https://matplotlib.org/stable/users/index.html
4. seaborn documentation (official). https://seaborn.pydata.org/

Note: For implementation details in this guide, prioritize official library documentation first, then textbook chapters, then research papers for deeper theory.


