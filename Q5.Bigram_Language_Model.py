"""
Bigram Language Model Implementation (MLE)

This program builds a simple bigram language model from a small training corpus.
It counts unigrams and bigrams, computes bigram probabilities using MLE, then
computes the probability of two test sentences and prints which one is higher.
"""



from collections import defaultdict


# -------------------------------
# Step 1: Define Training Corpus
# -------------------------------

corpus = [
    ["<s>", "I", "love", "NLP", "</s>"],
    ["<s>", "I", "love", "deep", "learning", "</s>"],
    ["<s>", "deep", "learning", "is", "fun", "</s>"]
]


# ---------------------------------
# Step 2: Compute Unigram Counts
# ---------------------------------

unigram_counts = defaultdict(int)

for sentence in corpus:
    for word in sentence:
        unigram_counts[word] += 1


# ---------------------------------
# Step 3: Compute Bigram Counts
# ---------------------------------

bigram_counts = defaultdict(int)

for sentence in corpus:
    for i in range(1, len(sentence)):
        prev_word = sentence[i - 1]
        current_word = sentence[i]
        bigram_counts[(prev_word, current_word)] += 1


# ---------------------------------
# Step 4: Bigram Probability (MLE)
# ---------------------------------

def bigram_probability(w1, w2):
    """
    Computes P(w2 | w1) using Maximum Likelihood Estimation.
    
    Formula:
        P(w2 | w1) = Count(w1, w2) / Count(w1)
    """
    if unigram_counts[w1] == 0:
        return 0
    return bigram_counts[(w1, w2)] / unigram_counts[w1]


# ---------------------------------
# Step 5: Sentence Probability
# ---------------------------------

def sentence_probability(sentence):
    """
    Computes the probability of a sentence
    by multiplying all bigram probabilities.
    """
    probability = 1.0

    for i in range(1, len(sentence)):
        w1 = sentence[i - 1]
        w2 = sentence[i]
        prob = bigram_probability(w1, w2)
        probability *= prob

    return probability


# ---------------------------------
# Step 6: Test Sentences
# ---------------------------------

sentence1 = ["<s>", "I", "love", "NLP", "</s>"]
sentence2 = ["<s>", "I", "love", "deep", "learning", "</s>"]

prob1 = sentence_probability(sentence1)
prob2 = sentence_probability(sentence2)


# ---------------------------------
# Step 7: Print Results
# ---------------------------------

print("Sentence 1:", " ".join(sentence1))
print("Probability:", prob1)
print()

print("Sentence 2:", " ".join(sentence2))
print("Probability:", prob2)
print()

if prob1 > prob2:
    print("Sentence 1 is more probable.")
elif prob2 > prob1:
    print("Sentence 2 is more probable.")
else:
    print("Both sentences are equally probable.")
