from typing import List

import torch
from torch.utils.data import Dataset

import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag


nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


CONTRACTION_MAP = { "ain't": "is not",
                    "aren't": "are not",
                    "can't": "cannot",
                    "can't've": "cannot have",
                    "'cause": "because",
                    "could've": "could have",
                    "couldn't": "could not",
                    "couldn't've": "could not have",
                    "didn't": "did not",
                    "doesn't": "does not",
                    "don't": "do not",
                    "hadn't": "had not",
                    "hadn't've": "had not have",
                    "hasn't": "has not",
                    "haven't": "have not",
                    "he'd": "he would",
                    "he'd've": "he would have",
                    "he'll": "he will",
                    "he'll've": "he he will have",
                    "he's": "he is",
                    "how'd": "how did",
                    "how'd'y": "how do you",
                    "how'll": "how will",
                    "how's": "how is",
                    "I'd": "I would",
                    "I ain't": "I am not",
                    "I'd've": "I would have",
                    "I'll": "I will",
                    "I'll've": "I will have",
                    "I'm": "I am",
                    "I've": "I have",
                    "i'd": "i would",
                    "i'd've": "i would have",
                    "i'll": "i will",
                    "i'll've": "i will have",
                    "i'm": "i am",
                    "i've": "i have",
                    "isn't": "is not",
                    "it'd": "it would",
                    "it'd've": "it would have",
                    "it'll": "it will",
                    "it'll've": "it will have",
                    "it's": "it is",
                    "let's": "let us",
                    "ma'am": "madam",
                    "mayn't": "may not",
                    "might've": "might have",
                    "mightn't": "might not",
                    "mightn't've": "might not have",
                    "must've": "must have",
                    "mustn't": "must not",
                    "mustn't've": "must not have",
                    "needn't": "need not",
                    "needn't've": "need not have",
                    "o'clock": "of the clock",
                    "oughtn't": "ought not",
                    "oughtn't've": "ought not have",
                    "shan't": "shall not",
                    "sha'n't": "shall not",
                    "shan't've": "shall not have",
                    "she'd": "she would",
                    "she'd've": "she would have",
                    "she'll": "she will",
                    "she'll've": "she will have",
                    "she's": "she is",
                    "should've": "should have",
                    "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so've": "so have",
                    "so's": "so as",
                    "that'd": "that would",
                    "that'd've": "that would have",
                    "that's": "that is",
                    "there'd": "there would",
                    "there'd've": "there would have",
                    "there's": "there is",
                    "they'd": "they would",
                    "they'd've": "they would have",
                    "they'll": "they will",
                    "they'll've": "they will have",
                    "they're": "they are",
                    "they've": "they have",
                    "to've": "to have",
                    "wasn't": "was not",
                    "we'd": "we would",
                    "we'd've": "we would have",
                    "we'll": "we will",
                    "we'll've": "we will have",
                    "we're": "we are",
                    "we've": "we have",
                    "weren't": "were not",
                    "what'll": "what will",
                    "what'll've": "what will have",
                    "what're": "what are",
                    "what's": "what is",
                    "what've": "what have",
                    "when's": "when is",
                    "when've": "when have",
                    "where'd": "where did",
                    "where's": "where is",
                    "where've": "where have",
                    "who'll": "who will",
                    "who'll've": "who will have",
                    "who's": "who is",
                    "who've": "who have",
                    "why's": "why is",
                    "why've": "why have",
                    "will've": "will have",
                    "won't": "will not",
                    "won't've": "will not have",
                    "would've": "would have",
                    "wouldn't": "would not",
                    "wouldn't've": "would not have",
                    "y'all": "you all",
                    "y'all'd": "you all would",
                    "y'all'd've": "you all would have",
                    "y'all're": "you all are",
                    "y'all've": "you all have",
                    "you'd": "you would",
                    "you'd've": "you would have",
                    "you'll": "you will",
                    "you'll've": "you will have",
                    "you're": "you are",
                    "you've": "you have"
                    }


class PreProcess:
    def __init__(self, lowercase_norm=False, period_norm=False, special_chars_norm=False, accented_norm=False, contractions_norm=False,
                 stemming_norm=False, lemma_norm=False, stopword_norm=False, proper_norm=False):

        self.lowercase_norm = lowercase_norm
        self.period_norm = period_norm
        self.special_chars_norm = special_chars_norm
        self.accented_norm = accented_norm
        self.contractions_norm = contractions_norm
        self.stemming_norm = stemming_norm
        self.lemma_norm = lemma_norm
        self.stopword_norm = stopword_norm
        self.proper_norm = proper_norm

    def lowercase_normalization(self, data):
        return data.lower()

    def period_remove(self, data):
        return data.replace(".", " ")

    def special_char_remove(self, data, remove_digits=False):  # Remove special characters
        tokens = self.tokenization(data)
        special_char_norm_data = []

        for token in tokens:
            sentence = ""
            for word in token:
                sentence += word + " "
            sentence.rstrip()

            clean_remove = re.compile('<.*?>')
            norm_sentence = re.sub(clean_remove, '', sentence)

            norm_sentence = re.sub(r'[^\x00-\x7F]+','', norm_sentence)
            norm_sentence = norm_sentence.replace("\\", "")
            norm_sentence = norm_sentence.replace("-", " ")
            norm_sentence = norm_sentence.replace(",", "")
            special_char_norm_data.append(norm_sentence)

        return special_char_norm_data

    def accented_word_normalization(self, data):  # Normalize accented chars/words
        tokens = self.tokenization(data)
        accented_norm_data = []

        for token in tokens:
            sentence = ""
            for word in token:
                sentence += word + " "
            sentence.rstrip()
            norm_sentence = unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore').decode('utf-8', 'ignore')

            accented_norm_data.append(norm_sentence)

        return accented_norm_data

    def expand_contractions(self, data, pycontrct=False):  # Expand contractions

        # Simple contraction removal based on pre-defined set of contractions
        contraction_mapping = CONTRACTION_MAP
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        tokens = self.tokenization(data)
        contraction_norm_data = []

        for token in tokens:
            sentence = ""
            for word in token:
                sentence += word + " "
            sentence.rstrip()

            expanded_text = contractions_pattern.sub(expand_match, sentence)
            expanded_text = re.sub("'", "", expanded_text)

            contraction_norm_data.append(expanded_text)

        return contraction_norm_data

    def stemming(self, data):
        stemmer = nltk.stem.PorterStemmer()
        tokens = self.tokenization(data)
        stemmed_data = []

        for i in range(len(tokens)):
            s1 = " ".join(stemmer.stem(tokens[i][j]) for j in range(len(tokens[i])))
            stemmed_data.append(s1)

        return stemmed_data

    def lemmatization(self, data):
        lemma = nltk.stem.WordNetLemmatizer()
        tokens = self.tokenization(data)
        lemmatized_data = []

        for i in range(len(tokens)):
            s1 = " ".join(lemma.lemmatize(tokens[i][j]) for j in range(len(tokens[i])))
            lemmatized_data.append(s1)

        return lemmatized_data

    def stopword_remove(self, data):  # Remove special characters
        filtered_sentence = []
        stop_words = set(stopwords.words('english'))
        data = self.tokenization(data)

        for i in range(len(data)):
            res = ""
            for j in range(len(data[i])):
                if data[i][j].lower() not in stop_words:
                    res = res + " " + data[i][j]
            filtered_sentence.append(res)

        return filtered_sentence

    def remove_proper_nouns(self, data):
        common_words = []
        data = self.tokenization(data)
        for i in range(len(data)):
            tagged_sent = pos_tag(data[i])
            proper_nouns = [word for word, pos in tagged_sent if pos == 'NNP']
            res = ""
            for j in range(len(data[i])):
                if data[i][j] not in proper_nouns:
                    res = res + " " + data[i][j]
            common_words.append(res)

        return common_words

    def tokenization(self, data):
        tokens = []
        for i in range(len(data)):
            tokenizer = nltk.tokenize.WhitespaceTokenizer()
            tokens.append(tokenizer.tokenize(data[i]))
        return tokens

    def fit(self, data):

        data = [str(data)]

        if self.special_chars_norm:
            data = self.special_char_remove(data, remove_digits=False)

        if self.accented_norm:
            data = self.accented_word_normalization(data)

        if self.stemming_norm:
            data = self.stemming(data)

        if self.proper_norm:
            data = self.remove_proper_nouns(data)

        if self.stopword_norm:
            data = self.stopword_remove(data)

        if self.lemma_norm:
            data = self.lemmatization(data)

        data = data[0]

        if self.lowercase_norm:
            data = self.lowercase_normalization(str(data))

        if self.period_norm:
            data = self.period_remove(str(data))

        return data


class EncodeEvalData(Dataset):
    def __init__(self, input_texts: List[str], tokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None):

        self.input_texts = input_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, index):
        text = self.input_texts[index]

        # Preprocessing
        preprocessor = PreProcess(special_chars_norm=True, lowercase_norm=True, period_norm=True, proper_norm=True, accented_norm=True)

        text = preprocessor.fit(text)

        padded_sequences = self.tokenizer(text, padding='max_length', max_length=self.max_sequence_length, truncation=True)

        return torch.tensor(padded_sequences['input_ids']), torch.tensor(padded_sequences['attention_mask'])
