import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import ujson

# Load the JSON file containing scraped results
with open('scrapped_results.json', 'r') as doc:
    scraper_results = doc.read()

# Extract author names from the JSON data
authors = []
data_dict = ujson.loads(scraper_results)
for item in data_dict:
    if "author_a" in item:
        authors.append(item["author_a"])
    elif "author_b" in item:
        authors.append(item["author_b"])
    elif "author_c" in item:
        authors.append(item["author_c"])
    elif "author_d" in item:
        authors.append(item["author_d"])
    elif "author_e" in item:
        authors.append(item["author_e"])
    elif "author_f" in item:
        authors.append(item["author_f"])
    elif "author_g" in item:
        authors.append(item["author_g"])
    elif "author_h" in item:
        authors.append(item["author_h"])

# Write the author names to a JSON file
with open('author_names.json', 'w') as f:
    ujson.dump(authors, f)

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the JSON file containing author names
with open('author_names.json', 'r') as f:
    author_data = f.read()

# Load JSON data
authors = ujson.loads(author_data)

# Preprocess the author names
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
authors_list_first_stem = []
authors_list = []

for author in authors:
    words = word_tokenize(author)
    stem_word = ""
    for word in words:
        if word.lower() not in stop_words:
            stem_word += stemmer.stem(word) + " "
    stripped_stem_word = stem_word.strip() if stem_word else stem_word
    authors_list_first_stem.append(stripped_stem_word)
    authors_list.append(author)

# Indexing process
data_dict = {}
for i, stemmed_author in enumerate(authors_list_first_stem):
    for word in stemmed_author.split():
        if word not in data_dict:
            data_dict[word] = [i]
        else:
            data_dict[word].append(i)

# Write the preprocessed author names and indexed dictionary to JSON files
with open('author_list_stemmed.json', 'w') as f:
    ujson.dump(authors_list_first_stem, f)

with open('author_indexed_dictionary.json', 'w') as f:
    ujson.dump(data_dict, f)
     