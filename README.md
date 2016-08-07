rebecca_stylometry:

The purpose of this repository is to calculate the set of ads most likely to share the same author with a given ad. It takes in the text and title from scraped Backpage advertisements and outputs usable JSON data.

Design

To run the pipeline from start to finish, create an object from the Model class. Initialize the model with the desired regularization constant, and the path where the Stanford POS tagger (models/english-left3words-distsim.tagger, lib/stanford-postagger-3.4.1.jar) and tokenizer (lib/stanford-postagger-3.4.1.jar) are stored. Train the model using an existing training set, then get the features for ad x and ads m by calling “extract_comparison_features”. Use those extracted features as input to “get_top_similar”, which outputs the ranked JSON data.

Input

Currently, the code queries the CDR for the text and title of an ad, given the cdr_id, in order to extract the features. Some portion of these features (e.g., the part-of-speech) that are time consuming to compute will eventually be pre-computed and stored in S3 dumps.

Output

Currently, the code outputs one JSON blob for each cdr_id associated with the target ad x.

Features

POS : given a particular POS tag, calculate the Jaccard similarity of all words with that tag, from ad1 and ad2 (plus the title of ad1, and the title of ad2)
Names : calculate the Jaccard similarity of all names from ad1 and ad2 (plus the title of ad1, and the title of ad2)
Word n-grams : calculate the Jaccard similarity of all word n-grams from ad1 and ad2 (plus the title of ad1, and the title of ad2)
Char n-grams : calculate the Jaccard similarity of all character n-grams from ad1 and ad2 (plus the title of ad1, and the title of ad2)
Breaklines : extract the distance of each breakline symbol from the subsequent appearance. 
	(1) calculate the difference in length of the two resulting vectors from ad1 and ad2
	(2) calculate the cosine similarity of the two resulting vectors from ad1 and ad2
Websites : calculate the Jaccard similarity of all websites from ad1 and ad2 (plus the title of ad1, and the title of ad2)

To-Do

Build and assess model from all of the scraped Memex data
Pre-compute and store time consuming features 
Design and implement clustering of ads from ranked output