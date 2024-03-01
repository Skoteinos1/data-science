'''
Read the introduction about garden path sentences and study a few of the examples provided on Wikipedia.
Find at least 2 garden path sentences from the web, or think up your own.
Store the sentences you have identified or created in a list called gardenpathSentences.

Add the following sentences to your list:
○ Mary gave the child a Band-Aid.
○ That Jill is never here hurts.
○ The cotton clothing is made of grows in Mississippi.

Tokenize each sentence in the list, and perform named entity recognition.
Examine how spaCy has categorised each sentence. Then, use spacy.explain to look up and print the meaning of entities that you don’t
understand. For example: print(spacy.explain("FAC"))

At the bottom of your file, write a comment about two entities that you looked up. For each entity answer the following questions:
○ What was the entity and its explanation that you looked up?
○ Did the entity make sense in terms of the word associated with it?
'''

import spacy
# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_md')

# Store sentences from internet
gardenpathSentences = [
    'The old man the boat.',
    'The complex houses married and single soldiers and their families.',
    'The horse raced past the barn fell.',
    ]

# Add more sentences to the list
gardenpathSentences.extend([
    'Mary gave the child a Band-Aid.',
    'That Jill is never here hurts.',
    'The cotton clothing is made of grows in Mississippi.'])

# Tokenization and named entity recognition.
for sentence in gardenpathSentences:
    doc = nlp(sentence)
    doc.text.split()
    [token.orth_ for token in doc]
    print([(token, token.orth_, token.orth) for token in doc], '\n')
    
    # Get labels and entities and print them
    nlp_sentence = nlp(sentence)
    print([(i, i.label_, i.label) for i in nlp_sentence.ents], '\n\n')
    

entity_fac = spacy.explain("GPE")
print(f"GPE:{entity_fac}")
entity_fac = spacy.explain("LOC")
print(f"LOC:{entity_fac}")
entity_fac = spacy.explain("PERSON")
print(f"PERSON:{entity_fac}")


'''
○ What was the entity and its explanation that you looked up?
Mary - PERSON - People, including fictional

○ Did the entity make sense in terms of the word associated with it?
Yes.


○ What was the entity and its explanation that you looked up?
Mississippi - GPE - Countries, cities, states

○ Did the entity make sense in terms of the word associated with it?
Yes. Cotton doesn't grow in river.

'''


