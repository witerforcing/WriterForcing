from __future__ import unicode_literals, print_function, division

import argparse
from io import open
import matplotlib; matplotlib.use('agg')
import spacy
nlp = spacy.load('en')

def main():
    parser = argparse.ArgumentParser()
    KEYWORDS_FROM_FILE = False

    parser.add_argument('--file', type=str, help="none, alpha_add, context_add, att_sum_mse, att_sum_coverage" )
    args = parser.parse_args()

    pop_first = True

    output_file = args.file
    sentences = open(output_file, 'r').readlines()


    score = 0.
    score_bleu2 = 0.


    def clean_sentence(candidate_sentence, pop_first = False, lemmatize = True):
        if pop_first:
            candidate_sentence.pop(0)
        candidate_sentence = list(filter(lambda x:x not in ["<pad>", "<sos>"], candidate_sentence))
        new_candidate_sentence = []
        for word in candidate_sentence: 
            if word == "<eos>":
                break
            new_candidate_sentence.append(word)
        result = ' '.join(new_candidate_sentence)
        result = result.replace('<unk>', 'unk')
        if lemmatize:
            doc = nlp(result)
            result = ' '.join([token.lemma_ for token in doc])
            
        
        return result


    def distinct_ngrams(inputs, n):
        output = {}
        for input in inputs:
            for i in range(len(input)-n+1):
                g = ' '.join(input[i:i+n])
                output.setdefault(g, 0)
                output[g] += 1
        if sum(output.values())==0:
            ratio = 0
        else:
            ratio = float(len(output.keys()))/ sum(output.values())

        return ratio

    def get_keywords(sentence):
        r.extract_keywords_from_text(sentence)
        keywords = r.get_ranked_phrases_with_scores()
        
        return keywords

    def get_keywords_set(sentence):
        r.extract_keywords_from_text(sentence)
        keywords = r.get_ranked_phrases_with_scores()
        set_keywords = set()
        for keyword in keywords:
            set_keywords.update(keyword[1].split())
        set_keywords.discard('unk')
        set_keywords.discard('pron')

        return set_keywords

    def get_keywords_set_from_list(keywords_list):
        set_keywords = set()
        for keywords in keywords_list:
            for keyword in keywords.split():
                set_keywords.add(keyword)
        set_keywords.discard('unk')
        set_keywords.discard('pron')

        return set_keywords

    list_references = []
    list_hypothesis= []
    list_score_candidate = []
    list_score_reference = []
    output_text_reference_list = []
    output_text_candidate_list = []
    for i in range(len(sentences)):
        line = sentences[i].split('\t')
        context = line[0]
        context = clean_sentence(context.split())
        reference = line[1]
        reference = clean_sentence(reference.split())
        output_text_reference_list.append(reference.split())
        candidate = line[2]
        candidate = clean_sentence(candidate.split(), pop_first = True)
        output_text_candidate_list.append(candidate.split())
        


    distinct_1 = distinct_ngrams(output_text_reference_list, 1)
    distinct_1_cand = distinct_ngrams(output_text_candidate_list, 1)
    print('Distinct-1', distinct_1, distinct_1_cand)
    distinct_2 = distinct_ngrams(output_text_reference_list, 2)
    distinct_2_cand = distinct_ngrams(output_text_candidate_list, 2)
    print('Distinct-2', distinct_2, distinct_2_cand)
    distinct_3 = distinct_ngrams(output_text_reference_list, 3)
    distinct_3_cand = distinct_ngrams(output_text_candidate_list, 3)
    print('Distinct-3', distinct_3, distinct_3_cand)


if __name__== "__main__":
    main()
