'''code modified from https://github.com/JianGuanTHU/StoryEndGen'''

import sys
import time
from random import shuffle
import random
import argparse
import torch
import math
import numpy as np
from rake_nltk import Rake

UNK_ID = 0
PAD_ID = 1
SOS_ID = 2
EOS_ID = 3
_START_VOCAB = [ '<unk>', '<pad>', '<sos>', '<eos>']
r = Rake()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Vocab:
    def __init__(self, vocab, vectors):
        self.vocab = vocab
        self.stoi = {id: i for i, id in enumerate(vocab)}
        self.itos = {i: id for i, id in enumerate(vocab)}
        self.vectors = vectors
        self._data_len = len(self.vocab)

    def __len__(self):
        return self._data_len

class Target_class:
    """ TO immitate a field object in torchtext which holds a vocab object"""

    def __init__(self, vocab, vectors):
        self.vocab = Vocab(vocab, vectors)

class DataTestIterator:
    """Class to implement an iterator"""

    def __init__(self, datasetobject, dataset, batchsize, test = False, samples = 1):

        self.datasetobject = datasetobject
        self.dataset = dataset
        self.max = len(dataset) 
        self.batchsize = batchsize
        self.num_repeats = samples

    def __len__(self):

        return math.ceil(len(self.dataset) *self.num_repeats/ self.batchsize)

    def __iter__(self):
        self.n = 0

        return self

    def __next__(self):
        if self.n < self.max:
            bound = self.n + int(self.batchsize /self.num_repeats)
            # if bound is greater than length of dataset
            if bound > self.max:
                bound = self.max

            # print(self.n,bound, len(self.dataset))
            result = self.datasetobject.gen_batched_ids_test(self.dataset[self.n:bound], self.num_repeats)

            self.n += int(self.batchsize/self.num_repeats)
            return result
        else:
            raise StopIteration



class DataIterator:
    """Class to implement an iterator"""

    def __init__(self, datasetobject, dataset, batchsize, test = False):

        self.datasetobject = datasetobject
        self.dataset = dataset
        self.max = len(dataset) 
        self.batchsize = batchsize

    def __len__(self):

        return math.ceil(len(self.dataset)/ self.batchsize)

    def __iter__(self):
        self.n = 0

        return self

    def __next__(self):
        if self.n < self.max:
            bound = self.n + self.batchsize
            # if bound is greater than length of dataset
            if bound > self.max:
                bound = self.max

            result = self.datasetobject.gen_batched_ids(self.dataset[self.n:bound])
            self.n += self.batchsize

            return result
        else:
            raise StopIteration




class ROCstories_data:

    def __init__(self, args, data_dir = None, datasetname = "ROCStories", is_shuffle = True, ie_only = True):
        print("Initializing ROCstories_data class")
        self.FLAGS = args
        self.ie_only = ie_only
        self.train_dataset, self.dev_dataset, self.test_dataset = self.setup_alldata(args, data_dir)
        if is_shuffle:
            shuffle(self.train_dataset)
        self.train_iterator = DataIterator(self, self.train_dataset, self.FLAGS.batchsize)
        self.valid_iterator = DataIterator(self, self.dev_dataset, self.FLAGS.batchsize)
        self.test_iterator = DataTestIterator(self, self.test_dataset, self.FLAGS.batchsize, test= True, samples = args.num_keyword_samples)

    def setup_alldata(self, args, data_dir = None):

        # global FLAGS, relation, vocab_dict
        if data_dir == None:
            data_dir = 'data'
        data_dev = self.load_data(data_dir, 'val')
        data_train = self.load_data(data_dir, 'train_utf')
        data_test = self.load_data(data_dir, 'test')
        self.vocab, self.embed, self.vocab_dict, self.vocab_frequency = self.build_vocab(data_dir, data_train)
        print(len(self.vocab), " length of vocabulary")
        # self.relation = self.load_relation(data_dir)

        self.TRG = Target_class(self.vocab, self.embed)

        return data_train, data_dev, data_test

    def load_data(self, path, fname):
        post = []
        with open('%s/%s.post' % (path, fname), encoding="utf8") as f:
            for line in f:
                tmp = [" ".join(line.strip().split("\t"))]#line.strip().split("\t")
                post.append([p.split() for p in tmp])

        with open('%s/%s.response' % (path, fname), encoding="utf8") as f:
            response = [line.strip().split() for line in f.readlines()]
        data = []
        for p, r in zip(post, response):
            data.append({'post': p, 'response': r})

        print(len(data), ' datalines')

        return data

    def load_relation(self, path):
        file = open('%s/triples_shrink.txt' % (path), "r")

        relation = {}
        numl = 0
        for line in file:
            numl+=1
            tmp = line.strip().split()
            if tmp[0] in relation:
                if tmp[2] not in relation[tmp[0]]:
                    relation[tmp[0]].append(tmp)
            else:
                relation[tmp[0]] = [tmp]
        for r in relation.keys():
            tmp_vocab = {}
            i = 0
            for re in relation[r]:
                if re[2] in self.vocab_dict.keys():
                    tmp_vocab[i] = self.vocab_dict[re[2]]
                i += 1
            tmp_list = sorted(tmp_vocab, key=tmp_vocab.get)[:self.FLAGS.triple_num] if len(tmp_vocab) > self.FLAGS.triple_num else sorted(tmp_vocab, key=tmp_vocab.get)
            new_relation = []
            for i in tmp_list:
                new_relation.append(relation[r][i])
            relation[r] = new_relation

        return relation

    def build_vocab(self, path, data):
        print("Creating vocabulary...")
        relation_vocab_list = []
        relation_file = open(path + "/relations.txt", "r")
        for line in relation_file:
            relation_vocab_list += line.strip().split()
        vocab = {}
        for i, pair in enumerate(data):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            for token in [word for p in pair['post'] for word in p]+pair['response']:
                token = token.lower()
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        # vocab_list = _START_VOCAB + relation_vocab_list + sorted(vocab, key=vocab.get, reverse=True)
        print('len vocab_list', len(vocab_list))

        vocab_frequency = {}
        lamda = self.FLAGS.lambda_weight
        for item in vocab_list:
            a_freq = vocab.get(item, 3000)
            frequency = 1.0/(a_freq**lamda)
            # print(item, frequency)
            vocab_frequency[item] = frequency

        vocab_frequency = list(vocab_frequency.values())

        if len(vocab_list) > self.FLAGS.vocabsize:
            vocab_list = vocab_list[:self.FLAGS.vocabsize]

        if len(vocab_frequency) > self.FLAGS.vocabsize:
            vocab_frequency = vocab_frequency[:self.FLAGS.vocabsize]

        print("Loading word vectors...")
        vectors = {}
        with open(path + '/glove.6B.200d.txt', 'r', encoding="utf8") as f:
            for i, line in enumerate(f):
                if i % 100000 == 0:
                    print("    processing line %d" % i)
                s = line.strip()
                word = s[:s.find(' ')]
                vector = s[s.find(' ')+1:]
                vectors[word] = vector

        embed = []
        num_word_present = 0
        total_words = 0
        for word in vocab_list:
            if word in vectors:
                vector = list(map(float,  vectors[word].split()))
                num_word_present+=1
            else:
                # vector = np.zeros((self.FLAGS.wordembsize), dtype=np.float32)
                vector = np.random.uniform(-0.1, 0.1, self.FLAGS.wordembsize).astype('f')
                total_words+=1
            embed.append(vector)
        print(num_word_present, ' word embeddings found out of ',total_words + num_word_present, ' total words')
        embed = torch.tensor(embed)

        return vocab_list, embed, vocab, vocab_frequency


    def gen_batched_ids(self, data):
        '''
        :param
        data: batch of raw lines of text form the story data files
        :return
        batched_dict: tensorized batches of sentences, masks, triplet of entities for each word
        '''

        vocab = self.TRG.vocab
        encoder_len = [max([len(item['post'][0]) for item in data]) + 1 ]
        decoder_len = max([len(item['response']) for item in data]) + 1
        posts_1, posts_length_1, keyword_scores1, responses, responses_length = [], [], [], [], []

        def padding(sent, l):
            # return ['<sos>'] + sent + ['<eos>'] + ['<pad>'] * (l - len(sent) - 1)
            padded_sent = [vocab.stoi['<sos>']] #[2]
            complete_sentence = ' '.join(sent).lower()
            padded_keyword_scores = [0.0] * (len(sent) + 1) # extra 1 for sos
            r.extract_keywords_from_text(complete_sentence)
            keywords = r.get_ranked_phrases_with_scores()
            num_sample = self.FLAGS.keywords_to_use
            if len(keywords)<num_sample:
                num_sample = len(keywords)
            keywords = keywords[:num_sample]
            for keyword_tuple in keywords:
                keyword = keyword_tuple[1]
                keyword_score = keyword_tuple[0]
                indexk = complete_sentence.find(keyword)
                if indexk <0:
                    #case where keywords have speciall characters like ++ --- ???
                    continue
                #gives the index of the word at that indexk character
                else:
                    word_index = len(complete_sentence[:indexk].split(' '))
                    for pi in range(len(keyword.split(' '))):
                        # indexk is the forst word index, pi iterates over all works in keyphrase, 1 is for sos
                        padded_keyword_scores[word_index + pi ] = keyword_score
            word_ids = []
            for w in sent:
                try:
                    index_element = vocab.stoi[w.lower()]#vocab.index(w)
                except (ValueError, KeyError):
                    index_element = 0 # 0 is unk
                word_ids.append(index_element)
            padded_sent += word_ids
            # 3 for sos 1 for padding at end
            padded_sent += [3] + [1] * (l - len(sent) - 1)
            padded_keyword_scores+= [0.0] + [0.0] * (l - len(sent) - 1)
            score_sum = sum(padded_keyword_scores)
            if score_sum>0:
                padded_keyword_scores =[x / score_sum for x in padded_keyword_scores]
            
            return padded_sent, padded_keyword_scores

        for item in data:
            padded_post0, padded_keyword_scores = padding(item['post'][0], encoder_len[0])
            posts_1.append(padded_post0)
            keyword_scores1.append(padded_keyword_scores)
            posts_length_1.append(len(item['post'][0]) + 2)
            padded_res, padded_keyword_scores = padding(item['response'], decoder_len)
            responses.append(padded_res)
            responses_length.append(len(item['response']) + 1)
        ## if want to run ie model
        if self.ie_only:
            batched_data = {'src1': self.convert_to_tensor(posts_1).permute(1,0).contiguous(),
                    'keyword_scores1': self.convert_to_floattensor(keyword_scores1).permute(1,0).contiguous(),
                    'trg': self.convert_to_tensor(responses).permute(1,0).contiguous(),
                    'trg_length': responses_length}
            batched_dict = dotdict(batched_data)

            return batched_dict

    def gen_batched_ids_test(self, data, num_repeats):
        '''
        :param
        data: batch of raw lines of text form the story data files
        :return
        batched_dict: tensorized batches of sentences, masks, triplet of entities for each word
        '''

        vocab = self.TRG.vocab
        encoder_len = [max([len(item['post'][0]) for item in data]) + 1 ]
        decoder_len = max([len(item['response']) for item in data]) + 1
        posts_1, posts_length_1, keyword_scores1, responses, responses_length, posts1_keywords = [], [], [], [], [], []

        def padding(sent, l):
            list_padded_sent, list_padded_keyword_scores, list_keywords= [], [], []
            for i in range(num_repeats):
                # return ['<sos>'] + sent + ['<eos>'] + ['<pad>'] * (l - len(sent) - 1)
                padded_sent = [vocab.stoi['<sos>']] #[2]
                complete_sentence = ' '.join(sent).lower()
                padded_keyword_scores = [0.0] * (len(sent) + 1) # extra 1 for sos
                r.extract_keywords_from_text(complete_sentence)
                keywords = r.get_ranked_phrases_with_scores()
                num_sample = self.FLAGS.keywords_to_use
                if len(keywords)<num_sample:
                    num_sample = len(keywords)
                # keywords = random.sample(keywords,k=num_sample)
                keywords = keywords[:num_sample]
                for keyword_tuple in keywords:
                    keyword = keyword_tuple[1]
                    keyword_score = keyword_tuple[0]
                    indexk = complete_sentence.find(keyword)
                    if indexk <0:
                        #case where keywords have speciall characters like ++ --- ???
                        continue
                    #gives the index of the word at that indexk character
                    else:
                        word_index = len(complete_sentence[:indexk].split(' '))
                        for pi in range(len(keyword.split(' '))):
                            # indexk is the forst word index, pi iterates over all works in keyphrase, 1 is for sos
                            padded_keyword_scores[word_index + pi ] = keyword_score
                word_ids = []
                for w in sent:
                    try:
                        index_element = vocab.stoi[w.lower()]#vocab.index(w)
                    except (ValueError, KeyError):
                        index_element = 0 # 0 is unk
                    word_ids.append(index_element)
                padded_sent += word_ids
                # 3 for sos 1 for padding at end
                padded_sent += [3] + [1] * (l - len(sent) - 1)
                list_padded_sent.append(padded_sent)
                padded_keyword_scores+= [0.0] + [0.0] * (l - len(sent) - 1)
                score_sum = sum(padded_keyword_scores)
                if score_sum>0:
                    padded_keyword_scores =[x / score_sum for x in padded_keyword_scores]
                list_padded_keyword_scores.append(padded_keyword_scores)
                list_keywords.append([k[1] for k in keywords])
               
            return list_padded_sent, list_padded_keyword_scores, list_keywords

        for item in data:
            padded_post0, padded_keyword_scores, list_keywords = padding(item['post'][0], encoder_len[0])
            posts_1.extend(padded_post0)
            keyword_scores1.extend(padded_keyword_scores)
            posts1_keywords.extend(list_keywords)

            posts_length_1.extend([len(item['post'][0]) + 2]*num_repeats)
            padded_res, padded_keyword_scores, list_keywords = padding(item['response'], decoder_len)
            responses.extend(padded_res)
            responses_length.append([len(item['response']) + 1]*num_repeats)
        ## if want to run ie model
        if self.ie_only:
            batched_data = {'src1': self.convert_to_tensor(posts_1).permute(1,0).contiguous(),
                    'keyword_scores1': self.convert_to_floattensor(keyword_scores1).permute(1,0).contiguous(),
                    'trg': self.convert_to_tensor(responses).permute(1,0).contiguous(),
                    'trg_length': responses_length,
                    'posts1_keywords': posts1_keywords}
            batched_dict = dotdict(batched_data)

            return batched_dict



    def convert_to_tensor(self, x):

        return torch.tensor(x, dtype=torch.long, device=self.FLAGS.device)

    def convert_to_floattensor(self, x):

        return torch.tensor(x, dtype=torch.float, device=self.FLAGS.device)

    def convert_array_to_string(self, vocab, posts_array):
        words = [[vocab.itos[ind] for ind in ex] for ex in posts_array]

        return [' '.join(ex) for ex in words]

    def convert_entities_to_string(self, vocab, posts_array):
        triplet_array_all = []
        for ex in posts_array:
            triplet_array = []
            for word in ex:
                triplet_str = ''
                # print(ind)
                for tri in word:
                    triplet_str += ' ' + vocab.itos[tri[0]] + ' ' + vocab.itos[tri[1]] + ' ' + vocab.itos[tri[2]]
                triplet_array.append(triplet_str)
                print(triplet_str)
            triplet_array_all.append(triplet_array)
                # words = [[vocab.itos[ind[0]] + '' +vocab.itos[ind[1]] + '' + vocab.itos[ind[2]] ] for ind in ex] for ex in posts_array]
        
        return [' '.join(ex) for ex in triplet_array_all]




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", default=True, type=bool)#, True, "Set to False to inference.")
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument("--vocabsize", default=10000, type=int, help= "vocabulary size.")
    parser.add_argument("--wordembsize", default=200, type=int, help= "Size of word embedding.")
    parser.add_argument("--units", default=512, type=int, help= "Size of each model layer.")
    parser.add_argument("--layers", default=2, type=int, help= "Number of layers in the model.")
    parser.add_argument("--batch_size", default=128, type=int, help= "Batch size to use during training.")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--train_dir", default="../train", help="Training directory.")
    parser.add_argument("--per_checkpoint", default=1000, type=int, help= "How many steps to do per checkpoint.")
    parser.add_argument("--inference_version", default=0,type=int, help=  "The version for inferencing.")
    parser.add_argument("--triple_num", default=10, type=int, help= "max number of triple for each query")
    parser.add_argument("--log_parameters", default=True, type=bool, help= "Set to True to show the parameters")
    parser.add_argument("--inference_path", default="./inf.txt", help="Set filename of inference, default isscreen")
    parser.add_argument('--batchsize', default=12, type=int)

    FLAGS = parser.parse_args()

    device = torch.device('cuda:' + str(FLAGS.gpu) if torch.cuda.is_available() else 'cpu')
    setattr(FLAGS, 'device', device)

    data_dir = 'data'

    rc = ROCstories_data(FLAGS, data_dir, is_shuffle = False)
    print(rc.vocab[:50])
    print(rc.TRG.vocab.itos[0], rc.TRG.vocab.stoi['<pad>'])


    iterator = rc.test_iterator
    max_batches = 0
    print(len(iterator), 'iter length')
    for i, batch in enumerate(iterator):

        print('batch.src1.shape, batch.trg.shape')
        print(batch.src1.shape, batch.trg.shape, batch.keyword_scores1.shape)
        print(batch.keyword_scores1.t())
        if i == max_batches:
            break
