from __future__ import unicode_literals, print_function, division
import os
import argparse
from time import gmtime, strftime
import copy
import math
import random
from io import open

import torch
import numpy as np
from torch import nn, optim
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

from model.seq2seq_storybaseline import Story_model
import model.preprocess_keywords as dataIt
from model.utils import load_evalIters, get_loss, save_model, eval_saved_model, save_model_dict, load_model_dict

def evaluate(epoch, model, iterator, criterion, data, adaptive_softmax = None):
    model.eval()
    epoch_loss = 0
    total_bleu_score = 0

    model.is_train = False
    with torch.no_grad():
        for i, batch in enumerate(iterator):

            src1 = batch.src1
            keyword_scores1 = batch.keyword_scores1

            trg = batch.trg

            output, attention_weights, generated_sequence, total_loss, last_loss, summed_attentions, total_coverage_loss  = model(
                src1,
                keyword_scores1,
                trg,
                teacher_forcing_ratio=1)

            if adaptive_softmax:
                loss = last_loss
                generated_sequence = output
            else:
                generated_sequence = torch.argmax(output, dim=2)
                output = output[1:].view(-1, output.shape[-1])
                trg_loss = trg[1:].view(-1)
                loss = criterion(output, trg_loss)

            total_bleu_score += get_blue_per_batch( generated_sequence, trg, data.TRG)
            epoch_loss += loss.item()

    print("Validation Bleu Score for this epoch ", epoch+1, ": " , total_bleu_score / len(iterator))

    return epoch_loss / len(iterator)

def test(args,epoch, model, iterator, data):
    model.eval()
    dir_path = args.keyword_attention + "_" + str(args.itf_loss) + "/"

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    output_file_name = dir_path + '/keyadd_outputs_story_seqs_' + str(epoch) + '.txt'
    output_file = open(output_file_name, 'w')
    output_story_file_name = dir_path + '/keyadd_outputs_storys_' + str(epoch) + '.txt'
    output_story_file = open(output_story_file_name, 'w')
    output_story_key_file_name = dir_path + '/keyadd_outputs_story_key_' + str(epoch) + '.txt'
    output_story_key_file = open(output_story_key_file_name, 'w')
    epoch_loss = 0
    total_bleu_score = 0

    model.is_train = False
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src1 = batch.src1
            keyword_scores1 = batch.keyword_scores1
            trg = batch.trg

            output, attention_weights, generated_sequence, total_loss, last_loss, summed_attentions, total_coverage_loss  = model(
                src1,
                keyword_scores1,
                trg,
                teacher_forcing_ratio=0)

            print("True Target:")
            print_outputs_in_words(data.TRG, trg)
            print("Predicted Target:")

            print_outputs_in_words(data.TRG, generated_sequence)
            total_bleu_score += get_blue_per_batch( generated_sequence, trg, data.TRG)

            generate_predictions(batch, generated_sequence, data.TRG, output_file)
            generate_predictions(batch, generated_sequence, data.TRG, output_story_file, type = 'story')
            generate_predictions_keywords(batch, generated_sequence, data.TRG, output_story_key_file, type = 'story')


    print("Test Bleu Score for this epoch ", epoch+1, ": " , total_bleu_score / len(iterator))
    output_file.close()
    output_story_file.close()
    output_story_key_file.close()

    return epoch_loss / len(iterator)



def get_bleu_per_sentence(candidate_sentence, reference_sentence, n_gram_weights = (1, 0)):
    candidate_sentence.pop(0)
    reference_sentence.pop(0)
    candidate_sentence = list(filter(lambda x:x not in ["<pad>", "<sos>"], candidate_sentence))

    new_candidate_sentence = []
    for word in candidate_sentence:
        if word == "<eos>":
            break
        new_candidate_sentence.append(word)

    candidate_sentence = new_candidate_sentence
    reference_sentence = [list(filter(lambda x: x not in ["<pad>", "<sos>","<eos>"], reference_sentence))]
    bleu_score = sentence_bleu(reference_sentence, candidate_sentence, weights=n_gram_weights)

    return bleu_score

def get_blue_per_batch(generated_sequences, target_tensors, TRG):
    generated_sequences = torch.tensor(np.array([i.detach().cpu().numpy() for i in generated_sequences]))
    generated_sequences = generated_sequences.permute(1,0).tolist()
    target_tensors = target_tensors.permute(1,0).tolist()
    total_blue_score_per_batch = 0

    for generated_sequence, target_tensor in zip(generated_sequences, target_tensors):
        generated_words = [TRG.vocab.itos[x] for x in generated_sequence]
        true_words = [TRG.vocab.itos[x] for x in target_tensor]
        blue_score = get_bleu_per_sentence(generated_words, true_words)
        total_blue_score_per_batch += blue_score

    return total_blue_score_per_batch/ len(generated_sequences)


def print_outputs_in_words(TRG, generated_sequence):
    if type(generated_sequence) == list:
        generated_sequence = torch.tensor(np.array([i.detach().cpu().numpy() for i in generated_sequence]))
    generated_sequence = generated_sequence.permute(1, 0).tolist()
    words = [[TRG.vocab.itos[ind] for ind in ex] for ex in generated_sequence]
    print("Output ", " ".join(words[0]))

def get_outputs_in_words(TRG, generated_sequence):
    if type(generated_sequence) == list:
        generated_sequence = torch.tensor(np.array([i.detach().cpu().numpy() for i in generated_sequence]))
    generated_sequence = generated_sequence.permute(1, 0).tolist()
    words = [[TRG.vocab.itos[ind] for ind in ex] for ex in generated_sequence]
    return [' '.join(ex) for ex in words]

def word_ids_to_sentence(id_tensor, vocab):
    if not isinstance(id_tensor, (list,)):
        id_tensor = id_tensor.t()
        ids = id_tensor.tolist()
    else:
        ids = id_tensor

    batch = [[vocab.itos[ind] for ind in ex] for ex in ids]

    def filter_special(tok):
        return tok not in ('<sos>', "<pad>")

    batch = [filter(filter_special, ex) for ex in batch]

    return [' '.join(ex) for ex in batch]


def generate_predictions(batch, output, TRG, f, type = 'seqs'):
    vocab = TRG.vocab
    src1 = word_ids_to_sentence(batch.src1, vocab)
    trg = word_ids_to_sentence(batch.trg, vocab)
    output_sentence = get_outputs_in_words(TRG, output)

    if type == 'seqs':
        zipped_sentences = list(zip(trg, output_sentence))
    else:
        zipped_sentences = list(zip(src1, trg, output_sentence))

    for item in zipped_sentences:
        f.write('\t'.join(s for s in item) + '\n')

def generate_predictions_keywords(batch, output, TRG, f, type = 'seqs'):
    vocab = TRG.vocab
    src1 = word_ids_to_sentence(batch.src1, vocab)
    trg = word_ids_to_sentence(batch.trg, vocab)
    output_sentence = get_outputs_in_words(TRG, output)
    keywords = batch.posts1_keywords
    keywords_joined = [','.join(ex) for ex in keywords]

    if type == 'seqs':
        zipped_sentences = list(zip(trg, output_sentence))
    else:
        zipped_sentences = list(zip(src1, trg, output_sentence, keywords_joined))

    for item in zipped_sentences:
        f.write('\t'.join(s for s in item) + '\n')

def train(model, iterator, optimizer, criterion, clip, kl_criterion, adaptive_softmax = None, keyword_attention = 'none'):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(tqdm(iterator, desc="train")):

        src1 = batch.src1
        keyword_scores1 = batch.keyword_scores1
        trg = batch.trg

        optimizer.zero_grad()

        probability_distributions, attention_weights, generated_seq, total_loss, last_loss, summed_attentions, total_coverage_loss = model(
            src1,
            keyword_scores1,
            trg)

        if adaptive_softmax:
            loss = last_loss
        else:
            output = probability_distributions
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            if keyword_attention !='att_sum_mse' or keyword_attention !='att_sum_coverage' :
                loss = criterion(output, trg)
            else:
                summed_attentions = F.normalize(summed_attentions, p=1, dim=1)
                reconstruction_loss = criterion(output, trg)
                keyword_scores1 = keyword_scores1.permute(1,0)

                if keyword_attention =='att_sum_mse':
                    keyword_loss = kl_criterion(summed_attentions,keyword_scores1)
                    loss = reconstruction_loss + keyword_loss
                else:
                    lamda = 10.0
                    loss = reconstruction_loss + lamda * total_coverage_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.95**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def trainIters(args, model, data):
    optimizer = optim.Adam(model.parameters())
    pad_idx = data.TRG.vocab.stoi['<pad>']
    vocab_frequency = data.vocab_frequency
    if args.itf_loss:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, weight = torch.tensor(vocab_frequency).to(args.device))
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    kl_criterion = nn.MSELoss()
    train_iterator = data.train_iterator
    valid_iterator = data.valid_iterator
    test_iterator = data.test_iterator

    best_valid_loss = 10000000
    best_model = None

    for epoch in range(args.epoch):

        train_loss = train(model, train_iterator, optimizer, criterion, args.norm, kl_criterion, adaptive_softmax=args.adaptivesoftmax, keyword_attention = args.keyword_attention)
        test(args,epoch, model, test_iterator, data, adaptive_softmax=args.adaptivesoftmax)
        valid_loss = evaluate(epoch, model, valid_iterator, criterion, data, adaptive_softmax=args.adaptivesoftmax)
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        if valid_loss<best_valid_loss:
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
            save_model_dict(args, best_model, optimizer, epoch, train_loss, model_type = "best")
        else:
            print('val perplexity decreased, decreasing lr')
            exp_lr_scheduler(optimizer, epoch, init_lr=0.001)

    save_model_dict(args, model, optimizer, args.epoch, train_loss, model_type = "last")

    return best_model, model



def main(data = None):

    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ROCStoriesGA')
    parser.add_argument('--model_name')
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--batchsize', default=128, type=int)
    parser.add_argument('--learningrate', default=0.1, type=float)
    parser.add_argument('--wordembsize', default=200, type=int)
    parser.add_argument('--vocabsize', default=15000, type=int)
    parser.add_argument('--norm', default=5.0, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--useglove', default=True, type=bool)
    parser.add_argument('--numworkers', default=4, type=int)
    parser.add_argument('--numlayers', default=2, type=int)
    parser.add_argument('--numunits', default=512, type=int)
    parser.add_argument('--istrain', default=True, type=bool)
    parser.add_argument('--learning_rate_decay_factor', default=0.95, type=float)
    parser.add_argument('--numsamples', default=512, type=int)
    parser.add_argument('--maxlength', default=30, type=int)
    parser.add_argument('--uselstm', default=True, type=bool)
    parser.add_argument('--bidirectional', default=False, type=bool)
    parser.add_argument("--triple_num", default=10, type=int, help= "max number of triple for each query")
    parser.add_argument('--teacherforcingratio', default=0.5, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('-f', default=100, type=str)
    parser.add_argument('--adaptivesoftmax', default=False, type=bool)

    parser.add_argument('--keywords_to_use', default=100, type=float)
    parser.add_argument('--num_keyword_samples', default=1, type=float)
    parser.add_argument('--lambda_weight', default=0.3, type=float)
    parser.add_argument('--lambda_inference', default=1.0, type=float)
    parser.add_argument('--keyword_attention', default='context_add', type=str, help="none, alpha_add, context_add, att_sum_mse, att_sum_coverage" )
    parser.add_argument('--itf_loss', default=True, type=bool)

    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    setattr(args, 'device', device)
    if args.model_name is None:
        setattr(args, 'model_name', strftime('%d%Y-%H%M%S', gmtime()))

   #load data
    if data is None:
        print("Preparing data")
        data = dataIt.ROCstories_data(args, datasetname = args.dataset)#ROCStories # Multi30k
    else:
        print("Have data from before")

    setattr(args, 'input_vocab_size', len(data.TRG.vocab))
    setattr(args, 'output_vocab_size', len(data.TRG.vocab))

    # choose the model to use
    gen_model = Story_model
    # initialize the model
    model = gen_model(args, data.TRG, embed = data.TRG.vocab.vectors)#, vocab, embed)

    if args.evaluate == True:
        print("Only evaluating")
        load_evalIters(args, model, data)
        return

    #run training loops
    best_model, last_model = trainIters(args, model, data)
    print('Finished training')



if __name__== "__main__":
    main()
