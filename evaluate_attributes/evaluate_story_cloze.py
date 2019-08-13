import json
from sklearn.metrics import confusion_matrix
import argparse
import os
import subprocess
import re


def format_into_bert_dst(examples, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for output, sentence, context in examples:
            sentence = sentence.strip()
            sentence = sentence.replace("<unk>", "")
            sentence = sentence.replace("<eos>", "")
            sentence = re.sub('\.', '', sentence)
            if(len(sentence) > 0):
                f.write(output + " ||| " + context + " ||| " + sentence + "\n")

def read_file(filename):
    examples = []
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.replace("\n",""), lines))
        for line in lines:
            temp = line.split("\t")
            context = temp[0]
            true_example = temp[1]
            predicted_example = temp[2]
            examples.append(("0", predicted_example, context))
    return examples

def read_output_file(filename):
    examples = []
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.replace("\n",""), lines))
        for line in lines:
            examples.append(line)
    return examples

def get_confusion_matrix(predicted_labels, true_labels):
    matrix = confusion_matrix(true_labels, predicted_labels)
    print(matrix)

def write_probabilities(predicted_probabilites, filename):
    with open(filename, "w") as f:
        for prob in predicted_probabilites:
            f.write(str(prob) + "\n")



def get_probabilites_bert(lines):
    predicted_labels = []
    for line in lines:
        example = line.split("\t")
        predicted_labels.append(float(example[2]))
    return predicted_labels

def evaluate_one_model(model_prediction_file, model_number):
    examples = read_file(model_prediction_file)
    stroy_cloze_file = "story-cloze" + str(model_number) + ".txt"
    format_into_bert_dst(examples, stroy_cloze_file)

    BERT_COMMAND = "python ../bert-classification-generic-story.py" \
                   " --task_name DST" \
                   " --do_eval --do_lower_case " \
                   "--data_dir ." \
                   " --input_train_file_name " + stroy_cloze_file + " --input_test_file_name " + stroy_cloze_file  +\
                   " --output_results_file_name eval_results.txt --output_predictions_file_name predictions.txt" \
                   " --bert_model bert-base-uncased" \
                   " --max_seq_length 200" \
                   " --train_batch_size 16 " \
                   "--learning_rate 2e-5" \
                   " --num_train_epochs 3.0 " \
                   "--output_dir ../../ --num_classes 2 --device_num " + str(args.gpu)

    process = subprocess.Popen(BERT_COMMAND, shell=True, stdout=subprocess.PIPE)
    process.wait()

    if process.returncode == 0:
        examples = read_output_file("../../predictions.txt")
        predicted_probabilites = get_probabilites_bert(examples)
        write_probabilities(predicted_probabilites, str(model_number))

def read_probs(filename):
    probs = []
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = list(filter(lambda x: x.strip(), lines))
        for line in lines:
            probs.append(float(line))
    return probs

def compare_models(model1_probs_file, model2_probs_file):
    probs1 = read_probs(model1_probs_file)
    probs2 = read_probs(model2_probs_file)
    better_outputs = []
    for i,j in zip(probs1, probs2):
        if i>j:
            better_outputs.append(0)
        else:
            better_outputs.append(1)
    return better_outputs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Pass the "storys" file present in the folders of the model's output directory based on the flags which has been set.
    parser.add_argument('--model1')
    parser.add_argument('--model2')
    parser.add_argument('--gpu',default=0)
    args = parser.parse_args()
    for i in range(10):
        model2 = args.model2 + str(i) + ".txt"
        evaluate_one_model(args.model1, 1)
        evaluate_one_model(model2, 2)
        better_outputs = compare_models("1","2")
        one_better = len(list(filter(lambda x: x == 0, better_outputs)))
        two_better = len(list(filter(lambda x: x == 1, better_outputs)))
        print("Number of things in which 1 is better=", one_better)
        print("Number of things in which 2 is better=", two_better)
        print("Number 1 percent:", one_better/(two_better + one_better))
        print("Number 1 percent:", two_better/(two_better + one_better))

