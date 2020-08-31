#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
import argparse
from model import Model
from dataset_batch import Dataset
from data_loader import data_loader
from evaluation import diff_model_label, calculation_measure, calculation_measure_ensemble
from scipy import stats
from tqdm import tqdm


def iteration_model(models, dataset, parameter, train=True):
    precision_count = np.zeros((parameter["num_ensemble"], 2))
    recall_count = np.zeros((parameter["num_ensemble"], 2))
    avg_cost = np.zeros(parameter["num_ensemble"])
    avg_correct = np.zeros(parameter["num_ensemble"])
    total_labels = np.zeros(parameter["num_ensemble"])
    correct_labels = np.zeros(parameter["num_ensemble"])
    dataset.shuffle_data()

    e_precision_count = np.array([ 0. , 0. ])
    e_recall_count = np.array([ 0. , 0. ])
    e_avg_correct = 0.0
    e_total_labels = 0.0

    if train:
        keep_prob = parameter["keep_prob"]
    else:
        keep_prob = 1.0

    batch_gen = dataset.get_data_batch_size(parameter["batch_size"], train)
    total_iter = int(len(dataset) / parameter["batch_size"] + 1)

    for morph, ne_dict, character, seq_len, char_len, label, step in tqdm(batch_gen, total=total_iter):
        ensemble = []

        for i, model in enumerate(models):
            feed_dict = {model.morph: morph,
                         model.ne_dict: ne_dict,
                         model.character: character,
                         model.sequence: seq_len,
                         model.character_len: char_len,
                         model.label: label,
                         model.dropout_rate: keep_prob
                         }
            if train:
                cost, tf_viterbi_sequence, _ = sess.run([model.cost, model.viterbi_sequence, model.train_op], feed_dict=feed_dict)
            else:
                cost, tf_viterbi_sequence = sess.run([model.cost, model.viterbi_sequence], feed_dict=feed_dict)
            ensemble.append(tf_viterbi_sequence)

            avg_cost[i] += cost

            mask = (np.expand_dims(np.arange(parameter["sentence_length"]), axis=0) <
                                np.expand_dims(seq_len, axis=1))
            total_labels[i] += np.sum(seq_len)

            correct_labels[i] = np.sum((label == tf_viterbi_sequence) * mask)
            avg_correct[i] += correct_labels[i]
            precision_count[i], recall_count[i] = diff_model_label(dataset, precision_count[i], recall_count[i], tf_viterbi_sequence, label, seq_len)

        # Calculation for ensemble measure
        ensemble = np.array(stats.mode(ensemble)[0][0])

        mask = (np.expand_dims(np.arange(parameter["sentence_length"]), axis=0) <
                np.expand_dims(seq_len, axis=1))
        e_total_labels += np.sum(seq_len)

        e_correct_labels = np.sum((label == ensemble) * mask)
        e_avg_correct += e_correct_labels
        e_precision_count, e_recall_count = diff_model_label(dataset, e_precision_count, e_recall_count,
                                                               ensemble, label, seq_len)



    return avg_cost / (step + 1), 100.0 * avg_correct / total_labels.astype(float), precision_count, recall_count, \
        100.0 * e_avg_correct / e_total_labels.astype(float), e_precision_count, e_recall_count



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0] + " description")

    parser.add_argument('--mode', type=str, default="train", required=False, help='Choice operation mode')
    parser.add_argument('--input_dir', type=str, default="data_in", required=False, help='Input data directory')
    parser.add_argument('--output_dir', type=str, default="data_out", required=False, help='Output data directory')
    parser.add_argument('--necessary_file', type=str, default="necessary.pkl")

    parser.add_argument('--epochs', type=int, default=20, required=False, help='Epoch value')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, required=False, help='Set learning rate')
    parser.add_argument('--keep_prob', type=float, default=0.65, required=False, help='Dropout_rate')

    parser.add_argument("--word_embedding_size", type=int, default=128, required=False, help='Word, WordPos Embedding Size')
    parser.add_argument("--char_embedding_size", type=int, default=128, required=False, help='Char Embedding Size')
    parser.add_argument("--tag_embedding_size", type=int, default=128, required=False, help='Tag Embedding Size')

    parser.add_argument('--lstm_units', type=int, default=128, required=False, help='Hidden unit size')
    parser.add_argument('--char_lstm_units', type=int, default=128, required=False, help='Hidden unit size for Char rnn')
    parser.add_argument('--sentence_length', type=int, default=180, required=False, help='Maximum words in sentence')
    parser.add_argument('--word_length', type=int, default=8, required=False, help='Maximum chars in word')
    parser.add_argument('--num_ensemble', type=int, default=1, required=False, help='Number of submodels')


    try:
        parameter = vars(parser.parse_args())
    except:
        parser.print_help()
        sys.exit(0)

    # Creating various information and training sets using the sentence-specific data set
    train_data = data_loader(parameter["input_dir"])
    train_loader = Dataset(parameter, train_data)
    test_data = data_loader(parameter["input_dir"])
    test_loader = Dataset(parameter, test_data)

    # Load model
    models = []
    for i in range(parameter["num_ensemble"]):
        models.append(Model(train_loader.parameter, i))
        models[i].build_model()

    # tensorflow session init
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Training
    if parameter["mode"] == "train":
        train_data = data_loader(parameter["input_dir"])
        train_loader.make_input_data(train_data)
        test_data = data_loader(parameter["input_dir"])
        test_loader.make_input_data(test_data)

        for epoch in range(parameter["epochs"]):
            # Training
            avg_cost, avg_correct, precision_count, recall_count, e_avg_correct, e_precision_count, e_recall_count = iteration_model(models, train_loader, parameter)
            # Individual and ensemble model's accuracy score
            for i in range(parameter["num_ensemble"]):
                print(str(i) + '_[Epoch: {:>4}] cost = {:>.6} Accuracy = {:>.6}'.format(epoch + 1, avg_cost[i], avg_correct[i]))
            print('Ensemble [Epoch: {:>4}]  Accuracy = {:>.6}'.format(epoch + 1, e_avg_correct))
            f1Measure, precision, recall = calculation_measure(parameter["num_ensemble"], precision_count, recall_count)

            # Individual and ensemble model's f1, precision and recall
            e_f1Measure, e_precision, e_recall = calculation_measure_ensemble(e_precision_count, e_recall_count)
            for i in range(parameter["num_ensemble"]):
                print(str(i) + '_[Train] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(f1Measure[i], precision[i], recall[i]))
            print('Ensemble [Train] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(e_f1Measure, e_precision, e_recall))
            print('='*100)

            # Inference validation / test set
            avg_cost, avg_correct, precision_count, recall_count, e_avg_correct, e_precision_count, e_recall_count = iteration_model(models, test_loader, parameter, False)
            # Individual and ensemble model's accuracy score on validation or test dataset
            for i in range(parameter["num_ensemble"]):
                print(str(i) + '_Val : [Epoch: {:>4}] cost = {:>.6} Accuracy = {:>.6}'.format(epoch + 1, avg_cost[i], avg_correct[i]))
            print('Ensemble [Epoch: {:>4}]  Accuracy = {:>.6}'.format(epoch + 1, e_avg_correct))
            f1Measure, precision, recall = calculation_measure(parameter["num_ensemble"], precision_count, recall_count)

            # Individual and ensemble model's f1, precisionb and recall score on validation or test dataset
            e_f1Measure, e_precision, e_recall = calculation_measure_ensemble(e_precision_count, e_recall_count)
            for i in range(parameter["num_ensemble"]):
                print(str(i) + '_Val : [Val] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(f1Measure[i], precision[i], recall[i]))
            print('Ensemble [Val] F1Measure : {:.6f} Precision : {:.6f} Recall : {:.6f}'.format(e_f1Measure, e_precision,  e_recall))
            print('=' * 100)