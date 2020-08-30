# -*- coding: utf-8 -*-

import os

def _read_data_file(file_path, train=True):
    sentences = []
    sentence = [[], [], []]
    for line in open(file_path, encoding="utf-8"):
        line = line.strip()
        if line == "":
            sentences.append(sentence)
            sentence = [[], [], []]
        else:
            idx, ejeol, ner_tag = line.split("\t")
            sentence[0].append(int(idx))
            sentence[1].append(ejeol)
            if train:
                sentence[2].append(ner_tag)
            else:
                sentence[2].append("-")

    return sentences

def test_data_loader(root_path):
    # [ idx, ejeols, nemed_entitis ] each sentence
    file_path = os.path.join(root_path, 'test.txt')

    return _read_data_file(file_path, False)

def data_loader(root_path):
    # [ idx, ejeols, nemed_entitis ] each sentence
    file_path = os.path.join(root_path, 'train.txt')

    return _read_data_file(file_path)

if __name__ == "__main__":
    sentences = data_loader("data_in")
    print(sentences[0])
    sentences = test_data_loader("data_in")
    print(sentences[0])
