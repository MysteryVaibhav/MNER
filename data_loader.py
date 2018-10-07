from util import *
import torch.utils.data
import os
from collections import Counter
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, x, x_c, img_x, y, s_idx, e_idx):
        self.x = x
        self.x_c = x_c
        self.img_x = img_x
        self.y = y
        self.s_idx = s_idx
        self.e_idx = e_idx
        self.num_of_samples = e_idx - s_idx

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        # TODO: Add the character embeddings into the model
        x = self.x[self.s_idx + idx]
        y = self.y[self.s_idx + idx]
        img_x = self.img_x[self.s_idx + idx]
        return x, y, img_x

    def collate(self, batch):
        x = np.array([x[0] for x in batch])
        y = np.array([x[1] for x in batch])
        img_x = np.array([x[2] for x in batch])

        bool_mask = x == 0
        mask = 1 - bool_mask.astype(np.int)

        # index of first 0 in each row, if no zero then idx = -1
        zero_indices = np.where(bool_mask.any(1), bool_mask.argmax(1), -1).astype(np.int)
        input_len = np.zeros(len(batch))
        for i in range(len(batch)):
            if zero_indices[i] == -1:
                input_len[i] = len(x[i])
            else:
                input_len[i] = zero_indices[i]
        sorted_input_arg = np.flipud(np.argsort(input_len))

        # Sort everything according to the sequence length
        x = x[sorted_input_arg]
        y = y[sorted_input_arg]
        img_x = img_x[sorted_input_arg]
        mask = mask[sorted_input_arg]
        input_len = input_len[sorted_input_arg]

        max_seq_len = int(input_len[0])

        trunc_x = np.zeros((len(batch), max_seq_len))
        trunc_y = np.zeros((len(batch), max_seq_len))
        trunc_mask = np.zeros((len(batch), max_seq_len))
        for i in range(len(batch)):
            trunc_x[i] = x[i, :max_seq_len]
            trunc_y[i] = y[i, :max_seq_len]
            trunc_mask[i] = mask[i, :max_seq_len]

        return to_tensor(trunc_x).long(), to_tensor(img_x), to_tensor(trunc_y).long(), to_tensor(trunc_mask).long(), \
               to_tensor(input_len).int()


class DataLoader:
    def __init__(self, params):
        '''
        self.x : sentence encoding with padding at word level
        self.x_c : sentence encoding with padding at character level
        self.x_img : image features corresponding to the sentences
        self.y : label corresponding to the words in the sentences
        :param params:
        '''
        self.params = params
        self.id_to_vocb, self.word_matrix, \
            self.sentences, self.datasplit, \
            self.x, self.x_c, self.img_x, self.y, \
            self.num_sentence, self.vocb, \
            self.vocb_char, self.labelVoc \
            = self.load_data()
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

        dataset_train = CustomDataSet(self.x, self.x_c, self.img_x, self.y, self.datasplit[0], self.datasplit[1])
        self.train_data_loader = torch.utils.data.DataLoader(dataset_train,
                                                             batch_size=self.params.batch_size,
                                                             collate_fn=dataset_train.collate,
                                                             shuffle=True, **kwargs)
        dataset_val = CustomDataSet(self.x, self.x_c, self.img_x, self.y, self.datasplit[1], self.datasplit[2])
        self.val_data_loader = torch.utils.data.DataLoader(dataset_val,
                                                           batch_size=self.params.batch_size,
                                                           collate_fn=dataset_val.collate,
                                                           shuffle=False, **kwargs)
        dataset_test = CustomDataSet(self.x, self.x_c, self.img_x, self.y, self.datasplit[2], self.datasplit[3])
        self.test_data_loader = torch.utils.data.DataLoader(dataset_test,
                                                            batch_size=self.params.batch_size,
                                                            collate_fn=dataset_test.collate,
                                                            shuffle=False, **kwargs)

    def load_data(self):
        print('calculating vocabulary...')
        datasplit, sentences, img_to_feature, sent_maxlen, word_maxlen, num_sentence = self.load_sentence(
            'IMGID', self.params.split_file, 'train', 'dev', 'test')
        id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv = self.vocab_bulid(sentences)
        word_matrix = self.load_word_matrix(vocb, size=self.params.embedding_dimension)
        x, x_c, img_x, y = self.pad_sequence(sentences, img_to_feature, vocb, vocb_char, labelVoc,
                                             word_maxlen=self.params.word_maxlen, sent_maxlen=self.params.sent_maxlen)
        return [id_to_vocb, word_matrix, sentences, datasplit, x, x_c, img_x, y, num_sentence, vocb, vocb_char,
                labelVoc]

    def load_sentence(self, IMAGEID, tweet_data_dir, train_name, dev_name, test_name):
        """
        read the word from doc, and build sentence. every line contain a word and it's tag
        every sentence is split with a empty line. every sentence begain with an "IMGID:num"

        """
        # IMAGEID='IMGID'
        img_id = []
        sentences = []
        sentence = []
        sent_maxlen = 0
        word_maxlen = 0
        img_to_feature = []
        datasplit = []

        for fname in (train_name, dev_name, test_name):
            datasplit.append(len(img_id))
            with open(os.path.join(tweet_data_dir, fname), 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.rstrip()
                    if line == '':
                        sent_maxlen = max(sent_maxlen, len(sentence))
                        sentences.append(sentence)
                        sentence = []
                    else:
                        if IMAGEID in line:
                            num = line[6:]
                            img_id.append(num)
                        else:
                            sentence.append(line.split('\t'))
                            word_maxlen = max(word_maxlen, len(str(line.split()[0])))

        sentences.append(sentence)
        datasplit.append(len(img_id))
        num_sentence = len(sentences)

        print("datasplit", datasplit)
        print(sentences[len(sentences) - 2])
        print(sentences[0])

        ## get image feature.
        for image in img_id:
            if self.params.image_features_dir != '':
                feature = np.load(self.params.image_features_dir + image + '.npy').reshape(self.params.regions_in_image,
                                                                                           self.params.visual_feature_dimension)
            else:
                feature = np.random.random((self.params.regions_in_image, self.params.visual_feature_dimension))
            np_feature = np.array(feature)
            img_to_feature.append(np_feature)

        print('sent_maxlen', sent_maxlen)
        print('word_maxlen', word_maxlen)
        print('number sentence', len(sentences))
        print('number image', len(img_id))
        return [datasplit, sentences, img_to_feature, sent_maxlen, word_maxlen, num_sentence]

    def load_word_matrix(self, vocabulary, size=200):
        """
            This function is used to convert words into word vectors
        """
        b = 0
        word_matrix = np.zeros((len(vocabulary) + 1, size))
        if self.params.word2vec_model != '':
            model = word2vec.Word2Vec.load(self.params.word2vec_model)
        else:
            model = None
        for word, i in vocabulary.items():
            try:
                word_matrix[i] = model[word.lower().encode('utf8')]
            except:
                # if a word is not include in the vocabulary, it's word embedding will be set by random.
                word_matrix[i] = np.random.uniform(-0.25, 0.25, size)
                b += 1
        print('there are %d words not in model' % b)
        return word_matrix

    def vocab_bulid(self, sentences):
        """
        input:
            sentences list,
            the element of the list is (word, label) pair.
        output:
            some dictionaries.

        """
        words = []
        chars = []
        labels = []

        for sentence in sentences:
            for word_label in sentence:
                words.append(word_label[0])
                labels.append(word_label[1])
                for char in word_label[0]:
                    chars.append(char)
        word_counts = Counter(words)
        vocb_inv = [x[0] for x in word_counts.most_common()]
        vocb = {x: i + 1 for i, x in enumerate(vocb_inv)}
        vocb['PAD'] = 0
        id_to_vocb = {i: x for x, i in vocb.items()}

        char_counts = Counter(chars)
        vocb_inv_char = [x[0] for x in char_counts.most_common()]
        vocb_char = {x: i + 1 for i, x in enumerate(vocb_inv_char)}

        labels_counts = Counter(labels)
        print('labels_counts', len(labels_counts))
        print(labels_counts)
        labelVoc_inv, labelVoc = self.label_index(labels_counts)
        print('labelVoc', labelVoc)

        return [id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv]

    @staticmethod
    def label_index(labels_counts):
        """
           the input is the output of Counter. This function defines the (label, index) pair,
           and it cast our datasets label to the definition (label, index) pair.
        """

        num_labels = len(labels_counts)
        labelVoc_inv = [x[0] for x in labels_counts.most_common()]

        labelVoc = {'0': 0,
                    'B-PER': 1, 'I-PER': 2,
                    'B-LOC': 3, 'I-LOC': 4,
                    'B-ORG': 5, 'I-ORG': 6,
                    'B-OTHER': 7, 'I-OTHER': 8,
                    'O': 9}
        if len(labelVoc) < num_labels:
            for key, value in labels_counts.items():
                if not labelVoc.has_key(key):
                    labelVoc.setdefault(key, len(labelVoc))
        return labelVoc_inv, labelVoc

    @staticmethod
    def pad_sequence(sentences, img_to_feature, vocabulary, vocabulary_char, labelVoc, word_maxlen=30, sent_maxlen=35):
        """
            This function is used to pad the word into the same length, the word length is set to 30.
            Moreover, it also pad each sentence into the same length, the length is set to 35.

        """

        print(sentences[0])
        x = []
        y = []
        for sentence in sentences:
            w_id = []
            y_id = []
            for word_label in sentence:
                w_id.append(vocabulary[word_label[0]])
                y_id.append(labelVoc[word_label[1]])
            x.append(w_id)
            y.append(y_id)

        y = pad_sequences(y, maxlen=sent_maxlen, padding='post', truncating='post').astype(np.int32)
        x = pad_sequences(x, maxlen=sent_maxlen, padding='post', truncating='post').astype(np.int32)

        img_x = np.asarray(img_to_feature)

        x_c = []
        for sentence in sentences:
            s_pad = np.zeros([sent_maxlen, word_maxlen], dtype=np.int32)
            s_c_pad = []
            for word_label in sentence:
                w_c = []
                char_pad = np.zeros([word_maxlen], dtype=np.int32)
                for char in word_label[0]:
                    w_c.append(vocabulary_char[char])
                if len(w_c) <= word_maxlen:
                    char_pad[:len(w_c)] = w_c
                else:
                    char_pad = w_c[:word_maxlen]

                s_c_pad.append(char_pad)

            for i in range(len(s_c_pad)):
                # Post truncating
                if i < sent_maxlen:
                    s_pad[i, :len(s_c_pad[i])] = s_c_pad[i]
            x_c.append(s_pad)

        x_c = np.asarray(x_c)
        x = np.asarray(x)
        y = np.asarray(y)

        return [x, x_c, img_x, y]
