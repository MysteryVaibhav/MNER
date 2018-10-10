import os
import argparse
from model import *
from data_loader import DataLoader
from evaluator import Evaluator
from trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for MNER')
    parser.add_argument("--image_features_dir", dest="image_features_dir", type=str, default='')
    parser.add_argument("--caption_file", dest="caption_file", type=str, default='')
    parser.add_argument("--split_file", dest="split_file", type=str, default='G:\\My Drive\\CMU\\Research\\Code\\sem3\\NERmultimodal\\data\\')
    parser.add_argument("--word2vec_model", dest="word2vec_model", type=str, default='')

    parser.add_argument("--hidden_dimension", dest="hidden_dimension", type=int, default=16)
    parser.add_argument("--hidden_dimension_char", dest="hidden_dimension_char", type=int, default=8)
    parser.add_argument("--embedding_dimension", dest="embedding_dimension", type=int, default=10)
    parser.add_argument("--embedding_dimension_char", dest="embedding_dimension_char", type=int, default=5)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=39034)
    parser.add_argument("--char_vocab_size", dest="char_vocab_size", type=int, default=127)
    parser.add_argument("--use_char_embedding", dest="use_char_embedding", type=int, default=1)

    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--lr", dest="lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=30)
    parser.add_argument("--lambda_1", dest="lambda_1", type=int, default=9)
    parser.add_argument("--n_layers", dest="n_layers", type=int, default=1)
    parser.add_argument("--clip_value", dest="clip_value", type=float, default=5)
    parser.add_argument("--wdecay", dest="wdecay", type=float, default=0.00001)
    parser.add_argument("--step_size", dest="step_size", type=int, default=15)
    parser.add_argument("--gamma", dest="gamma", type=int, default=10)
    parser.add_argument("--validate_every", dest="validate_every", type=int, default=1)

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    parser.add_argument("--mode", dest="mode", type=int, default=0)
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default="model_weights.t7")
    parser.add_argument("--sent_maxlen", dest="sent_maxlen", type=int, default=35)
    parser.add_argument("--word_maxlen", dest="word_maxlen", type=int, default=30)
    parser.add_argument("--visual_feature_dimension", dest="visual_feature_dimension", type=int,
                        default=20)
    parser.add_argument("--regions_in_image", dest="regions_in_image", type=int, default=5)
    return parser.parse_args()


def main():
    params = parse_arguments()
    print("Constructing data loaders...")
    dl = DataLoader(params)
    evaluator = Evaluator(params, dl)
    print("Constructing data loaders...[OK]")

    if params.mode == 0:
        print("Training...")
        t = Trainer(params, dl, evaluator)
        t.train()
        print("Training...[OK]")
    elif params.mode == 1:
        print("Loading model...")
        model = MNER(params)
        model_file_path = os.path.join(params.model_dir, params.model_file_name)
        model.load_state_dict(torch.load(model_file_path))
        if torch.cuda.is_available():
            model = model.cuda()
        print("Loading model...[OK]")

        print("Evaluating model on test set...")
        acc, f1, prec, rec = evaluator.get_accuracy(model, 'test')
        print("Accuracy : {}".format(acc))
        print("F1 : {}".format(f1))
        print("Precision : {}".format(prec))
        print("Recall : {}".format(rec))
        print("Evaluating model on test set...[OK]")


if __name__ == '__main__':
    main()
