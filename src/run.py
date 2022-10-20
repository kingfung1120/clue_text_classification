import argparse

parser = argparse.ArgumentParser(description='Clue Text Classification')
parser.add_argument('--model', type=str, required=True, help='model: bert/bert_cnn')
args = parser.parse_args()


if __name__ == '__main__':
    from model.bert_model import BertClassifier
    from model.bert_cnn_model import BertCNNClassifier
    from read_data import Dataset, get_all_data_df, get_labels_df
    from train_eval import train, evaluation

    from sklearn.metrics import classification_report
    from torch.optim import Adam
    from torch import nn
    from tqdm import tqdm

    train_df, valid_df, test_df = get_all_data_df()
    labels_df = get_labels_df()

    if args.model == 'bert':
        model = BertClassifier(num_class=len(labels_df))
    elif args.model == 'bert_cnn':
        model = BertCNNClassifier(num_class=len(labels_df))

    lr = 5e-5
    epoch = 2
    batch_size = 8
    train_acc, valid_acc, train_loss, valid_loss = [], [], [], []

    train(model, train_df, valid_df, lr, epoch, train_acc, valid_acc, train_loss, valid_loss, batch_size)

    acc, prediction, gt = evaluation(model, test_df, batch_size)

    gt = [int(i) for i in gt]
    prediction = [int(i) for i in prediction]

    print(classification_report(gt, prediction, target_names=labels_df['label_des'].to_list()))
    print('Done')
