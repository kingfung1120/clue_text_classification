from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from read_data import Dataset
import torch
import json


def train(model, train_data, val_data, learning_rate, epochs,
    train_acc, valid_acc, train_loss, valid_loss,
    batch_size=8, evaluate_per_batch=100, save_result=True):

    train_dataset, val_dataset = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using GPU:', use_cuda)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            total_data_train = 0

            print('Number of iterations: {}'.format(len(train_dataloader)))
            print('Evaluate valid set every {} iterations'.format(evaluate_per_batch))

            pbar = tqdm(enumerate(train_dataloader), desc='Training')

            for batch_num, (train_input, train_label) in pbar:

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
                total_data_train += len(train_label)

                pbar.set_description('Training acc: {}|'.format(round(total_acc_train/total_data_train, 3)))

                batch_loss.backward()
                optimizer.step()
                model.zero_grad()

                if batch_num%evaluate_per_batch == 0 and batch_num != 0:

                    total_acc_val = 0
                    total_loss_val = 0
                    total_data_val = 0

                    pbar.set_description('Evaluating valid set...')

                    for ind, (val_input, val_label) in enumerate(val_dataloader):

                        if ind > evaluate_per_batch:
                            break

                        pbar.set_description('Evaluating valid set {}%'.format(round(100*ind/evaluate_per_batch, 2)))

                        val_label = val_label.to(device)
                        mask = val_input['attention_mask'].to(device)
                        input_id = val_input['input_ids'].squeeze(1).to(device)

                        output = model(input_id, mask)

                        batch_loss = criterion(output, val_label)
                        acc = (output.argmax(dim=1) == val_label).sum().item()

                        total_loss_val += batch_loss.item()
                        total_acc_val += acc
                        total_data_val += len(val_label)

                    train_acc.append(total_acc_train/total_data_train)
                    valid_acc.append(total_acc_val/total_data_val)
                    train_loss.append(total_loss_train/total_data_train)
                    valid_loss.append(total_loss_val/total_data_val)

                    total_acc_train = 0
                    total_loss_train = 0
                    total_data_train = 0

                    if save_result:
                        result = {
                            'train_acc': train_acc,
                            'valid_acc': valid_acc,
                            'train_loss': train_loss,
                            'valid_loss': valid_loss,
                        }
                        with open('exp_dump/result.json', 'w') as fp:
                            json.dump(result, fp)



            # total_acc_val = 0
            # total_loss_val = 0
            # total_data_val = 0

            # with torch.no_grad():

            #     pbar = tqdm(val_dataloader, desc='Validation')

            #     for val_input, val_label in pbar:

            #         val_label = val_label.to(device)
            #         mask = val_input['attention_mask'].to(device)
            #         input_id = val_input['input_ids'].squeeze(1).to(device)

            #         output = model(input_id, mask)

            #         batch_loss = criterion(output, val_label)
            #         total_loss_val += batch_loss.item()

            #         acc = (output.argmax(dim=1) == val_label).sum().item()
            #         total_acc_val += acc
            #         total_data_val += len(val_label)

            #         pbar.set_description('Validation acc: {}|'.format(round(total_acc_val/total_data_val, 3)))

            # print(
            #     f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            #     | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            #     | Val Loss: {total_loss_val / len(val_data): .3f} \
            #     | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluation(model, test_df, batch_size=8):
    test_dataset = Dataset(test_df)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    pbar = tqdm(test_dataloader)

    test_acc = 0
    test_total = 0
    prediction = []
    gt = []

    for test_input, test_label in pbar:

        test_label = test_label.to(device)
        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)
        acc = (output.argmax(dim=1) == test_label).sum().item()
        test_acc += acc
        test_total += len(test_label)

        prediction.extend(output.argmax(dim=1))
        gt.extend(test_label)

    return acc, prediction, gt
