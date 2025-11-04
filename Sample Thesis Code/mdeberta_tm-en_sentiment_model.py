from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import numpy as np
import pandas as pd
# from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from collections import defaultdict
import seaborn as sns
from sklearn.utils import class_weight
import random
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for seed_val in [55, 87, 107, 123]:
    sv = seed_val
    seed = sv
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")

    PRE_TRAINED_MODEL_NAME = 'microsoft/mdeberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, use_fast=False)

    def process_DS():
        train_df = pd.read_csv(base_path+'Tamil_CM_train.csv') 
        dev_df = pd.read_csv(base_path+'Tamil_CM_dev.csv') #
        test_df = pd.read_csv(base_path+'Tamil_CM_test.csv') #
        return (train_df, dev_df, test_df)


    class TamilSentimentDataset(Dataset):

        def __init__(self, sentences, labels, tokenizer, max_len):
            self.sentences = sentences
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
        
        def __len__(self):
            return len(self.sentences)
        
        def __getitem__(self, item):
            sentence = str(self.sentences[item])
            label = self.labels[item]

            encoding = self.tokenizer.encode_plus(
              sentence,
              add_special_tokens=True,
              truncation=True,
              max_length=self.max_len,
              return_token_type_ids=False,
              padding='max_length',
              return_attention_mask=True,
              return_tensors='pt',
            )

            return {
              'sentence_text': sentence,
              'input_ids': encoding['input_ids'].flatten(),
              'attention_mask': encoding['attention_mask'].flatten(),
              'labels': torch.tensor(label, dtype=torch.long)
            }


    def create_data_loader(df, tokenizer, max_len, batch_size, is_shuffled):
        ds = TamilSentimentDataset(
          sentences=df.Sentence.to_numpy(),
          labels=df.Label.to_numpy(),
          tokenizer=tokenizer,
          max_len=max_len
        )

        return DataLoader(
          ds,
          shuffle=is_shuffled,
          batch_size=batch_size
          # num_workers=4
        )


    def make_data_loader():
        train_df, dev_df, test_df = process_DS()
        BATCH_SIZE = 16
        MAX_LEN = 128

        train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE, True)
        val_data_loader = create_data_loader(dev_df, tokenizer, MAX_LEN, BATCH_SIZE, False)
        test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE, False)

        return (train_data_loader, val_data_loader, test_data_loader)


    def get_features(model_op, input_ids):
        all_hidden_states = torch.stack(model_op[1])
        concatenate_pooling = torch.cat((all_hidden_states[-4], all_hidden_states[-3], all_hidden_states[-2], all_hidden_states[-1]),-1)
        
        iil = input_ids.tolist()
        input_ids_list = []
        for seq_ids in iil:
          input_ids_list.append([idn for idn in seq_ids if idn != 0])

        special_ids = [1, 2]
        seq_itr = 0
        all_seqs = []
        for ip_seq_emb in concatenate_pooling:
            subtokens = []
            all_tokens = []
            mean_subtokens = torch.Tensor([0])
            mean_all_tokens = torch.Tensor([0])
            prev_tok = ''
            prev_tok_emb = 0
            prev_tok_added_flag = False
            for i in range(len(input_ids_list[seq_itr])):
                if input_ids_list[seq_itr][i] not in special_ids:
                    tok = tokenizer.convert_ids_to_tokens(input_ids_list[seq_itr][i])
                    if not (tok[0].encode() == b'\xe2\x96\x81'): # is a subtoken
                        if prev_tok_added_flag: 
                            subtokens.append(ip_seq_emb[i, :])
                        else:
                            subtokens.append(prev_tok_emb) # Appending first subtoken in subtokens of a token
                            subtokens.append(ip_seq_emb[i, :])
                            prev_tok_added_flag = True
                    else:
                        if len(subtokens) != 0:
                            sum_tensors = 0
                            for st in subtokens:
                                sum_tensors += st
                            mean_subtokens = sum_tensors / len(subtokens) # mean of subtokens
                            subtokens.clear()
                            prev_tok_added_flag = False
                    # ------------------------------------------------------------------
                    if mean_subtokens.sum() != 0:
                        all_tokens.pop()
                        all_tokens.append(mean_subtokens)
                        mean_subtokens = torch.Tensor([0])

                    elif (mean_subtokens.sum() == 0) and (len(subtokens) > 0) and (i == (len(input_ids_list[seq_itr])-2)): # if subtok is at end
                        sum_tensors = 0
                        for st in subtokens:
                            sum_tensors += st
                        mean_subtokens = sum_tensors / len(subtokens) # mean of subtokens
                        subtokens.clear()
                        prev_tok_added_flag = False
                        all_tokens.pop()
                        all_tokens.append(mean_subtokens)
                        
                    # ---------------------------------------------------------------------
                    if (tok[0].encode() == b'\xe2\x96\x81'):
                        all_tokens.append(ip_seq_emb[i, :])

                    prev_tok = tok
                    prev_tok_emb = ip_seq_emb[i]
            sum_tensors = 0
            for st in all_tokens:
                sum_tensors += st
            mean_all_tokens = sum_tensors / len(all_tokens)
            all_seqs.append(mean_all_tokens)
            seq_itr += 1
            
        T = torch.stack([seq for seq in all_seqs])
        return T




    class SentimentClassifier(nn.Module):

        def __init__(self, n_classes):
            super(SentimentClassifier, self).__init__()
            self.mdeberta = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
            # self.drop = nn.Dropout(p=0.3)
            # self.non_linearity = nn.ReLU()
          
            self.out = nn.Linear(768*4, n_classes)
            # torch.nn.init.kaiming_uniform_(self.out.weight, mode="fan_out", nonlinearity="relu")
            # torch.nn.init.xavier_uniform_(self.out.weight)
        
        def forward(self, input_ids, attention_mask):
            with torch.no_grad():  
                model_op = self.mdeberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            rep = get_features(model_op, input_ids)
            # output = self.drop(rep)
            return self.out(rep)


    def compute_metrics(gt, preds):
        f1 = f1_score(gt, preds, average='macro')
        precision = precision_score(gt, preds, average='macro')
        recall = recall_score(gt, preds, average='macro')
        acc = accuracy_score(gt, preds)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


    def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
        model = model.train()

        losses = []
        correct_predictions = 0
        
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            optimizer.step()
            optimizer.zero_grad()

        return correct_predictions.double() / n_examples, np.mean(losses)


    ground_truths = []
    predictions = []
    def eval_model(model, data_loader, loss_fn, device, n_examples):
        model = model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                loss = loss_fn(outputs, labels)

                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())

                numpied_gt = labels.cpu().numpy()
                numpied_preds = preds.cpu().numpy()
                ground_truths.extend(numpied_gt)
                predictions.extend(numpied_preds)

        return correct_predictions.double() / n_examples, np.mean(losses)


    train_df, dev_df, _  = process_DS()
    # class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(list(train_df['Label'])), y=np.array(list(train_df['Label'])))
    # class_weights = torch.tensor(class_weights,dtype=torch.float)
    # 0:0.67, 1:0.03, 2:0.05, 3:0.13, 4:0.11
    class_weights = torch.tensor([0.67, 0.03, 0.05, 0.13, 0.11])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentClassifier(5)
    model = model.to(device)

    EPOCHS = 70

    optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)



    # def train_features():
    history = defaultdict(list)
    best_f1 = 0
    train_data_loader, val_data_loader, test_data_loader = make_data_loader()

    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, len(train_df))

        print(f'Train loss {train_loss} accuracy {train_acc}')
        
        ground_truths = []
        predictions = []

        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(dev_df))
        val_metrics = compute_metrics(ground_truths, predictions)

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()
        print('Other val metrics: ', val_metrics)
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])

        if val_metrics['f1'] > best_f1:
            torch.save(model.state_dict(), 'best_model_state_mdeberta_tam_'+str(sv)+'.bin')
            best_f1 = val_metrics['f1']


    def make_loss_plot():
        train_loss = [ele.item() for ele in history['train_loss']]
        val_loss = [ele.item() for ele in history['val_loss']] 
        plt.figure(figsize=(8, 8)) 
        plt.plot(train_loss, label='train loss')
        plt.plot(val_loss, label='validation loss')
        plt.title('Training history')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, 5])
        plt.show()
        plt.savefig('loss_curve_mdeberta_tam_'+str(sv)+'.png')
        plt.clf()

    def make_f1_plot():
        plt.figure(figsize=(8, 8))
        plt.plot(history['val_f1'], label='validation F1')
        plt.title('F1 score history')
        plt.ylabel('F1')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, 1])
        plt.show()
        plt.savefig('f1_curve_mdeberta_tam'+str(sv)+'.png')
        plt.clf()

    make_loss_plot()
    make_f1_plot()

    def get_test_scores(model, data_loader):
        model = model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sentence_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in data_loader:

                texts = d["sentence_text"]
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                probs = torch.nn.functional.softmax(outputs, dim=1)

                texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(labels)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return texts, predictions, prediction_probs, real_values


    def classification_report_csv(report):
        report_data = []
        lines = report.split('\n')
        for line in lines[2:]:
            if len(line) != 0:
                row = {}
                row_data = line.split()
                # print(row_data)
                if len(row_data) == 3:
                    row['class'] = row_data[0]
                    row['precision'] = '-'
                    row['recall'] = '-'
                    row['f1_score'] = float(row_data[1])
                    row['support'] = int(row_data[2])
                elif len(row_data) == 6:
                    row['class'] = row_data[0]+" "+row_data[1]
                    row['precision'] = float(row_data[2])
                    row['recall'] = float(row_data[3])
                    row['f1_score'] = float(row_data[4])
                    row['support'] = int(row_data[5])
                else:
                    row['class'] = row_data[0]
                    row['precision'] = float(row_data[1])
                    row['recall'] = float(row_data[2])
                    row['f1_score'] = float(row_data[3])
                    row['support'] = int(row_data[4])
                report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv('classification_report_TamEn_mdeberta_'+str(sv)+'.csv', index = False)



    model = SentimentClassifier(5)
    model.load_state_dict(torch.load('best_model_state_mdeberta_tam_'+str(sv)+'.bin'))
    model = model.to(device)

    texts, y_pred, y_pred_probs, y_test = get_test_scores(model, test_data_loader)
    map_dt = {0:'Other language', 1:'Positive', 2:'Negative', 3:'Neutral', 4:'Mixed feelings'}
    rep = classification_report(y_test, y_pred, digits=4, target_names=list(map_dt.values()))
    classification_report_csv(rep)



    def show_confusion_matrix(confusion_matrix):
        plt.figure(figsize=(8, 8))
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True sentiment')
        plt.xlabel('Predicted sentiment')
        plt.show()
        plt.savefig('CM_TamEn_mdeberta_'+str(sv)+'.png')
        plt.clf()

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=list(map_dt.values()), columns=list(map_dt.values()))
    show_confusion_matrix(df_cm)



    def gen_pred_file():

        test_df = pd.read_csv(base_path+'Tamil_CM_test.csv')
        label_map = {'Other language':0, 'Positive':1, 'Negative':2, 'Neutral':3, 'Mixed feelings':4}
        test_labels = list(test_df['Label'])
        test_sents = list(test_df['Sentence'])

        pred_file = open('test_preds_mdeberta_'+str(sv)+'.txt', 'w', encoding='utf-8')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(len(test_sents)):
            encoded_review = tokenizer.encode_plus(test_sents[i], max_length=128, add_special_tokens=True, return_token_type_ids=False,
                                                  padding='max_length', return_attention_mask=True, return_tensors='pt')
            input_ids = encoded_review['input_ids'].to(device)
            attention_mask = encoded_review['attention_mask'].to(device)
            try:
                output = model(input_ids, attention_mask)
                _, prediction = torch.max(output, dim=1)
            except:
                continue

            pred_file.write('\n' + test_sents[i] + '\n')
            if prediction.item() != test_labels[i]:
                pred_file.write('- MISMATCH' + '\n' + 'Gold: ' + map_dt[test_labels[i]] + '\n' + 'Predicted: ' + map_dt[prediction.item()] + '\n' + 200*'-')
            else:
                pred_file.write('Gold: ' + map_dt[test_labels[i]] + '\n' + 'Predicted: ' + map_dt[prediction.item()] + '\n'+ 200*'-')


    gen_pred_file()
