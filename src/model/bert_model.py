from torch import nn
from transformers import AutoModelForMaskedLM

class BertClassifier(nn.Module):

    def __init__(self, num_class, dropout=0.1):

        super(BertClassifier, self).__init__()

        bert = AutoModelForMaskedLM.from_pretrained('bert-base-chinese')
        self.bert = list(bert.children())[0] # To remove the decoder head
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear1 = nn.Linear(768, 768)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(768, num_class)

    def forward(self, input_id, mask):

        output, _ = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        pooled_output = output[:, 0, :] # get the embedding for the <cls> token
        pooled_output = self.linear1(pooled_output)
        pooled_output = self.tanh(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear2(pooled_output)

        return pooled_output
