from torch import nn
import torch
from transformers import AutoModelForMaskedLM

class BertCNNClassifier(nn.Module):

    def __init__(self, num_class, num_filters=64, filter_sizes=(2, 3, 4), dropout=0.1):

        super(BertCNNClassifier, self).__init__()

        bert = AutoModelForMaskedLM.from_pretrained('bert-base-chinese')
        self.bert = list(bert.children())[0] # To remove the decoder head
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, 768)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_filters * len(filter_sizes), num_class)

    def conv_and_pool(self, x, conv):
        x = nn.functional.relu(conv(x)).squeeze(3)
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_id, mask):

        output, _ = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        output = output.unsqueeze(1)
        output = torch.cat([self.conv_and_pool(output, conv) for conv in self.convs], 1)
        output = self.dropout(output)
        output = self.linear(output)

        return output
