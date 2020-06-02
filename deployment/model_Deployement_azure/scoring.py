import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import transformers
import tokenizers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
#####
class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 2
    BERT_PATH = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'bert_files')
    TOKENIZER = tokenizers.BertWordPieceTokenizer(
        os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'bert_files', 'vocab.txt'), 
        lowercase=True
    )



def process_data(tweet, sentiment, tokenizer, max_len):
  
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids[1:-1]
    tweet_offsets = tok_tweet.offsets[1:-1]
    
    
    input_ids = [102] + input_ids_orig + [102]
    token_type_ids = [0] + [1] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 1 + tweet_offsets + [(0, 0)]
    
    target=sentiment

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'target': target,
        'orig_tweet': tweet,
        'offsets': tweet_offsets
    }


class TweetModel(transformers.BertPreTrainedModel):
    """
    Model class that combines a pretrained bert model with a linear later
    """
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        # Load the pretrained BERT model
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf)
        # Set 10% dropout to be applied to the BERT backbone's output
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        # Return the hidden states from the BERT backbone
        _, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        ) # bert_layers x bs x SL x (768)

        out = self.drop_out(out) # bs x SL x (768 * 2)
        # The "dropped out" hidden vectors are now fed into the linear layer to output two scores
        logits = self.l0(out) # bs x SL x 2
        
        

        return logits

def init():
    global model1
    device = torch.device("cpu")
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)    
    model_config.output_hidden_states = False

    model1 = TweetModel(conf=model_config)
    model1.to(device)
    model1.load_state_dict(torch.load(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'bert_files', 'modelv1'),map_location='cpu'))


input_sample = "text"

def run(input_sample):
    try:
        device = torch.device("cpu")
        data=process_data(input_sample,0,config.TOKENIZER,config.MAX_LEN)

        ids=torch.tensor([data["ids"]], dtype=torch.long)
        mask=torch.tensor([data["mask"]], dtype=torch.long)
        token_type_ids=torch.tensor([data["token_type_ids"]], dtype=torch.long)
        target=torch.tensor(data["target"], dtype=torch.long)
        orig_tweet=data["orig_tweet"]
        offsets=torch.tensor(data["offsets"], dtype=torch.long)

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        target = target.to(device, dtype=torch.long)

        model1.zero_grad()
        logit = model1(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )

        logit= torch.softmax(logit, dim=1).cpu().detach().numpy()

        print(logit)
        return logit.tolist()
    except Exception as e:
        error = str(e)
        return error


