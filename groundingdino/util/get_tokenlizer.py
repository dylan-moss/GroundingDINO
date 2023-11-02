from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
import os

def get_tokenlizer(text_encoder_type):
    tokenizer = AutoTokenizer.from_pretrained('tokenizer_config.json')
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    return BertModel.from_pretrained('config.json')

