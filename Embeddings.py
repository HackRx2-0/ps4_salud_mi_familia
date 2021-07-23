import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class WordEmbedding:

    def __init__(self, text):

        self._text = text
        self._method = type(text)
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,)
        self.embedding = None
    
    def _bert_formatter(self, text):
        text = text.lower()
        return f'[CLS] {text} [SEP]'
    
    def get_embeddings(self):
        if self._method == list:
            self._text = [self._bert_formatter(t) for t in self._text]
            tokenized_texts = [self._tokenizer.tokenize(t) for t in self._text]
            indexed_texts = [self._tokenizer.convert_tokens_to_ids(t) for t in tokenized_texts]
            segments_ids = [[1] * len(t) for t in tokenized_texts]

            tokens_tensors = [torch.tensor([t]) for t in indexed_texts]
            segments_tensors = [torch.tensor([s]) for s in segments_ids]

            hidden_states = []
            for i in range(len(self._text)):
                with torch.no_grad():
                    outputs = self._model(tokens_tensors[i], segments_tensors[i])
                    hidden_states.append(outputs[2])
            
            embedding = [torch.stack(hidden_state, dim=0) for hidden_state in hidden_states]
            embedding = [torch.squeeze(token_embedding, dim=1) for token_embedding in embedding] 
            embedding = [token_embedding.permute(1,0,2) for token_embedding in embedding]

            self.embedding = []
            for e in embedding:
                vec_mean = []
                for token in e:
                    mean = torch.mean(token[-4:], dim=0)
                    vec_mean.append(mean)
                self.embedding.append(vec_mean)
        
        elif self._method == str:
            self._text = self._bert_formatter(self._text)
            tokenized_texts = self._tokenizer.tokenize(self._text)
            indexed_texts = self._tokenizer.convert_tokens_to_ids(tokenized_texts)
            segments_ids = [1] * len(tokenized_texts)

            tokens_tensors = torch.tensor([indexed_texts])
            segments_tensors = torch.tensor([segments_ids])

            with torch.no_grad():
                outputs = self._model(tokens_tensors, segments_tensors)
                hidden_state = outputs[2]
            
            embedding = torch.stack(hidden_state, dim=0)
            embedding = torch.squeeze(embedding, dim=1)
            embedding = embedding.permute(1,0,2)

            self.embedding = []
            for token in embedding:
                mean = torch.mean(token[-4:], dim=0)
                self.embedding.append(mean)



class SentenceEmbedding:

    def __init__(self, text):

        self._text = text
        self._method = type(text)
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,)
        self.embedding = None
    
    def _bert_formatter(self, text):
        text = text.lower()
        return f'[CLS] {text} [SEP]'
    
    def get_embeddings(self):
        if self._method == list:
            self._text = [self._bert_formatter(t) for t in self._text]
            tokenized_texts = [self._tokenizer.tokenize(t) for t in self._text]
            indexed_texts = [self._tokenizer.convert_tokens_to_ids(t) for t in tokenized_texts]
            segments_ids = [[1] * len(t) for t in tokenized_texts]

            tokens_tensors = [torch.tensor([t]) for t in indexed_texts]
            segments_tensors = [torch.tensor([s]) for s in segments_ids]

            hidden_states = []
            for i in range(len(self._text)):
                with torch.no_grad():
                    outputs = self._model(tokens_tensors[i], segments_tensors[i])
                    hidden_states.append(outputs[2])
            
            token_vecs = [h[-2][0] for h in hidden_states]
            self.embedding = [torch.mean(t, dim=0) for t in token_vecs]
        
        elif self._method == str:
            self._text = self._bert_formatter(self._text)
            tokenized_texts = self._tokenizer.tokenize(self._text)
            indexed_texts = self._tokenizer.convert_tokens_to_ids(tokenized_texts)
            segments_ids = [1] * len(tokenized_texts)

            tokens_tensors = torch.tensor([indexed_texts])
            segments_tensors = torch.tensor([segments_ids])

            with torch.no_grad():
                outputs = self._model(tokens_tensors, segments_tensors)
                hidden_state = outputs[2]
            
            token_vecs = hidden_state[-2][0]
            self.embedding = torch.mean(token_vecs, dim=0)