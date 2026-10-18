import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_length):

        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_length = seq_length

        self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        src_target_pair = self.dataset[idx]
        # Get the source and target sentences
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize source and target sentences
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_length - len(enc_input_tokens) - 2  # Subtract 2 for SOS and EOS
        dec_num_padding_tokens = self.seq_length - len(dec_input_tokens) - 1  # Subtract 1 for SOS

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sequence length is too long for the given input. Please increase the seq_length or use shorter sentences.")
        
        # Add SOS and EOS tokens to the encoder input (source text)
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)

            ]
        )

        assert encoder_input.size(0) == self.seq_length
        assert decoder_input.size(0) == self.seq_length
        assert label.size(0) == self.seq_length
        '''An assert statement is a hard sanity check. If the condition is true, nothing happens. If the condition is false, the program instantly crashes with an error. Here, it guarantees that every token array is exactly padded or truncated to match the uniform seq_length expected by the Transformer'''

        return {
            "encoder_input": encoder_input, # (Seq_Len)
            "decoder_input": decoder_input, # (Seq_Len)

            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_Len)
            
            # When short sentences are padded with empty [PAD] tokens to make them reach seq_length, we don't want the model paying attention to those blank tokens. This line creates a matrix of 1s (for real words) and 0s (for padding tokens) and uses unsqueeze(0).unsqueeze(0) to add placeholder dimensions so it matches the 4D attention matrix shape (Batch, Heads, Seq_Len, Seq_Len)

            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size
            (0)), # (1, Seq_Len) & (1, Seq_Len, Seq_Len)
            
            "label": label, # (Seq_Len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0