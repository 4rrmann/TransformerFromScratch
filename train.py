import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader

from dataset import BilingualDataset, casual_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import warnings
'''
# Load the IIT Bombay English-Hindi dataset
dataset = load_dataset("cfilt/iitb-english-hindi")

# View a sample from the training split
print(dataset["train"][1000])
# Output will look just like opus_books:
# {'translation': {'en': 'Give your car new life.', 'hi': 'अपनी कार को नया जीवन दें।'}}
'''


def greedy_decode(model, source, source_mask, source_tokenizer, target_tokenizer, max_len, device):
    # source/encoder_input   &   source_mask/encoder_mask
    sos_idx = target_tokenizer.token_to_id('[SOS]')
    eos_idx = target_tokenizer.token_to_id('[EOS]') 

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target of the decoder
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])
        # Select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, source_tokenizer, target_tokenizer, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    # Size of the control window (just use a default value)
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count +=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, source_tokenizer, target_tokenizer, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = target_tokenizer.decode(model_out.detach().cpu().numpy())

            # Print it to the console
            print_msg('-'*console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {source_text}")
            print_msg(f"PREDICTED: {source_text}")

            if count == num_examples:
                break



def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]
        '''By using yield, the function becomes a stream. Your model can read a sentence, train on it, wipe it from memory, and ask for the next one, allowing you to easily train on massive datasets that are much larger than your computer's RAM.'''

def get_or_build_tokenizer(config, dataset, lang):
    '''This function handles your text processing engine. It checks your hard drive to see if a Tokenizer has already been created for your language. If it finds one, it loads it instantly; if not, it scans your dataset, builds a brand new vocabulary dictionary, assigns special slots for padding and markers, trains itself, and saves it to a file so you never have to wait for it to build again.'''
    tokenizer_path = Path(config['tokenizer_path'])
    if not Path.exists(tokenizer_path):
        # Build a new tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # specifying a fallback token primitive [UNK] to catch out-of-vocabulary (OOV) tokens
        tokenizer.pre_tokenizer = Whitespace() # Splitting Words
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], minimum_frequency=2)
        '''It configures the training engine. It reserves special structural tokens:
        [UNK] (unknown),
        [PAD] (padding to make sentences equal length),
        [SOS] (Start of Sentence), and
        [EOS] (End of Sentence).
        It also ignores rare words that show up fewer than 2 times across your whole dataset (minimum_frequency=2)'''
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# def get_dataset(config):
#     ds_raw = load_dataset('opus_books', f"{config['source_lang']}-{config['target_lang']}", split='train')

#     # Build tokenizers for source and target languages
#     source_tokenizer = get_or_build_tokenizer(config, ds_raw, config['source_lang'])
#     target_tokenizer = get_or_build_tokenizer(config, ds_raw, config['target_lang'])

#     # keep 90% for training, 10% for validation
#     train_ds_size = int(0.9 * len(ds_raw))
#     val_ds_size = len(ds_raw) - train_ds_size
#     train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])



# def get_dataset(config):
#     # 1. Load the pre-defined splits directly from Hugging Face
#     train_ds_raw = load_dataset('cfilt/iitb-english-hindi', split='train')
#     val_ds_raw = load_dataset('cfilt/iitb-english-hindi', split='validation')

#     # 2. Build tokenizers using the training split data
#     # (Assuming get_or_build_tokenizer extracts the text correctly from the 'translation' key)
#     source_tokenizer = get_or_build_tokenizer(config, train_ds_raw, config['source_lang'])
#     target_tokenizer = get_or_build_tokenizer(config, train_ds_raw, config['target_lang'])


#     train_ds = BilingualDataset(train_ds_raw, source_tokenizer, target_tokenizer, config['source_lang'], config['target_lang'], config['seq_len'])
#     val_ds = BilingualDataset(val_ds_raw, source_tokenizer, target_tokenizer, config['source_lang'], config['target_lang'], config['seq_len'])

#     max_len_src = 0
#     max_len_tgt = 0

#     for item in train_ds_raw:
#         src_ids = source_tokenizer.encode(item['translation'][config['source_lang']]).ids
#         tgt_ids = source_tokenizer.encode(item['translation'][config['source_lang']]).ids

#         max_len_src = max(max_len_src, len(src_ids))
#         max_len_tgt = max(max_len_tgt, len(tgt_ids))

#     print(f"Max length of source sentence: {max_len_src}")
#     print(f"Max length of target sentence: {max_len_tgt}")

#     train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
#     val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

#     return train_dataloader, val_dataloader, source_tokenizer, target_tokenizer





def get_dataset(config):
    # 1. Load the pre-defined splits directly from Hugging Face
    train_ds_raw = load_dataset('cfilt/iitb-english-hindi', split='train')
    val_ds_raw = load_dataset('cfilt/iitb-english-hindi', split='validation')

    # 2. Build tokenizers using the training split data
    source_tokenizer = get_or_build_tokenizer(config, train_ds_raw, config['source_lang'])
    target_tokenizer = get_or_build_tokenizer(config, train_ds_raw, config['target_lang'])

    # 3. Create PyTorch datasets
    train_ds = BilingualDataset(train_ds_raw, source_tokenizer, target_tokenizer, config['source_lang'], config['target_lang'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, source_tokenizer, target_tokenizer, config['source_lang'], config['target_lang'], config['seq_len'])

    # 4. Calculate Max Lengths
    max_len_src = 0
    max_len_tgt = 0

    # Windows/CPU Optimization: Calculate lengths using a large sample (e.g. 75,000 sentences)
    # This prevents the script from hanging for 20+ minutes just to print the length.
    sample_ds = train_ds_raw.select(range(min(75000, len(train_ds_raw))))

    for item in sample_ds:
        src_ids = source_tokenizer.encode(item['translation'][config['source_lang']]).ids
        tgt_ids = target_tokenizer.encode(item['translation'][config['target_lang']]).ids # Fixed target variables

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence (sampled): {max_len_src}")
    print(f"Max length of target sentence (sampled): {max_len_tgt}")

    # 5. Create Dataloaders 
    # NOTE: On Windows, setting num_workers > 0 inside DataLoader can sometimes cause Multiprocessing/Pickling errors. Keeping it at default (0) is safest.
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, source_tokenizer, target_tokenizer



def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, source_tokenizer, target_tokenizer = get_dataset(config)
    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)

#Tensorboard

    # Initialize the TensorBoard SummaryWriter 
    # It creates a directory named 'runs' to save logs
    writer = SummaryWriter(log_dir=config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_stop = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] +1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_stop = state['global_stop']


    loss_func = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    for epoch in range(initial_epoch, config['num_epochs']):
        
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            model.train()

            encoder_input = batch['encoder_input'].to(device) # (B, Seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, Seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, Seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, Seq_len, Seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encoder(encoder_input, encoder_mask) # (B, Seq_len, d_model)
            decoder_output = model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask) # (B, Seq_len, d_model)

            output_projection = model.projection(decoder_output) # (B, Seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (B, Seq_len)

            # (B, Seq_len, tgt_vocab_size) --> # (B * Seq_len, tgt_vocab_size)
            loss = loss_func(output_projection.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_stop)
            writer.flush()

            # Backpropagate the loss
            loss.backward()
            '''PyTorch traces backward through your Transformer model from the loss value all the way to the initial input. It calculates exactly how much each weight layer contributed to the error (gradients).'''

            # Update the weights
            optimizer.step()
            '''The Adam optimizer uses those freshly calculated gradients to adjust the internal weights of your model, bringing it one step closer to making accurate English-to-Hindi translations.'''
            optimizer.zero_grad()
            '''By default, PyTorch accumulates (adds up) gradients on every backward pass. If you forget this line, the gradients from the new batch will blend with the old batch, which will break your mathematical updates and cause the model to fail to converge.'''

            # Increments the step tracker for TensorBoard logging
            global_stop +=1

        run_validation(model, val_dataloader, source_tokenizer, target_tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_stop, writer)
        
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_stop": global_stop
        }, model_filename)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)