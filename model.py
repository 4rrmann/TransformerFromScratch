import torch
import torch.nn as nn



class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int,):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        '''
        hey [ self.embedding ] lookup table, take this tensor of word IDs (x), go find the matching rows inside your matrix, and return those vector rows to me.

        hey extracted word vectors, your values currently have a variance close to 1. Let's scale you up by multiplying every element by [ sqrt{d_model} ] to increase your magnitude. That way, when the Positional Encoding tensor is added in the next step, your core semantic signals won't get structurally distorted or dominated by the positional values.

        [ x ] is a tensor of shape (B, T) where B is the batch size and T is the sequence length. Each element in x is an integer representing a word ID in the vocabulary.

        A [ Tensor ] from a transformation perspective is a geometric or mathematical object that remains invariant (unchanged) when you change your coordinate system or frame of reference, even though the numerical components used to describe it must transform according to specific mathematical rules
        '''
        return self.embedding(x) * (self.d_model ** 0.5)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, maxSeq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.maxSeq_len = maxSeq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (maxSeq_len, d_model) to hold the positional encodings
        pe = torch.zeros(maxSeq_len, d_model)

        # Create a tensor of shape (maxSeq_len, 1) containing the position indices
        position = torch.arange(0, maxSeq_len, dtype=torch.float).unsqueeze(1)

        #divide the embedding dimension into even and odd indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Fill the even indices with sine values
        pe[:, 0::2] = torch.sin(position * div_term)

        # Fill the odd indices with cosine values
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # pe.unsqueeze(0) adds a new dimension of size 1 at position 0 (the batch axis). It changes your positional encoding matrix from a 2D matrix into a 3D tensor, resulting in shape (1, maxSeq_len, d_model)

        # Register the positional encodings as a buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        [ x ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The positional encoding tensor [ self.pe ] has shape (1, maxSeq_len, d_model). When added to the input tensor x, it provides each position in the sequence with a unique representation based on its position.

        The addition of positional encodings allows the model to take into account the order of tokens in the sequence, which is crucial for understanding context and meaning in natural language processing tasks.
        '''
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    


class LayerNormalization(nn.Module):

    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        # in standardization formula, to ensure numerical stability during the normalization step, let's inject a smoothing parameter epsilon into the denominator. This acts as a regularization constant that prevents division-by-zero runtime exceptions and guards against floating-point overflow (NaN propagation) when the batch feature variance approaches zero.
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied (learnable scaling parameter)
        self.bias = nn.Parameter(torch.zeros(1))  # Added (learnable shifting parameter)

    def forward(self, x):
        '''
        [ x ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The mean and standard deviation are computed along the last dimension (d_model) for each position in the sequence, resulting in tensors of shape (B, T, 1). This ensures that each position is normalized independently based on its own statistics.

        The normalization is performed by subtracting the mean and dividing by the standard deviation (with epsilon added for numerical stability). This standardization process ensures that the input features have a mean of 0 and a variance of 1, which helps stabilize and accelerate training.

        The learnable parameters alpha and bias allow the model to scale and shift the normalized values, enabling it to adapt the normalization process to the specific needs of the task at hand. This flexibility can improve model performance by allowing it to learn optimal feature distributions.
        '''
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias
    


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and B1 (already includes bias true by default)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and B2 (already includes bias true by default)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        [ x ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The feedforward block consists of two linear transformations with a ReLU activation in between. The first linear layer projects the input from the embedding dimension (d_model) to a higher-dimensional space (d_ff), allowing the model to learn more complex representations. The dropout layer helps prevent overfitting by randomly setting a fraction of the input units to zero during training. The ReLU activation introduces non-linearity, enabling the model to capture intricate patterns in the data.

        The second linear layer projects the output back to the original embedding dimension (d_model), ensuring that the output can be combined with other components of the transformer architecture. The dropout layer helps prevent overfitting by randomly setting a fraction of the input units to zero during training.
        '''
        # (B, T, d_model) --> (B, T, d_ff) --> (B, T, d_model)
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
    


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads # h

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.head_dim = d_model // num_heads # dk = dm / h

        self.Wquery = nn.Linear(d_model, d_model) # Wq
        self.Wkey = nn.Linear(d_model, d_model) # Wk
        self.Wvalue = nn.Linear(d_model, d_model) # Wv

        self.out_proj = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    # Scaled Dot-Product Attention
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.size(-1) # head_dim

        attention_scores = (query @ key.transpose(-2, -1)) / (d_k ** 0.5) # (Batch, num_heads, maxSeq_len, head_dim) @ (Batch, num_heads, head_dim, maxSeq_len) --> (Batch, num_heads, maxSeq_len, maxSeq_len)

        if mask is not None:
            # This hides future words (like in a decoder) or padding tokens. Anywhere the mask is 0, it replaces the attention score with negative infinity (-inf).
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf')) # (Batch, num_heads, maxSeq_len, maxSeq_len)

        attention_scores = attention_scores.softmax(dim=-1) # (Batch, num_heads, maxSeq_len, d_k) --> (Batch, num_heads, maxSeq_len, maxSeq_len)

        if dropout is not None:
            # Helps model to not rely too heavily on just one word connection, preventing overfitting.
            attention_scores = dropout(attention_scores) # (Batch, num_heads, maxSeq_len, maxSeq_len)

        x = attention_scores @ value # (Batch, num_heads, maxSeq_len, maxSeq_len) @ (Batch, num_heads, maxSeq_len, head_dim) --> (Batch, num_heads, maxSeq_len, head_dim)

        return x, attention_scores


    def forward(self, q, k, v, mask):
        query = self.Wquery(q) # (Batch, maxSeq_len, d_model) --> (Batch, maxSeq_len, d_model)
        key = self.Wkey(k) # (Batch, maxSeq_len, d_model) --> (Batch, maxSeq_len, d_model)
        value = self.Wvalue(v) # (Batch, maxSeq_len, d_model) --> (Batch, maxSeq_len, d_model)

        query = query.view(query.size(0), query.size(1), self.num_heads, self.head_dim).transpose(1, 2) # (Batch, num_heads, maxSeq_len, head_dim)
        key = key.view(key.size(0), key.size(1), self.num_heads, self.head_dim).transpose(1, 2) # (Batch, num_heads, maxSeq_len, head_dim)
        value = value.view(value.size(0), value.size(1), self.num_heads, self.head_dim).transpose(1, 2) # (Batch, num_heads, maxSeq_len, head_dim)
        
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout) # (Batch, num_heads, maxSeq_len, head_dim)

        # (Batch, h, maxSeq_len, d_k) --> (Batch, maxSeq_len, h, d_k) --> (Batch, maxSeq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.num_heads * self.head_dim)

        # (Batch, maxSeq_len, d_model) --> (Batch, maxSeq_len, d_model)
        return self.out_proj(x)
    


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization()

    def forward(self, x, sublayer):
        '''
        [ x ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The residual connection adds the input tensor x to the output of the sublayer. This helps to mitigate the vanishing gradient problem and allows for better gradient flow during backpropagation. It also enables the model to learn identity mappings more easily, which can improve training stability and convergence.

        The layer normalization is applied after the addition of the residual connection. This normalizes the combined output, ensuring that it has a mean of 0 and a variance of 1 across the embedding dimension. This helps stabilize training and improves convergence by preventing internal covariate shift.
        '''
        return x + self.dropout(sublayer(self.layer_norm(x)))
    


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        '''
        [ x ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The encoder block consists of a multi-head self-attention mechanism followed by a feedforward neural network. The residual connections and layer normalization help stabilize training and improve convergence.

        The mask is used to prevent the model from attending to certain positions in the input sequence. This can be useful for tasks like language modeling or machine translation where certain tokens should not influence the attention mechanism.
        '''
        # MultiHeadAttention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        '''
        hey residual sub-layer interface, since our attention block signature expects separate query, key, value, and mask inputs, we are wrapping it in an anonymous lambda closure. This unifies its interface to accept a single tensor argument x, enabling seamless functional passing into the ResidualConnection instance while explicitly routing x to fulfill self-attention scaling requirements.
        '''
        # FeedForwardBlock
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()

    def forward(self, x, src_mask):
        '''
        [ x ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The encoder consists of a stack of encoder blocks. Each block processes the input sequentially, allowing the model to learn hierarchical representations of the input data.

        The mask is passed to each encoder block to ensure that the attention mechanism does not attend to certain positions in the input sequence. This can be useful for tasks like language modeling or machine translation where certain tokens should not influence the attention mechanism.
        '''
        for block in self.layers:
            x = block(x, src_mask)
        return self.layer_norm(x)
    


class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        '''
        [ x ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The decoder block consists of a multi-head self-attention mechanism, a multi-head cross-attention mechanism, and a feedforward neural network. The residual connections and layer normalization help stabilize training and improve convergence.

        The self-attention mask is used to prevent the model from attending to future positions in the target sequence. This ensures that the predictions for a given position can only depend on known outputs at earlier positions.

        The cross-attention mask is used to prevent the model from attending to certain positions in the source sequence. This can be useful for tasks like machine translation where certain tokens in the source sequence should not influence the attention mechanism.
        '''
        # MultiHeadAttention (self-attention)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        '''
        hey residual sub-layer interface, since our attention block signature expects separate query, key, value, and mask inputs, we are wrapping it in an anonymous lambda closure. This unifies its interface to accept a single tensor argument x, enabling seamless functional passing into the ResidualConnection instance while explicitly routing x to fulfill self-attention scaling requirements.
        '''
        # MultiHeadAttention (cross-attention)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        '''
        hey residual sub-layer interface, since our attention block signature expects separate query, key, value, and mask inputs, we are wrapping it in an anonymous lambda closure. This unifies its interface to accept a single tensor argument x, enabling seamless functional passing into the ResidualConnection instance while explicitly routing x to fulfill cross-attention scaling requirements.
        '''
        # FeedForwardBlock
        x = self.residual_connections[2](x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        '''
        [ x ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The decoder consists of a stack of decoder blocks. Each block processes the input sequentially, allowing the model to learn hierarchical representations of the target data while attending to the source data.

        The self-attention mask is passed to each decoder block to ensure that the attention mechanism does not attend to future positions in the target sequence. This ensures that the predictions for a given position can only depend on known outputs at earlier positions.

        The cross-attention mask is passed to each decoder block to ensure that the attention mechanism does not attend to certain positions in the source sequence. This can be useful for tasks like machine translation where certain tokens in the source sequence should not influence the attention mechanism.
        '''
        for block in self.layers:
            x = block(x, encoder_output, src_mask, tgt_mask)
        return self.layer_norm(x)
    


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        '''
        [ x ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The projection layer maps the output of the decoder to the vocabulary space. This allows the model to produce a probability distribution over the vocabulary for each position in the target sequence.

        The output of this layer can be used to compute the loss during training and to generate predictions during inference.
        '''
        return torch.log_softmax(self.linear_projection(x), dim=-1)
    


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embd: InputEmbedding, tgt_embd: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder_layer = encoder
        self.decoder_layer = decoder
        self.input_embedding = src_embd
        self.target_embedding = tgt_embd
        self.positional_encoding = src_pos
        self.target_positional_encoding = tgt_pos
        self.projection_layer = projection_layer


    def encoder(self, src, src_mask):
        '''
        [ src ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The encoder processes the source input sequence and produces a context representation that captures the relationships between tokens in the source sequence. This context representation is then used by the decoder to generate predictions for the target sequence.

        The mask is used to prevent the model from attending to certain positions in the source sequence. This can be useful for tasks like machine translation where certain tokens in the source sequence should not influence the attention mechanism.
        '''
        src_embd = self.input_embedding(src) # (B, T, d_model)
        src_pos = self.positional_encoding(src_embd) # (B, T, d_model)
        return self.encoder_layer(src_pos, src_mask) # (B, T, d_model)
    

    def decoder(self, tgt, encoder_output, src_mask, tgt_mask):
        '''
        [ tgt ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The decoder processes the target input sequence and generates predictions for the next tokens in the sequence. It attends to both the target input and the context representation produced by the encoder to make informed predictions.

        The self-attention mask is used to prevent the model from attending to future positions in the target sequence. This ensures that the predictions for a given position can only depend on known outputs at earlier positions.

        The cross-attention mask is used to prevent the model from attending to certain positions in the source sequence. This can be useful for tasks like machine translation where certain tokens in the source sequence should not influence the attention mechanism.
        '''
        tgt_embd = self.target_embedding(tgt) # (B, T, d_model)
        tgt_pos = self.target_positional_encoding(tgt_embd) # (B, T, d_model)
        return self.decoder_layer(tgt_pos, encoder_output, src_mask, tgt_mask) # (B, T, d_model)
    

    def projection(self, decoder_output):
        '''
        [ decoder_output ] is a tensor of shape (B, T, d_model) where B is the batch size, T is the sequence length, and d_model is the embedding dimension.

        The projection layer maps the output of the decoder to the vocabulary space. This allows the model to produce a probability distribution over the vocabulary for each position in the target sequence.

        The output of this layer can be used to compute the loss during training and to generate predictions during inference.
        '''
        return self.projection_layer(decoder_output) # (B, T, vocab_size)
    


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1):

    # Create the embedding layers for the source and target sequences
    src_embd = InputEmbedding(d_model, src_vocab_size)
    tgt_embd = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers for the source and target sequences
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(num_layers):
        self_attention_block = MultiHeadAttention(d_model, nhead, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(EncoderBlock(self_attention_block, feed_forward_block, dropout))

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(num_layers):
        self_attention_block = MultiHeadAttention(d_model, nhead, dropout)
        cross_attention_block = MultiHeadAttention(d_model, nhead, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout))

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = nn.Linear(d_model, tgt_vocab_size)

    # Create the transformer with explicit keyword arguments
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embd=src_embd,
        tgt_embd=tgt_embd,
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        projection_layer=projection_layer
    )


    # Initialize the parameters of the transformer using Xavier initialization
    '''When we first create a model, PyTorch fills its weight matrices with random numbers. If those numbers are too big, our model's signals will blow up (Exploding Gradients); if they are too small, the signals will vanish into zero (Vanishing Gradients). Xavier initialization calculates the mathematical sweet spot based on the number of input and output connections a layer has, ensuring that data flows cleanly through the network during the first steps of training.'''

    for p in transformer.parameters():
        '''It looks up every single weight matrix and bias tensor inside the entire model so it can iterate through them one by one.'''
        if p.dim() > 1:
            '''It checks if the parameter is a multi-dimensional matrix (like a weight layer) or a simple 1D vector (like a bias vector). Xavier initialization only works on 2D grids (matrices) or higher.'''
            nn.init.xavier_uniform_(p)
        '''Let's execute an in-place tensor modification using the Xavier uniform criterion, overriding the tensor's elements to stabilize variance and optimize gradient flow across deep networks.'''
        
    return transformer