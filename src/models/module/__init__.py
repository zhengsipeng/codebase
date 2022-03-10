from .layers.transformer import TransformerEncoder, TransformerEncoderLayer


def build_cross_encoder(config):
    d_model = config.Model.hidden_dim
    nhead = config.Model.encoder_num_heads
    dim_feedforward = config.Model.dim_feedforward
    dropout = config.Model.dropout
    activation = config.Model.activation
    normalize_before = config.Model.pre_norm
    num_encoder_layers = config.Model.encoder_depth
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, 
                                            activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    cross_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return cross_encoder