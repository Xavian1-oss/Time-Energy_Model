import torch
from argparse import Namespace

from energy.EnergyModelsV2 import ArbitraryMLPDecoder, MultisampleMLPDecoder
from energy.EnergyModelsV3 import AutoformerNeoEBM
from models import TimesNet


class TimesNetNeoEBM_concat(AutoformerNeoEBM):
    def __init__(
        self,
        setting: Namespace,
        times_net_model: TimesNet,
        x_dim: int,  
        y_dim: int,
        x_encoder_lstm_hidden_size: int = 16,
        x_encoder_code_dim: int = 8,
        x_encoder_lstm_num_layers: int = 1,
        x_decoder_num_layers: int = 1,
        x_decoder_code_dim: int = 8,
        y_decoder_num_layers: int = 1,
        y_decoder_code_dim: int = 8,
        xy_decoder_hidden_dim: int = 128,
        xy_decoder_num_layers: int = 3,
        use_normalizing_constant: bool = False,
        batch_samples: bool = False,
    ):
        super(TimesNetNeoEBM_concat, self).__init__(
            setting,
            times_net_model,
            x_dim,
            y_dim,
            x_encoder_lstm_hidden_size,
            x_encoder_code_dim,
            x_encoder_lstm_num_layers,
            x_decoder_num_layers,
            x_decoder_code_dim,
            y_decoder_num_layers,
            y_decoder_code_dim,
            xy_decoder_hidden_dim,
            xy_decoder_num_layers,
            use_normalizing_constant,
            batch_samples,
        )
        self.times_net_model = times_net_model

    def get_assumed_enc_out_shape(self, args):
        assumed_enc_out_shape = [
            args.batch_size,
            args.seq_len + args.pred_len,
            args.d_model,
        ]
        return assumed_enc_out_shape

    def get_assumed_dec_out_shape(self, args):
        assumed_dec_out_shape = [
            args.batch_size,
            args.seq_len + args.pred_len,
            args.c_out,
        ]
        return assumed_dec_out_shape

    
    
    def setup_y_encoder_and_xy_decoder_(
        self, seq_len, label_len, pred_len, d_model, c_out, dec_out_2_dim
    ):
        

        self.y_encoder = MultisampleMLPDecoder(
            
            input_dim=(pred_len) * c_out,
            output_dim=(dec_out_2_dim) * d_model,
            num_layers=self.y_decoder_num_layers,
            hidden_size=self.y_decoder_code_dim,
        )
        self.xy_decoder = ArbitraryMLPDecoder(
            input_dim=(
                dec_out_2_dim * 2 * d_model
            ),  
            output_dim=1,  
            num_layers=self.decoder_num_layers,
            hidden_size=self.decoder_hidden_dim,
        )
        self.orig_model_seq_len = seq_len
        self.orig_model_label_len = label_len
        self.orig_model_pred_len = pred_len
        self.orig_model_d_model = d_model

        print(f"Setup for of Y encoder and XY decoder done!")

    def _forward_y_enc(self, batch_y):
        actual_y = batch_y[:, -self.orig_model_pred_len :, :]
        
        
        
        
        encoded_y = self.y_encoder(actual_y)
        reshaped_encoded_y = encoded_y.reshape(
            actual_y.shape[0],
            
            (self.orig_model_seq_len + self.orig_model_pred_len),
            self.orig_model_d_model,
        )
        return reshaped_encoded_y

    def _forward_y_enc(self, batch_y):
        """Encode target sequence y for TimesNet-based NeoEBM.

        MultisampleMLPDecoder flattens the (batch, sample/channel) axes,
        and for TimesNet the configured output_dim can include an extra
        multiplicative factor (e.g., number of input channels). Instead
        of assuming a fixed output dimension, we infer this factor and
        reshape back to a tensor compatible with the backbone encoder
        output: [B, seq_len + pred_len, d_model].
        """

        # Use only the prediction window; TimesNet batches batch_y
        # typically as [B, pred_len, C_out].
        actual_y = batch_y[:, -self.orig_model_pred_len :, :]

        B = actual_y.shape[0]
        encoded_y = self.y_encoder(actual_y)

        # Collapse any sample/channel dimension introduced by
        # MultisampleMLPDecoder into a single flat dimension per batch
        # element.
        encoded_y = encoded_y.contiguous().view(B, -1)

        total_seq = self.orig_model_seq_len + self.orig_model_pred_len
        D = self.orig_model_d_model
        base = B * total_seq * D
        total_numel = int(encoded_y.numel())

        if total_numel % base != 0:
            raise RuntimeError(
                f"TimesNetNeoEBM_concat: encoded_y.numel()={total_numel} "
                f"is not divisible by B * (seq_len+pred_len) * d_model = {base}. "
                f"Shapes: B={B}, total_seq={total_seq}, D={D}, "
                f"encoded_y.shape={tuple(encoded_y.shape)}"
            )

        # Inferred multiplicity factor (e.g., enc_in or other flattening).
        K = total_numel // base
        encoded_y = encoded_y.view(B, K, total_seq, D)

        # Aggregate over the multiplicity dimension to recover a
        # [B, total_seq, D] representation compatible with enc_out.
        reshaped_encoded_y = encoded_y.mean(dim=1)
        return reshaped_encoded_y

    def _get_decoded(
        self,
        batch_y: torch.Tensor,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_out,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        encoded_x = enc_out
        encoded_y = self._forward_y_enc(
            batch_y
        )  
        xy_encoded = torch.cat([encoded_x, encoded_y], dim=2)
        xy_encoded_reshaped = xy_encoded.view(xy_encoded.shape[0], -1)
        score = self.xy_decoder(xy_encoded_reshaped)

        
        return score
