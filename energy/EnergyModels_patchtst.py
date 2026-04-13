import torch
from argparse import Namespace

from energy.EnergyModelsV2 import ArbitraryMLPDecoder, MultisampleMLPDecoder
from energy.EnergyModelsV3 import AutoformerNeoEBM
from models import PatchTST


class PatchTSTNeoEBM_concat(AutoformerNeoEBM):
    def __init__(
        self,
        setting: Namespace,
        patch_tst_model: PatchTST,
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
        super(PatchTSTNeoEBM_concat, self).__init__(
            setting,
            patch_tst_model,
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
        self.patch_tst_model = patch_tst_model

        # head_nf = d_model * magic_number
        self.magic_number = patch_tst_model.head_nf // patch_tst_model.d_model
        if self.magic_number != 12:
            raise ValueError(f"Magic number is not 12, but {self.magic_number}!")

    def get_assumed_enc_out_shape(self, args):
        # [batch, n_vars, d_model, magic_number]
        assumed_enc_out_shape = [
            args.batch_size,
            args.enc_in,
            args.d_model,
            self.magic_number,
        ]
        return assumed_enc_out_shape

    def get_assumed_dec_out_shape(self, args):
        # [batch, seq_len + pred_len, n_vars]
        assumed_dec_out_shape = [
            args.batch_size,
            args.seq_len + args.pred_len,
            args.dec_in,
        ]
        return assumed_dec_out_shape

    def setup_y_encoder_and_xy_decoder_(
        self, seq_len, label_len, pred_len, d_model, c_out, dec_out_2_dim
    ):
        # Multisample MLP over the prediction window and output channels
        self.y_encoder = MultisampleMLPDecoder(
            input_dim=pred_len * c_out,
            output_dim=dec_out_2_dim * d_model * self.magic_number,
            num_layers=self.y_decoder_num_layers,
            hidden_size=self.y_decoder_code_dim,
        )
        self.xy_decoder = ArbitraryMLPDecoder(
            input_dim=(dec_out_2_dim * d_model * self.magic_number) * 2,
            output_dim=1,
            num_layers=self.decoder_num_layers,
            hidden_size=self.decoder_hidden_dim,
        )
        self.orig_model_seq_len = seq_len
        self.orig_model_label_len = label_len
        self.orig_model_pred_len = pred_len
        self.orig_model_d_model = d_model

        self.dec_out_2_dim = dec_out_2_dim
        print("Setup for of Y encoder and XY decoder done!")

    def _forward_y_enc(self, batch_y: torch.Tensor) -> torch.Tensor:
        """Encode prediction-window targets; shape matches ``enc_out`` for ``cat``."""

        actual_y = batch_y[:, -self.orig_model_pred_len :, :]
        B, T, C = actual_y.shape
        # setup uses input_dim = pred_len * c_out; MultisampleMLPDecoder needs
        # [B, pred_len * c_out, 1], not [B, pred_len, c_out] (misread as length pred_len
        # with num_samples = c_out).
        y_in = actual_y.reshape(B, T * C, 1)
        encoded_y = self.y_encoder(y_in)

        odim = self.y_encoder.output_dim
        dmd = self.orig_model_d_model
        mn = self.magic_number
        dec_dim = self.dec_out_2_dim
        expected = dec_dim * dmd * mn
        if odim != expected:
            raise RuntimeError(
                f"PatchTSTNeoEBM: y_encoder.output_dim={odim} != "
                f"dec_out_2_dim*d_model*magic_number={expected}"
            )
        if encoded_y.shape != (B, odim):
            raise RuntimeError(
                f"PatchTSTNeoEBM: encoded_y shape {tuple(encoded_y.shape)}; "
                f"expected ({B}, {odim})"
            )
        return encoded_y.view(B, dec_dim, dmd, mn)

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
        encoded_y = self._forward_y_enc(batch_y)
        xy_encoded = torch.cat([encoded_x, encoded_y], dim=2)
        xy_encoded_reshaped = xy_encoded.view(xy_encoded.shape[0], -1)
        score = self.xy_decoder(xy_encoded_reshaped)
        return score
