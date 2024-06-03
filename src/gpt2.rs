use std::{f32::consts::PI, io::Read};

use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

#[derive(Debug)]
pub struct Gpt2Config {
    /// The number of layers in the model.
    pub n_layer: usize,
    /// The number of hidden units in the model.
    pub n_embed: usize,
    /// The size of the vocabulary.
    pub vocab_size: usize,
    /// The padded size of the vocabulary.
    pub padded_vocab_size: usize,
    /// The maximum sequence length.
    pub max_seq_len: usize,
    /// The number of attention heads.
    pub n_heads: usize,
}

/// The parameter of the GPT-2 model.
pub struct ParameterTensors {
    /// Weights for token embeddings, shape (Vpadded, C)
    wte: Vec<f32>,
    /// Weights for positional embeddings, shape (maxT, C)
    wpe: Vec<f32>,
    /// Weights for the first layer normalisation, shape (L, C)
    ln1w: Vec<f32>,
    /// Biases for the first layer normalisation, shape (L, C)
    ln1b: Vec<f32>,
    /// Weights for the self-attention, shape (L, 3 * C, C)
    qkvw: Vec<f32>,
    /// Biases for the self-attention, shape (L, 3 * C)
    qkvb: Vec<f32>,
    /// Weights for the self-attention output, shape (L, C, C)
    attprojw: Vec<f32>,
    /// Biases for the self-attention output, shape (L, C)
    attprojb: Vec<f32>,
    /// Weights for the second layer normalisation, shape (L, C)
    ln2w: Vec<f32>,
    /// Biases for the second layer normalisation, shape (L, C)
    ln2b: Vec<f32>,
    /// Weights for the feed-forward, shape (L, 4 * C, C)
    fcw: Vec<f32>,
    /// Biases for the feed-forward, shape (L, 4 * C)
    fcb: Vec<f32>,
    /// Weights for the feed-forward output, shape (L, C, 4 * C)
    fcprojw: Vec<f32>,
    /// Biases for the feed-forward output, shape (L, C)
    fcprojb: Vec<f32>,
    /// Weights for the final layer normalisation, shape (C)
    lnfw: Vec<f32>,
    /// Biases for the final layer normalisation, shape (C)
    lnfb: Vec<f32>,
}

impl ParameterTensors {
    pub fn zeros(config: &Gpt2Config) -> Self {
        let l = config.n_layer;
        let c = config.n_embed;
        let max_t = config.max_seq_len;
        let vp = config.padded_vocab_size;

        macro_rules! zeros {
            ($field:ident, $shape:expr) => {
                vec![0.0; $shape]
            };
        }

        Self {
            wte: zeros!(wte, vp * c),
            wpe: zeros!(wpe, max_t * c),
            ln1w: zeros!(ln1w, l * c),
            ln1b: zeros!(ln1b, l * c),
            qkvw: zeros!(qkvw, l * (3 * c) * c),
            qkvb: zeros!(qkvb, l * (3 * c)),
            attprojw: zeros!(attprojw, l * c * c),
            attprojb: zeros!(attprojb, l * c),
            ln2w: zeros!(ln2w, l * c),
            ln2b: zeros!(ln2b, l * c),
            fcw: zeros!(fcw, l * (4 * c) * c),
            fcb: zeros!(fcb, l * (4 * c)),
            fcprojw: zeros!(fcprojw, l * c * (4 * c)),
            fcprojb: zeros!(fcprojb, l * c),
            lnfw: zeros!(lnfw, c),
            lnfb: zeros!(lnfb, c),
        }
    }

    pub fn num_parameters(&self) -> usize {
        self.wte.len()
            + self.wpe.len()
            + self.ln1w.len()
            + self.ln1b.len()
            + self.qkvw.len()
            + self.qkvb.len()
            + self.attprojw.len()
            + self.attprojb.len()
            + self.ln2w.len()
            + self.ln2b.len()
            + self.fcw.len()
            + self.fcb.len()
            + self.fcprojw.len()
            + self.fcprojb.len()
            + self.lnfw.len()
            + self.lnfb.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.wte
            .iter()
            .chain(self.wpe.iter())
            .chain(self.ln1w.iter())
            .chain(self.ln1b.iter())
            .chain(self.qkvw.iter())
            .chain(self.qkvb.iter())
            .chain(self.attprojw.iter())
            .chain(self.attprojb.iter())
            .chain(self.ln2w.iter())
            .chain(self.ln2b.iter())
            .chain(self.fcw.iter())
            .chain(self.fcb.iter())
            .chain(self.fcprojw.iter())
            .chain(self.fcprojb.iter())
            .chain(self.lnfw.iter())
            .chain(self.lnfb.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.wte
            .iter_mut()
            .chain(self.wpe.iter_mut())
            .chain(self.ln1w.iter_mut())
            .chain(self.ln1b.iter_mut())
            .chain(self.qkvw.iter_mut())
            .chain(self.qkvb.iter_mut())
            .chain(self.attprojw.iter_mut())
            .chain(self.attprojb.iter_mut())
            .chain(self.ln2w.iter_mut())
            .chain(self.ln2b.iter_mut())
            .chain(self.fcw.iter_mut())
            .chain(self.fcb.iter_mut())
            .chain(self.fcprojw.iter_mut())
            .chain(self.fcprojb.iter_mut())
            .chain(self.lnfw.iter_mut())
            .chain(self.lnfb.iter_mut())
    }

    pub fn from_buffer(
        config: &Gpt2Config,
        buffer: &mut std::io::BufReader<std::fs::File>,
    ) -> Self {
        macro_rules! read_tensor {
            ($field:expr, $shape:expr) => {{
                let mut buf = vec![0u8; 4];
                for i in 0..$shape {
                    buffer.read_exact(&mut buf).unwrap();
                    let val = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
                    $field[i] = val;
                }
            }};
        }

        let mut params = Self::zeros(config);

        let max_t = config.max_seq_len;
        let vp = config.padded_vocab_size;
        let c = config.n_embed;
        let l = config.n_layer;

        read_tensor!(params.wte, vp * c);
        read_tensor!(params.wpe, max_t * c);
        read_tensor!(params.ln1w, l * c);
        read_tensor!(params.ln1b, l * c);
        read_tensor!(params.qkvw, l * (3 * c) * c);
        read_tensor!(params.qkvb, l * (3 * c));
        read_tensor!(params.attprojw, l * c * c);
        read_tensor!(params.attprojb, l * c);
        read_tensor!(params.ln2w, l * c);
        read_tensor!(params.ln2b, l * c);
        read_tensor!(params.fcw, l * (4 * c) * c);
        read_tensor!(params.fcb, l * (4 * c));
        read_tensor!(params.fcprojw, l * c * (4 * c));
        read_tensor!(params.fcprojb, l * c);
        read_tensor!(params.lnfw, c);
        read_tensor!(params.lnfb, c);

        params
    }
}

/// The activation tensors of the GPT-2 model.
///
/// This corresponds to the results of the forward/backward pass of the model.
pub struct ActivationTensors {
    /// The encoded tensor, (B, T, C)
    pub encoded: Vec<f32>,
    /// The layer normalisation tensor, (L, B, T, C)
    pub ln1: Vec<f32>,
    /// The mean of the first layer normalisation tensor, (L, B, T)
    pub ln1_mean: Vec<f32>,
    /// The rstd of the first layer normalisation tensor, (L, B, T)
    pub ln1_rstd: Vec<f32>,
    /// The self-attention tensor, (L, B, T, 3 * C)
    pub qkv: Vec<f32>,
    /// (L, B, T, C)
    pub atty: Vec<f32>,
    /// (L, B, NH, T, T)
    pub preatt: Vec<f32>,
    /// (L, B, NH, T, T)
    pub att: Vec<f32>,
    /// (L, B, T, C)
    pub attproj: Vec<f32>,
    /// (L, B, T, C)
    pub residual2: Vec<f32>,
    /// (L, B, T, C)
    pub ln2: Vec<f32>,
    /// (L, B, T)
    pub ln2_mean: Vec<f32>,
    /// (L, B, T)
    pub ln2_rstd: Vec<f32>,
    /// (L, B, T, 4 * C)
    pub fch: Vec<f32>,
    /// (L, B, T, 4 * C)
    pub fch_gelu: Vec<f32>,
    /// (L, B, T, C)
    pub fcproj: Vec<f32>,
    /// (L, B, T, C)
    pub residual3: Vec<f32>,
    /// (B, T, C)
    pub lnf: Vec<f32>,
    /// (B, T)
    pub lnf_mean: Vec<f32>,
    /// (B, T)
    pub lnf_rstd: Vec<f32>,
    /// (B, T, V)
    pub logits: Vec<f32>,
    /// (B, T, V)
    pub probs: Vec<f32>,
    /// (B, T)
    pub losses: Vec<f32>,
}

impl ActivationTensors {
    pub fn zeros(config: &Gpt2Config, batch_size: usize, seq_len: usize) -> Self {
        let b = batch_size;
        let t = seq_len;
        let c = config.n_embed;
        let l = config.n_layer;
        let vp = config.padded_vocab_size;
        let nh = config.n_heads;

        macro_rules! zeros {
            ($field:ident, $shape:expr) => {
                vec![0.0; $shape]
            };
        }

        Self {
            encoded: zeros!(encoded, b * t * c),
            ln1: zeros!(ln1, l * b * t * c),
            ln1_mean: zeros!(ln1_mean, l * b * t),
            ln1_rstd: zeros!(ln1_rstd, l * b * t),
            qkv: zeros!(qkv, l * b * t * (3 * c)),
            atty: zeros!(atty, l * b * t * c),
            preatt: zeros!(preatt, l * b * nh * t * t),
            att: zeros!(att, l * b * nh * t * t),
            attproj: zeros!(attproj, l * b * t * c),
            residual2: zeros!(residual2, l * b * t * c),
            ln2: zeros!(ln2, l * b * t * c),
            ln2_mean: zeros!(ln2_mean, l * b * t),
            ln2_rstd: zeros!(ln2_rstd, l * b * t),
            fch: zeros!(fch, l * b * t * (4 * c)),
            fch_gelu: zeros!(fch_gelu, l * b * t * (4 * c)),
            fcproj: zeros!(fcproj, l * b * t * c),
            residual3: zeros!(residual3, l * b * t * c),
            lnf: zeros!(lnf, b * t * c),
            lnf_mean: zeros!(lnf_mean, b * t),
            lnf_rstd: zeros!(lnf_rstd, b * t),
            logits: zeros!(logits, b * t * vp),
            probs: zeros!(probs, b * t * vp),
            losses: zeros!(losses, b * t),
        }
    }

    pub fn num_parameters(&self) -> usize {
        self.encoded.len()
            + self.ln1.len()
            + self.ln1_mean.len()
            + self.ln1_rstd.len()
            + self.qkv.len()
            + self.atty.len()
            + self.preatt.len()
            + self.att.len()
            + self.attproj.len()
            + self.residual2.len()
            + self.ln2.len()
            + self.ln2_mean.len()
            + self.ln2_rstd.len()
            + self.fch.len()
            + self.fch_gelu.len()
            + self.fcproj.len()
            + self.residual3.len()
            + self.lnf.len()
            + self.lnf_mean.len()
            + self.lnf_rstd.len()
            + self.logits.len()
            + self.probs.len()
            + self.losses.len()
    }
}

/// The GPT-2 model.
pub struct Gpt2 {
    /// The configuration of the model.
    pub config: Gpt2Config,
    /// The parameters of the model.
    pub params: ParameterTensors,
    /// The gradients of the parameters.
    pub grads: ParameterTensors,
    /// The activation tensors of the forward pass.
    pub acts: ActivationTensors,
    /// The gradients of the activation tensors.
    pub grads_acts: ActivationTensors,
    /// The batch size of the current forward pass.
    pub batch_size: usize,
    /// The sequence length of the current forward pass.
    pub seq_len: usize,
    /// The mean loss of the current forward pass.
    pub mean_loss: Option<f32>,
    /// AdamW m memory
    pub m_memory: ParameterTensors,
    /// AdamW v memory
    pub v_memory: ParameterTensors,
}

pub fn encoder_forward(
    out: &mut [f32],
    inp: &[i32],
    wte: &[f32],
    wpe: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    out.chunks_mut(T * C)
        .zip(inp.chunks(T))
        .take(B)
        .for_each(|(out_b, inp_b)| {
            out_b
                .chunks_mut(C)
                .zip(inp_b.iter())
                .zip(wpe.chunks(C))
                .take(T)
                .for_each(|((out_bt, &ix), wpe_t)| {
                    let wte_ix = &wte[ix as usize * C..(ix as usize + 1) * C];
                    out_bt
                        .iter_mut()
                        .zip(wte_ix.iter().zip(wpe_t.iter()))
                        .for_each(|(out_bti, (&wte_ixi, &wpe_ti))| {
                            *out_bti = wte_ixi + wpe_ti;
                        });
                });
        });
}

pub fn encoder_backward(
    dwte: &mut [f32],
    dwpe: &mut [f32],
    dout: &[f32],
    inp: &[i32],
    B: usize,
    T: usize,
    C: usize,
) {
    dout.chunks(T * C)
        .zip(inp.chunks(T))
        .take(B)
        .for_each(|(dout_b, inp_b)| {
            dout_b
                .chunks(C)
                .zip(inp_b.iter())
                .zip(dwpe.chunks_mut(C))
                .take(T)
                .for_each(|((dout_bt, &ix), dwpe_t)| {
                    let dwte_ix = &mut dwte[ix as usize * C..(ix as usize + 1) * C];
                    dout_bt
                        .iter()
                        .zip(dwte_ix.iter_mut().zip(dwpe_t.iter_mut()))
                        .for_each(|(dout_bti, (dwte_ixi, dwpe_ti))| {
                            *dwte_ixi += dout_bti;
                            *dwpe_ti += dout_bti;
                        });
                });
        });
}

pub fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    let eps = 1e-5f32;
    for b in 0..B {
        for t in 0..T {
            let x = &inp[b * T * C + t * C..b * T * C + (t + 1) * C];
            let m = x.iter().take(C).sum::<f32>() / C as f32;
            let v = x.iter().take(C).map(|xi| (xi - m) * (xi - m)).sum::<f32>() / C as f32;

            let s = 1.0f32 / (v + eps).sqrt();

            let out_bt = &mut out[b * T * C + t * C..b * T * C + (t + 1) * C];
            for i in 0..C {
                let n = s * (x[i] - m);
                let o = n * weight[i] + bias[i];
                out_bt[i] = o;
            }

            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

pub fn layernorm_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    mean: &[f32],
    rstd: &[f32],
    B: usize,
    T: usize,
    C: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let dout_bt = &dout[b * T * C + t * C..b * T * C + (t + 1) * C];
            let inp_bt = &inp[b * T * C + t * C..b * T * C + (t + 1) * C];
            let dinp_bt = &mut dinp[b * T * C + t * C..b * T * C + (t + 1) * C];
            let mean_bt = mean[b * T + t];
            let rstd_bt = rstd[b * T + t];

            let mut dnorm_mean = 0.0f32;
            let mut dnorm_norm_mean = 0.0f32;
            for i in 0..C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;

            for i in 0..C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                let dnorm_i = weight[i] * dout_bt[i];
                dbias[i] += dout_bt[i];
                dweight[i] += norm_bti * dout_bt[i];

                let mut dval = 0.0f32;
                dval += dnorm_i;
                dval -= dnorm_mean;
                dval -= norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                dinp_bt[i] += dval;
            }
        }
    }
}

pub fn crossentropy_forward(
    losses: &mut [f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    Vp: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let probs_bt = &probs[b * T * Vp + t * Vp..b * T * Vp + (t + 1) * Vp];
            let ix = targets[b * T + t] as usize;
            losses[b * T + t] = -probs_bt[ix].ln();
        }
    }
}

pub fn crossentropy_softmax_backward(
    dlogits: &mut [f32],
    dlosses: &[f32],
    probs: &[f32],
    targets: &[i32],
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    for b in 0..B {
        for t in 0..T {
            let dlogits_bt = &mut dlogits[b * T * Vp + t * Vp..b * T * Vp + (t + 1) * Vp];
            let probs_bt = &probs[b * T * Vp + t * Vp..b * T * Vp + (t + 1) * Vp];
            let dloss = dlosses[b * T + t];
            let ix = targets[b * T + t] as usize;
            for i in 0..V {
                let p = probs_bt[i];
                let indicator = if i == ix { 1.0 } else { 0.0 };
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

pub fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    for b in 0..B {
        let out_b = &mut out[b * T * OC..b * T * OC + T * OC];
        let inp_b = &inp[b * T * C..b * T * C + T * C];
        // compute each row of the output
        out_b
            .par_chunks_mut(OC)
            .zip(inp_b.par_chunks(C))
            .take(T)
            .for_each(|(row, inp_bt)| {
                row.iter_mut().enumerate().for_each(|(o, val)| {
                    let mut sum = if let Some(bias) = bias { bias[o] } else { 0.0 };
                    let wrow = &weight[o * C..(o + 1) * C];
                    for i in 0..C {
                        sum += inp_bt[i] * wrow[i];
                    }
                    *val = sum;
                });
            })
    }
}

pub fn matmul_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: Option<&mut [f32]>,
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    // backward into inp first
    for b in 0..B {
        let dout_b = &dout[b * T * OC..b * T * OC + T * OC];
        let dinp_b = &mut dinp[b * T * C..b * T * C + T * C];
        dout_b
            .par_chunks(OC)
            .zip(dinp_b.par_chunks_mut(C))
            .take(T)
            .for_each(|(dout_bt, dinp_bt)| {
                dout_bt.iter().enumerate().for_each(|(o, d)| {
                    let wrow = &weight[o * C..(o + 1) * C];
                    for i in 0..C {
                        dinp_bt[i] += wrow[i] * d;
                    }
                });
            });
    }

    if let Some(&mut ref mut dbias) = dbias {
        dweight
            .par_chunks_mut(C)
            .zip(dbias.par_iter_mut())
            .take(OC)
            .enumerate()
            .for_each(|(o, (dwrow, dbias_o))| {
                for b in 0..B {
                    for t in 0..T {
                        let dout_bt = &dout[b * T * OC + t * OC..b * T * OC + (t + 1) * OC];
                        let inp_bt = &inp[b * T * C + t * C..b * T * C + (t + 1) * C];
                        let d = dout_bt[o];
                        *dbias_o += d;
                        for i in 0..C {
                            dwrow[i] += inp_bt[i] * d;
                        }
                    }
                }
            })
    } else {
        dweight
            .par_chunks_mut(C)
            .take(OC)
            .enumerate()
            .for_each(|(o, dwrow)| {
                for b in 0..B {
                    for t in 0..T {
                        let dout_bt = &dout[b * T * OC + t * OC..b * T * OC + (t + 1) * OC];
                        let inp_bt = &inp[b * T * C + t * C..b * T * C + (t + 1) * C];
                        let d = dout_bt[o];
                        for i in 0..C {
                            dwrow[i] += inp_bt[i] * d;
                        }
                    }
                }
            });
    }
}

pub fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3;
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt();
    // TODO: parallelize this
    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let query_t =
                    &inp[b * T * C3 + t * C3 + h * hs..b * T * C3 + t * C3 + (h + 1) * hs];
                let preatt_bth = &mut preatt
                    [b * NH * T * T + h * T * T + t * T..b * NH * T * T + h * T * T + (t + 1) * T];
                let att_bth = &mut att
                    [b * NH * T * T + h * T * T + t * T..b * NH * T * T + h * T * T + (t + 1) * T];

                // pass 1: calculate query dot key and maxval
                let mut maxval = -10000.0f32; // just like in llm.c
                for t2 in 0..=t {
                    let key_t2 = &inp
                        [b * T * C3 + t2 * C3 + h * hs + C..b * T * C3 + t2 * C3 + h * hs + C + hs];

                    // (query_t) dot (key_t2)
                    let mut val = 0.0f32;
                    for i in 0..hs {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                let mut expsum = 0.0f32;
                for t2 in 0..=t {
                    let expv = (preatt_bth[t2] - maxval).exp();
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                att_bth
                    .iter_mut()
                    .take(t + 1)
                    .for_each(|x| *x *= expsum_inv);
                att_bth
                    .iter_mut()
                    .take(T)
                    .skip(t + 1)
                    .for_each(|x| *x = 0.0);

                // pass 4: accumulate weighted values into the output of attention
                let out_bth =
                    &mut out[b * T * C + t * C + h * hs..b * T * C + t * C + (h + 1) * hs];
                out_bth.iter_mut().for_each(|x| *x = 0.0);
                for t2 in 0..=t {
                    let value_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + 2 * C
                        ..b * T * C3 + t2 * C3 + h * hs + 2 * C + hs];
                    let att_btht2 = att_bth[t2];
                    for i in 0..hs {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

pub fn attention_backward(
    dinp: &mut [f32],
    dpreatt: &mut [f32],
    datt: &mut [f32],
    dout: &[f32],
    inp: &[f32],
    att: &[f32],
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3;
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt();

    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                let att_bth = &att
                    [b * NH * T * T + h * T * T + t * T..b * NH * T * T + h * T * T + (t + 1) * T];
                let datt_bth = &mut datt
                    [b * NH * T * T + h * T * T + t * T..b * NH * T * T + h * T * T + (t + 1) * T];
                let dpreatt_bth = &mut dpreatt
                    [b * NH * T * T + h * T * T + t * T..b * NH * T * T + h * T * T + (t + 1) * T];
                let query_t =
                    &inp[b * T * C3 + t * C3 + h * hs..b * T * C3 + t * C3 + (h + 1) * hs];

                // backward pass 4, through the value accumulation
                let dout_bth = &dout[b * T * C + t * C + h * hs..b * T * C + t * C + (h + 1) * hs];
                for t2 in 0..=t {
                    let value_t2 = &inp[b * T * C3 + t2 * C3 + h * hs + 2 * C
                        ..b * T * C3 + t2 * C3 + h * hs + 2 * C + hs];
                    let dvalue_t2 = &mut dinp[b * T * C3 + t2 * C3 + h * hs + 2 * C
                        ..b * T * C3 + t2 * C3 + h * hs + 2 * C + hs];
                    for i in 0..hs {
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                        let local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                let mut dquery_t_add = vec![0.0; hs];
                for t2 in 0..=t {
                    let key_t2 = &inp
                        [b * T * C3 + t2 * C3 + h * hs + C..b * T * C3 + t2 * C3 + h * hs + C + hs];
                    let dkey_t2 = &mut dinp
                        [b * T * C3 + t2 * C3 + h * hs + C..b * T * C3 + t2 * C3 + h * hs + C + hs];
                    for i in 0..hs {
                        dquery_t_add[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
                let dquery_t =
                    &mut dinp[b * T * C3 + t * C3 + h * hs..b * T * C3 + t * C3 + (h + 1) * hs];
                for i in 0..hs {
                    dquery_t[i] += dquery_t_add[i];
                }
            }
        }
    }
}

pub fn gelu_forward(out: &mut [f32], inp: &[f32], N: usize) {
    let GELU_SCALING_FACTOR: f32 = (2.0 / PI).sqrt();
    out.iter_mut()
        .zip(inp.iter())
        .take(N)
        .for_each(|(out_i, &inp_i)| {
            let x = inp_i;
            let cube = 0.044715 * x * x * x;
            *out_i = 0.5 * x * (1.0 + (GELU_SCALING_FACTOR * (x + cube)).tanh());
        });
}

pub fn gelu_backward(dinp: &mut [f32], inp: &[f32], dout: &[f32], N: usize) {
    let GELU_SCALING_FACTOR: f32 = (2.0 / PI).sqrt();
    dinp.iter_mut()
        .zip(inp.iter().zip(dout.iter()))
        .take(N)
        .for_each(|(dinp_i, (x, dout_i))| {
            let cube = 0.044715 * x * x * x;
            let tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            let tanh_out = tanh_arg.tanh();
            let coshf_out = tanh_arg.cosh();
            let sech_out = 1.0 / (coshf_out * coshf_out);
            let local_grad = 0.5 * (1.0 + tanh_out)
                + x * 0.5 * sech_out * GELU_SCALING_FACTOR * (1.0 + 3.0 * 0.044715 * x * x);
            *dinp_i += local_grad * dout_i;
        });
}

pub fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32], N: usize) {
    out.iter_mut()
        .zip(inp1.iter().zip(inp2.iter()))
        .take(N)
        .for_each(|(out_i, (&inp1_i, &inp2_i))| {
            *out_i = inp1_i + inp2_i;
        });
}

pub fn residual_backward(dinp1: &mut [f32], dinp2: &mut [f32], dout: &[f32], N: usize) {
    dout.iter()
        .zip(dinp1.iter_mut().zip(dinp2.iter_mut()))
        .take(N)
        .for_each(|(&dout_i, (dinp1_i, dinp2_i))| {
            *dinp1_i += dout_i;
            *dinp2_i += dout_i;
        });
}

pub fn softmax_forward(probs: &mut [f32], logits: &[f32], B: usize, T: usize, V: usize, Vp: usize) {
    for b in 0..B {
        for t in 0..T {
            let logits_bt = &logits[b * T * Vp + t * Vp..b * T * Vp + (t + 1) * Vp];
            let probs_bt = &mut probs[b * T * Vp + t * Vp..b * T * Vp + (t + 1) * Vp];

            let maxval = logits_bt.iter().take(V).fold(-10000.0f32, |a, b| a.max(*b));
            let mut sum = 0.0f32;
            for i in 0..V {
                probs_bt[i] = (logits_bt[i] - maxval).exp();
                sum += probs_bt[i];
            }
            probs_bt.iter_mut().take(V).for_each(|x| *x /= sum);
            probs_bt.iter_mut().take(Vp).skip(V).for_each(|x| *x = 0.0);
        }
    }
}

impl Gpt2 {
    pub fn from_ckpt(ckpt_path: &str, batch_size: usize, seq_len: usize) -> Self {
        let model_file = std::fs::File::open(ckpt_path).unwrap();
        let mut reader = std::io::BufReader::new(model_file);

        let mut model_header = [0i32; 256];

        // read the header
        let mut header_bytes = [0u8; 1024];
        reader.read_exact(&mut header_bytes).unwrap();
        for i in 0..256 {
            model_header[i] = i32::from_le_bytes([
                header_bytes[i * 4],
                header_bytes[i * 4 + 1],
                header_bytes[i * 4 + 2],
                header_bytes[i * 4 + 3],
            ]);
        }

        assert_eq!(model_header[0], 20240326, "bad magic model fuke");
        assert_eq!(model_header[1], 3, "bad version in model file");

        let config = Gpt2Config {
            max_seq_len: model_header[2] as usize,
            vocab_size: model_header[3] as usize,
            n_layer: model_header[4] as usize,
            n_heads: model_header[5] as usize,
            n_embed: model_header[6] as usize,
            padded_vocab_size: model_header[7] as usize,
        };

        let params = ParameterTensors::from_buffer(&config, &mut reader);

        let grads = ParameterTensors::zeros(&config);
        let acts = ActivationTensors::zeros(&config, batch_size, seq_len);
        let grads_acts = ActivationTensors::zeros(&config, batch_size, seq_len);
        let m_memory = ParameterTensors::zeros(&config);
        let v_memory = ParameterTensors::zeros(&config);

        Self {
            config,
            params,
            grads,
            acts,
            grads_acts,
            batch_size,
            seq_len,
            mean_loss: None,
            m_memory,
            v_memory,
        }
    }

    pub fn forward(&mut self, inputs: &[i32], targets: Option<&[i32]>) {
        // prepare for the forward pass
        let v = self.config.vocab_size;
        let vp = self.config.padded_vocab_size;
        let nh = self.config.n_heads;
        let c = self.config.n_embed;

        // validate inputs, all indices must be in the range [0, v)
        inputs
            .iter()
            .for_each(|&x| assert!(x >= 0 && x < v as i32, "input index out of range"));
        if let Some(t) = targets {
            t.iter()
                .for_each(|&x| assert!(x >= 0 && x < v as i32, "target index out of range"))
        }

        let b = self.batch_size;
        let t = self.seq_len;
        // zero the activation tensors
        self.acts = ActivationTensors::zeros(&self.config, b, t);

        let params = &self.params;
        let acts = &mut self.acts;

        encoder_forward(&mut acts.encoded, inputs, &params.wte, &params.wpe, b, t, c);
        for layer in 0..self.config.n_layer {
            let residual = if layer == 0 {
                &acts.encoded
            } else {
                &acts.residual3[(layer - 1) * b * t * c..layer * b * t * c]
            };
            let l_ln1 = &mut acts.ln1[layer * b * t * c..(layer + 1) * b * t * c];
            let l_ln1_mean = &mut acts.ln1_mean[layer * b * t..(layer + 1) * b * t];
            let l_ln1_rstd = &mut acts.ln1_rstd[layer * b * t..(layer + 1) * b * t];
            let l_ln1w = &params.ln1w[layer * c..(layer + 1) * c];
            let l_ln1b = &params.ln1b[layer * c..(layer + 1) * c];
            layernorm_forward(
                l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, b, t, c,
            );

            let l_qkv = &mut acts.qkv[layer * b * t * (3 * c)..(layer + 1) * b * t * (3 * c)];
            let l_qkvw = &params.qkvw[layer * (3 * c) * c..(layer + 1) * (3 * c) * c];
            let l_qkvb = &params.qkvb[layer * (3 * c)..(layer + 1) * (3 * c)];
            matmul_forward(l_qkv, l_ln1, l_qkvw, Some(l_qkvb), b, t, c, 3 * c);

            let l_atty = &mut acts.atty[layer * b * t * c..(layer + 1) * b * t * c];
            let l_preatt = &mut acts.preatt[layer * b * nh * t * t..(layer + 1) * b * nh * t * t];
            let l_att = &mut acts.att[layer * b * nh * t * t..(layer + 1) * b * nh * t * t];
            attention_forward(l_atty, l_preatt, l_att, l_qkv, b, t, c, nh);

            let l_attproj = &mut acts.attproj[layer * b * t * c..(layer + 1) * b * t * c];
            let l_attprojw = &params.attprojw[layer * c * c..(layer + 1) * c * c];
            let l_attprojb = &params.attprojb[layer * c..(layer + 1) * c];
            matmul_forward(l_attproj, l_atty, l_attprojw, Some(l_attprojb), b, t, c, c);

            let l_residual2 = &mut acts.residual2[layer * b * t * c..(layer + 1) * b * t * c];
            residual_forward(l_residual2, residual, l_attproj, b * t * c);

            let l_ln2 = &mut acts.ln2[layer * b * t * c..(layer + 1) * b * t * c];
            let l_ln2_mean = &mut acts.ln2_mean[layer * b * t..(layer + 1) * b * t];
            let l_ln2_rstd = &mut acts.ln2_rstd[layer * b * t..(layer + 1) * b * t];
            let l_ln2w = &params.ln2w[layer * c..(layer + 1) * c];
            let l_ln2b = &params.ln2b[layer * c..(layer + 1) * c];
            layernorm_forward(
                l_ln2,
                l_ln2_mean,
                l_ln2_rstd,
                l_residual2,
                l_ln2w,
                l_ln2b,
                b,
                t,
                c,
            );

            let l_fch = &mut acts.fch[layer * b * t * (4 * c)..(layer + 1) * b * t * (4 * c)];
            let l_fcw = &params.fcw[layer * (4 * c) * c..(layer + 1) * (4 * c) * c];
            let l_fcb = &params.fcb[layer * (4 * c)..(layer + 1) * (4 * c)];
            matmul_forward(l_fch, l_ln2, l_fcw, Some(l_fcb), b, t, c, 4 * c);

            let l_fch_gelu =
                &mut acts.fch_gelu[layer * b * t * (4 * c)..(layer + 1) * b * t * (4 * c)];
            gelu_forward(l_fch_gelu, l_fch, b * t * (4 * c));

            let l_fcproj = &mut acts.fcproj[layer * b * t * c..(layer + 1) * b * t * c];
            let l_fcprojw = &params.fcprojw[layer * c * (4 * c)..(layer + 1) * c * (4 * c)];
            let l_fcprojb = &params.fcprojb[layer * c..(layer + 1) * c];
            matmul_forward(
                l_fcproj,
                l_fch_gelu,
                l_fcprojw,
                Some(l_fcprojb),
                b,
                t,
                4 * c,
                c,
            );

            let l_residual3 = &mut acts.residual3[layer * b * t * c..(layer + 1) * b * t * c];
            let l_residual2 = &acts.residual2[layer * b * t * c..(layer + 1) * b * t * c];
            residual_forward(l_residual3, l_residual2, l_fcproj, b * t * c);
        }
        let residual =
            &acts.residual3[(self.config.n_layer - 1) * b * t * c..self.config.n_layer * b * t * c];
        layernorm_forward(
            &mut acts.lnf,
            &mut acts.lnf_mean,
            &mut acts.lnf_rstd,
            residual,
            &params.lnfw,
            &params.lnfb,
            b,
            t,
            c,
        );
        matmul_forward(&mut acts.logits, &acts.lnf, &params.wte, None, b, t, c, vp);
        softmax_forward(&mut acts.probs, &acts.logits, b, t, v, vp);

        if let Some(targets) = targets {
            crossentropy_forward(&mut acts.losses, &acts.probs, targets, b, t, vp);
            self.mean_loss = Some(acts.losses.iter().sum::<f32>() / (b * t) as f32);
        } else {
            self.mean_loss = None;
        }
    }

    pub fn zero_grad(&mut self) {
        self.grads = ParameterTensors::zeros(&self.config);
        self.grads_acts = ActivationTensors::zeros(&self.config, self.batch_size, self.seq_len);
    }

    pub fn backward(&mut self, inputs: &[i32], targets: &[i32]) {
        assert!(self.mean_loss.is_some(), "no loss to backpropagate");

        self.zero_grad();

        let b = self.batch_size;
        let t = self.seq_len;
        let v = self.config.vocab_size;
        let vp = self.config.padded_vocab_size;
        let l = self.config.n_layer;
        let nh = self.config.n_heads;
        let c = self.config.n_embed;

        let params = &self.params;
        let acts = &self.acts;
        let grads = &mut self.grads;
        let grads_acts = &mut self.grads_acts;

        let dloss_mean = 1.0 / (b * t) as f32;
        grads_acts.losses = vec![dloss_mean; b * t];

        crossentropy_softmax_backward(
            &mut grads_acts.logits,
            &grads_acts.losses,
            &acts.probs,
            targets,
            b,
            t,
            v,
            vp,
        );
        matmul_backward(
            &mut grads_acts.lnf,
            &mut grads.wte,
            None,
            &grads_acts.logits,
            &acts.lnf,
            &params.wte,
            b,
            t,
            c,
            vp,
        );
        let residual = &acts.residual3[(l - 1) * b * t * c..l * b * t * c];
        let dresidual = &mut grads_acts.residual3[(l - 1) * b * t * c..l * b * t * c];
        layernorm_backward(
            dresidual,
            &mut grads.lnfw,
            &mut grads.lnfb,
            &grads_acts.lnf,
            residual,
            &params.lnfw,
            &acts.lnf_mean,
            &acts.lnf_rstd,
            b,
            t,
            c,
        );

        for layer in (0..l).rev() {
            let dl_residual2 =
                &mut grads_acts.residual2[layer * b * t * c..(layer + 1) * b * t * c];
            let dl_fcproj = &mut grads_acts.fcproj[layer * b * t * c..(layer + 1) * b * t * c];
            let dl_residual3 =
                &mut grads_acts.residual3[layer * b * t * c..(layer + 1) * b * t * c];
            residual_backward(dl_residual2, dl_fcproj, dl_residual3, b * t * c);

            let dl_fch_gelu =
                &mut grads_acts.fch_gelu[layer * b * t * (4 * c)..(layer + 1) * b * t * (4 * c)];
            let dl_fcprojw = &mut grads.fcprojw[layer * c * (4 * c)..(layer + 1) * c * (4 * c)];
            let dl_fcprojb = &mut grads.fcprojb[layer * c..(layer + 1) * c];
            let l_fch_gelu = &acts.fch_gelu[layer * b * t * (4 * c)..(layer + 1) * b * t * (4 * c)];
            let l_fcprojw = &params.fcprojw[layer * c * (4 * c)..(layer + 1) * c * (4 * c)];
            matmul_backward(
                dl_fch_gelu,
                dl_fcprojw,
                Some(dl_fcprojb),
                dl_fcproj,
                l_fch_gelu,
                l_fcprojw,
                b,
                t,
                4 * c,
                c,
            );

            let dl_fch =
                &mut grads_acts.fch[layer * b * t * (4 * c)..(layer + 1) * b * t * (4 * c)];
            let l_fch = &acts.fch[layer * b * t * (4 * c)..(layer + 1) * b * t * (4 * c)];
            let dl_fch_gelu =
                &grads_acts.fch_gelu[layer * b * t * (4 * c)..(layer + 1) * b * t * (4 * c)];
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, b * t * (4 * c));

            let dl_ln2 = &mut grads_acts.ln2[layer * b * t * c..(layer + 1) * b * t * c];
            let dl_fcw = &mut grads.fcw[layer * (4 * c) * c..(layer + 1) * (4 * c) * c];
            let dl_fcb = &mut grads.fcb[layer * (4 * c)..(layer + 1) * (4 * c)];
            let dl_fch = &grads_acts.fch[layer * b * t * (4 * c)..(layer + 1) * b * t * (4 * c)];
            let l_ln2 = &acts.ln2[layer * b * t * c..(layer + 1) * b * t * c];
            let l_fcw = &params.fcw[layer * (4 * c) * c..(layer + 1) * (4 * c) * c];
            matmul_backward(
                dl_ln2,
                dl_fcw,
                Some(dl_fcb),
                dl_fch,
                l_ln2,
                l_fcw,
                b,
                t,
                c,
                4 * c,
            );

            let dl_ln2w = &mut grads.ln2w[layer * c..(layer + 1) * c];
            let dl_ln2b = &mut grads.ln2b[layer * c..(layer + 1) * c];
            let dl_ln2 = &grads_acts.ln2[layer * b * t * c..(layer + 1) * b * t * c];
            let l_residual2 = &acts.residual2[layer * b * t * c..(layer + 1) * b * t * c];
            let l_ln2w = &params.ln2w[layer * c..(layer + 1) * c];
            let l_ln2_mean = &acts.ln2_mean[layer * b * t..(layer + 1) * b * t];
            let l_ln2_rstd = &acts.ln2_rstd[layer * b * t..(layer + 1) * b * t];
            layernorm_backward(
                dl_residual2,
                dl_ln2w,
                dl_ln2b,
                dl_ln2,
                l_residual2,
                l_ln2w,
                l_ln2_mean,
                l_ln2_rstd,
                b,
                t,
                c,
            );

            let dresidual = if layer == 0 {
                &mut grads_acts.encoded
            } else {
                &mut grads_acts.residual3[(layer - 1) * b * t * c..layer * b * t * c]
            };
            let dl_attproj = &mut grads_acts.attproj[layer * b * t * c..(layer + 1) * b * t * c];
            residual_backward(dresidual, dl_attproj, dl_residual2, b * t * c);

            let dl_atty = &mut grads_acts.atty[layer * b * t * c..(layer + 1) * b * t * c];
            let dl_attprojw = &mut grads.attprojw[layer * c * c..(layer + 1) * c * c];
            let dl_attprojb = &mut grads.attprojb[layer * c..(layer + 1) * c];
            let dl_attproj = &grads_acts.attproj[layer * b * t * c..(layer + 1) * b * t * c];
            let l_atty = &acts.atty[layer * b * t * c..(layer + 1) * b * t * c];
            let l_attprojw = &params.attprojw[layer * c * c..(layer + 1) * c * c];
            matmul_backward(
                dl_atty,
                dl_attprojw,
                Some(dl_attprojb),
                dl_attproj,
                l_atty,
                l_attprojw,
                b,
                t,
                c,
                c,
            );

            let dl_qkv =
                &mut grads_acts.qkv[layer * b * t * (3 * c)..(layer + 1) * b * t * (3 * c)];
            let dl_preatt =
                &mut grads_acts.preatt[layer * b * nh * t * t..(layer + 1) * b * nh * t * t];
            let dl_att = &mut grads_acts.att[layer * b * nh * t * t..(layer + 1) * b * nh * t * t];
            let dl_atty = &grads_acts.atty[layer * b * t * c..(layer + 1) * b * t * c];
            let l_qkv = &acts.qkv[layer * b * t * (3 * c)..(layer + 1) * b * t * (3 * c)];
            let l_att = &acts.att[layer * b * nh * t * t..(layer + 1) * b * nh * t * t];
            attention_backward(
                dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, b, t, c, nh,
            );

            let dl_ln1 = &mut grads_acts.ln1[layer * b * t * c..(layer + 1) * b * t * c];
            let dl_qkvw = &mut grads.qkvw[layer * (3 * c) * c..(layer + 1) * (3 * c) * c];
            let dl_qkvb = &mut grads.qkvb[layer * (3 * c)..(layer + 1) * (3 * c)];
            let l_ln1 = &acts.ln1[layer * b * t * c..(layer + 1) * b * t * c];
            let l_qkvw = &params.qkvw[layer * (3 * c) * c..(layer + 1) * (3 * c) * c];
            matmul_backward(
                dl_ln1,
                dl_qkvw,
                Some(dl_qkvb),
                dl_qkv,
                l_ln1,
                l_qkvw,
                b,
                t,
                c,
                3 * c,
            );

            let dl_ln1w = &mut grads.ln1w[layer * c..(layer + 1) * c];
            let dl_ln1b = &mut grads.ln1b[layer * c..(layer + 1) * c];
            let dl_ln1 = &grads_acts.ln1[layer * b * t * c..(layer + 1) * b * t * c];
            let residual = if layer == 0 {
                &acts.encoded
            } else {
                &acts.residual3[(layer - 1) * b * t * c..layer * b * t * c]
            };
            let l_ln1w = &params.ln1w[layer * c..(layer + 1) * c];
            let l_ln1_mean = &acts.ln1_mean[layer * b * t..(layer + 1) * b * t];
            let l_ln1_rstd = &acts.ln1_rstd[layer * b * t..(layer + 1) * b * t];
            layernorm_backward(
                dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, b,
                t, c,
            );
        }
        encoder_backward(
            &mut grads.wte,
            &mut grads.wpe,
            &grads_acts.encoded,
            inputs,
            b,
            t,
            c,
        );
    }

    pub fn update(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, t: i32) {
        self.params
            .iter_mut()
            .zip(self.grads.iter())
            .zip(self.m_memory.iter_mut())
            .zip(self.v_memory.iter_mut())
            .for_each(|(((param, grad), m), v)| {
                let m_updated = beta1 * *m + (1.0 - beta1) * grad;
                let v_updated = beta2 * *v + (1.0 - beta2) * grad * grad;
                let m_hat = m_updated / (1.0 - beta1.powi(t));
                let v_hat = v_updated / (1.0 - beta2.powi(t));

                *m = m_updated;
                *v = v_updated;
                *param -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * *param);
            });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gpt2_parameters() {
        use super::*;

        // gpt2-small
        let config = Gpt2Config {
            n_layer: 12,
            n_embed: 768,
            vocab_size: 50257,
            padded_vocab_size: 50304,
            max_seq_len: 1024,
            n_heads: 12,
        };

        let params = ParameterTensors::zeros(&config);
        // the same batch_size and seq_len as the llm.c
        let acts = ActivationTensors::zeros(&config, 4, 64);
        assert_eq!(params.num_parameters(), 124475904);
        assert_eq!(acts.num_parameters(), 73347840);
    }
}
