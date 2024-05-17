#![forbid(unsafe_code)]
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use std::{
    io::{self, Write},
    ops::Shr,
};

use data_loader::DataLoader;
use gpt2::Gpt2;
use tokenizer::Tokenizer;

pub mod data_loader;
pub mod gpt2;
pub mod tokenizer;

const TRAIN_DATA: &str = "data/tiny_shakespeare_train.bin";
const VALID_DATA: &str = "data/tiny_shakespeare_val.bin";
const B: usize = 4;
const T: usize = 64;
const GEN_T: usize = 64;

fn main() -> io::Result<()> {
    let mut gpt2 = Gpt2::from_ckpt("gpt2_124M.bin", B, T);
    let mut train_loader = DataLoader::new(TRAIN_DATA, B, T);
    let mut valid_loader = DataLoader::new(VALID_DATA, B, T);

    dbg!(&gpt2.config);
    dbg!(gpt2.params.num_parameters());
    dbg!(gpt2.acts.num_parameters());

    let mut tokenizer = Tokenizer::default();
    tokenizer.init("gpt2_tokenizer.bin")?;

    let val_num_batches = 5;

    let mut rng_state = 1337u64;
    let mut gen_tokens;

    for step in 0..=1000 {
        if step % 10 == 0 {
            let mut val_loss = 0.0f32;
            valid_loader.reset();
            for _ in 0..val_num_batches {
                valid_loader.next_batch();
                let inputs = valid_loader.inputs();
                let targets = valid_loader.targets();

                gpt2.forward(inputs, Some(targets));
                val_loss += gpt2.mean_loss.unwrap();
            }
            val_loss /= val_num_batches as f32;
            println!("val_loss: {}", val_loss);
        }

        if step > 0 && step % 20 == 0 {
            gen_tokens = vec![tokenizer.eot_token as i32; B * T];

            println!("generating:\n---");
            for t in 1..GEN_T {
                gpt2.forward(&gen_tokens, None);
                let probs = &gpt2.acts.probs[(t - 1) * gpt2.config.padded_vocab_size..];
                let coin = random_f32(&mut rng_state);
                let next_token = sample_mult(probs, gpt2.config.vocab_size, coin) as i32;
                gen_tokens[t] = next_token;
                if tokenizer.init_ok {
                    if let Some(token) = tokenizer.decode(next_token as u32) {
                        Tokenizer::safe_print(token);
                    } else {
                        print!("<unk> ");
                    }
                } else {
                    print!("<{}> ", next_token);
                }
                // flush stdout
                io::stdout().flush()?;
            }
            println!("\n---");
        }

        // a training step;
        let start = std::time::Instant::now();
        train_loader.next_batch();
        let inputs = train_loader.inputs();
        let targets = train_loader.targets();
        gpt2.forward(inputs, Some(targets));
        gpt2.zero_grad();
        gpt2.backward(inputs, targets);
        gpt2.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);
        let elapsed = start.elapsed();
        println!(
            "step {}: train loss {} (took {} ms)",
            step,
            gpt2.mean_loss.unwrap(),
            elapsed.as_millis()
        );
    }

    Ok(())
}

fn random_u32(state: &mut u64) -> u32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (*state).wrapping_mul(0x2545F4914F6CDD1D).shr(32) as u32
}

fn random_f32(state: &mut u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16777216.0
}

fn sample_mult(probabilities: &[f32], n: usize, coin: f32) -> usize {
    let probabilities = &probabilities[..n];
    let mut cdf = 0.0;
    for (i, &p) in probabilities.iter().enumerate() {
        cdf += p;
        if coin < cdf {
            return i;
        }
    }
    n - 1
}
