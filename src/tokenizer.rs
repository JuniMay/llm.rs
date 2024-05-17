use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

#[derive(Debug)]
pub struct Tokenizer {
    vocab_size: u32,
    token_table: Vec<String>,
    pub init_ok: bool,
    pub eot_token: u32,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self {
            vocab_size: 0,
            token_table: Vec::new(),
            init_ok: false,
            eot_token: 50256,
        }
    }
}

impl Tokenizer {
    pub fn safe_print(piece: &str) {
        if piece
            .chars()
            .all(|c| c.is_ascii() && (c.is_ascii_graphic() || c.is_whitespace()))
        {
            print!("{}", piece);
        }
    }

    pub fn init(&mut self, filename: &str) -> io::Result<()> {
        let path = Path::new(filename);
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut header = [0u32; 256];
        for i in 0..256 {
            header[i] = Self::read_u32(&mut reader)?;
        }
        assert_eq!(header[0], 20240328);
        let version = header[1];
        self.vocab_size = header[2];

        if version == 1 {
            assert_eq!(self.vocab_size, 50257);
        } else if version == 2 {
            self.eot_token = header[3];
        } else {
            panic!(
                "Tokenizer model file {} has bad version: {}",
                filename, version
            );
        }

        self.token_table = Vec::with_capacity(self.vocab_size as usize);
        for _ in 0..self.vocab_size {
            let length = Self::read_u8(&mut reader)? as usize;
            assert!(length > 0);
            let mut token_bytes = vec![0; length];
            reader.read_exact(&mut token_bytes)?;
            let token_str = String::from_utf8_lossy(&token_bytes).into_owned();
            self.token_table.push(token_str);
        }

        self.init_ok = true;
        Ok(())
    }

    pub fn decode(&self, token_id: u32) -> Option<&str> {
        if !self.init_ok {
            return None;
        }
        if token_id < self.vocab_size {
            Some(&self.token_table[token_id as usize])
        } else {
            eprintln!("invalid token id {}!", token_id);
            None
        }
    }

    fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer)?;
        Ok(u32::from_le_bytes(buffer))
    }

    fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
        let mut buffer = [0u8; 1];
        reader.read_exact(&mut buffer)?;
        Ok(buffer[0])
    }
}
