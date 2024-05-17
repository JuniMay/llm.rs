use std::io::{Read, Seek};

pub struct DataLoader {
    b: usize,
    t: usize,
    file: std::fs::File,
    file_size: usize,
    cursor: usize,
    batch: Vec<i32>,
    num_batches: usize,
}

impl DataLoader {
    pub fn new(path: &str, b: usize, t: usize) -> Self {
        let file = std::fs::File::open(path).unwrap();
        let file_size = file.metadata().unwrap().len() as usize;
        assert!(file_size >= (b * t + 1) * 4, "file too small");

        let num_batches = file_size / (b * t * 4);
        let batch = vec![0; b * t + 1];

        Self {
            b,
            t,
            file,
            file_size,
            cursor: 0,
            batch,
            num_batches,
        }
    }

    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    pub fn inputs(&self) -> &[i32] {
        &self.batch[..self.b * self.t]
    }

    pub fn targets(&self) -> &[i32] {
        &self.batch[1..self.b * self.t + 1]
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn next_batch(&mut self) {
        let b = self.b;
        let t = self.t;

        if self.cursor + (b * t + 1) * 4 > self.file_size {
            self.reset();
        }

        self.file
            .seek(std::io::SeekFrom::Start(self.cursor as u64))
            .unwrap();

        let mut bytes = vec![0; (b * t + 1) * 4];

        self.file.read_exact(&mut bytes).unwrap();

        for i in 0..b * t + 1 {
            self.batch[i] = i32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }

        self.cursor += b * t * 4;
    }
}
