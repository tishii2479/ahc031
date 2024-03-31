use crate::util::*;
use proconio::input;

pub const EXCEED_COST: i64 = 100;
pub const TIME_LIMIT: f64 = 2.95;

#[allow(non_snake_case)]
pub struct Input {
    pub W: i64,
    pub D: usize,
    pub N: usize,
    pub A: Vec<Vec<i64>>,
}

impl Input {
    #[allow(non_snake_case)]
    pub fn read_input() -> Input {
        input! {
            W: i64, D: usize, N: usize,
            A: [[i64; N]; D]
        }
        Input { W, D, N, A }
    }
}

pub struct Answer {
    pub p: Vec<Vec<(i64, i64, i64, i64)>>,
    pub score: i64,
}

impl Answer {
    pub fn new(d: usize, n: usize, score: i64) -> Answer {
        Answer {
            p: vec![vec![(0, 0, 0, 0); n]; d],
            score,
        }
    }

    pub fn output(&self) {
        for vec in self.p.iter() {
            for p in vec.iter() {
                println!("{} {} {} {}", p.0, p.1, p.2, p.3);
            }
        }
        eprintln!(
            "result: {{\"score\": {}, \"duration\": {:.4}}}",
            self.score + 1,
            time::elapsed_seconds(),
        );
    }
}

pub struct Param {
    pub start_temp: f64,
    pub end_temp: f64,
    pub d_ratio: f64,
}
