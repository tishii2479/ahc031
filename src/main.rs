mod util;

use crate::util::*;
use proconio::input;

#[allow(non_snake_case)]
struct Input {
    W: usize,
    D: usize,
    N: usize,
    A: Vec<Vec<usize>>,
}

impl Input {
    #[allow(non_snake_case)]
    fn read_input() -> Input {
        input! {
            W: usize, D: usize, N: usize,
            A: [[usize; N]; D]
        }
        Input { W, D, N, A }
    }
}

struct Answer {
    p: Vec<Vec<(usize, usize, usize, usize)>>,
}

impl Answer {
    fn new(d: usize, n: usize) -> Answer {
        Answer {
            p: vec![vec![(0, 0, 0, 0); n]; d],
        }
    }

    fn output(&self) {
        for vec in self.p.iter() {
            for p in vec.iter() {
                println!("{} {} {} {}", p.0, p.1, p.2, p.3);
            }
        }
    }
}

fn create_bins(input: &Input) -> Vec<usize> {
    vec![0, 400, 700, 800, 900, 950, input.W]
}

fn solve(input: &Input) -> Answer {
    let bins = create_bins(input);
    let mut ans = Answer::new(input.D, input.N);

    for d in 0..input.D {
        let mut height = vec![0; bins.len() - 1];
        for i in (0..input.N).rev() {
            let mut best_remain = 1 << 30;
            let mut best_j = !0;
            for j in 0..bins.len() - 1 {
                if height[j] >= input.W {
                    continue;
                }
                let w = bins[j + 1] - bins[j];
                let remain = w * (input.W - height[j]);
                if input.A[d][i] <= remain && remain < best_remain {
                    best_remain = remain;
                    best_j = j;
                }
            }

            assert!(best_j < bins.len() - 1, "{} {} {:?}", d, i, height);
            let w = bins[best_j + 1] - bins[best_j];
            let h = (input.A[d][i] + w - 1) / w;
            ans.p[d][i] = (
                height[best_j],
                bins[best_j],
                height[best_j] + h,
                bins[best_j + 1],
            );
            height[best_j] += h;
        }
    }

    ans
}

fn solve2(input: &Input) -> Answer {
    let bins = create_bins(input);
    let mut slides = vec![vec![0, input.W]; bins.len() - 1];
    let mut ans = Answer::new(input.D, input.N);

    let mut rect_count = bins.len() - 1;
    while rect_count < input.N {
        let i = rnd::gen_index(bins.len() - 1);
        let y = rnd::gen_range(1, input.W);
        match slides[i].binary_search(&y) {
            Ok(_) => {}
            Err(pos) => {
                slides[i].insert(pos, y);
                rect_count += 1;
            }
        }
    }

    for d in 0..input.D {
        let mut v = Vec::with_capacity(input.N);
        for i in 0..bins.len() - 1 {
            let w = bins[i + 1] - bins[i];
            for j in 0..slides[i].len() - 1 {
                let h = slides[i][j + 1] - slides[i][j];
                v.push((
                    w * h,
                    (slides[i][j], bins[i], slides[i][j + 1], bins[i + 1]),
                ));
            }
        }
        v.sort();
        for i in 0..input.N {
            ans.p[d][i] = v[i].1;
        }
    }

    ans
}

fn main() {
    let input = Input::read_input();
    let ans = solve2(&input);
    ans.output();
}
