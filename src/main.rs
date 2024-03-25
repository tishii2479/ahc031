mod util;

use crate::util::*;
use proconio::input;

const TIME_LIMIT: f64 = 2.9;

#[allow(non_snake_case)]
struct Input {
    W: i64,
    D: usize,
    N: usize,
    A: Vec<Vec<i64>>,
}

impl Input {
    #[allow(non_snake_case)]
    fn read_input() -> Input {
        input! {
            W: i64, D: usize, N: usize,
            A: [[i64; N]; D]
        }
        Input { W, D, N, A }
    }
}

struct Answer {
    p: Vec<Vec<(i64, i64, i64, i64)>>,
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

fn create_bins(input: &Input) -> Vec<i64> {
    let mut ws = vec![200; 5];
    let mut r = vec![vec![vec![]; ws.len()]; input.D];
    for d in 0..input.D {
        let mut height = vec![0; ws.len()];
        for i in (0..input.N).rev() {
            let a = input.A[d][i];
            let mut best_j = !0;
            let mut best_rem = 1 << 30;
            for (j, &w) in ws.iter().enumerate() {
                let h = ceil_div(a, w);
                let rem = (input.W - height[j] - h) * w;
                if rem >= 0 && rem < best_rem {
                    best_j = j;
                    best_rem = rem;
                }
            }
            if best_j < ws.len() {
                // どこかに入る場合は最も余裕が少なくなる（ギリギリ入る）列に入れる
                r[d][best_j].push(i);
                height[best_j] += ceil_div(a, ws[best_j]);
            } else {
                // 最もはみ出す高さが低いところに入れる
                let mut best_j = !0;
                let mut best_h = 1 << 30;
                for (j, &w) in ws.iter().enumerate() {
                    let h = ceil_div(a, w) + height[j];
                    if h < best_h {
                        best_j = j;
                        best_h = h;
                    }
                }
                r[d][best_j].push(i);
                height[best_j] += ceil_div(a, ws[best_j]);
            }
        }
        eprintln!("height:  {:?}", height);
    }

    fn eval_d(d: usize, ws: &Vec<i64>, r: &Vec<Vec<Vec<usize>>>, input: &Input) -> i64 {
        let mut score = 0;
        for (col, rs) in r[d].iter().enumerate() {
            let mut height = 0;
            for &e in rs {
                height += ceil_div(input.A[d][e], ws[col]);
            }
            if height > input.W {
                score += (height - input.W) * ws[col] * 100;
            } else if rs.len() > 0 {
                // 含まれていない列はスコアに加算しない
                score -= (input.W - height) * ws[col];
            }
        }
        score
    }

    fn eval(ws: &Vec<i64>, r: &Vec<Vec<Vec<usize>>>, input: &Input) -> i64 {
        let mut score = 0;
        for d in 0..input.D {
            score += eval_d(d, ws, &r, input);
        }
        score
    }

    let mut cur_score = eval(&ws, &r, input);
    let mut iteration = 0;
    eprintln!("cur_score    = {}", cur_score);

    while time::elapsed_seconds() < TIME_LIMIT {
        let p = rnd::nextf();
        if p < 0.2 {
            // 列iを列jにマージする
            let (i, j) = (rnd::gen_index(ws.len()), rnd::gen_index(ws.len()));
            if i == j {
                continue;
            }
            let _ws = ws.clone();
            let _r = r.clone();
            ws[j] += ws[i];
            ws.remove(i);
            for d in 0..input.D {
                let rs = r[d].remove(i);
                let j = if j < i { j } else { j - 1 }; // NOTE: j>=iの時はindexがずれている
                r[d][j].extend(rs); // NOTE: マージソートすれば昇順が保たれる
            }
            let new_score = eval(&ws, &r, input);
            if new_score < cur_score {
                eprintln!("merge:   {} -> {}", cur_score, new_score);
                eprintln!("ws:      {:?}", ws);
                cur_score = new_score;
            } else {
                // ロールバック
                ws = _ws;
                r = _r;
            }
        } else if p < 0.4 {
            // 列iを2つの列に分割する
            let i = rnd::gen_index(ws.len());

            // 適当に分割する
            // 比率をp:1-pになるように分ける
            // NOTE: 比率を変えるとどうなる？
            let ratio = rnd::gen_rangef(0.01, 0.99);
            let a_w = (ws[i] as f64 * ratio).round() as i64;
            let b_w = ws[i] - a_w;
            if a_w < 1 || b_w < 1 {
                continue;
            }

            let _ws = ws.clone();
            let _r = r.clone();
            for d in 0..input.D {
                let rs = r[d].remove(i);
                let mut a = vec![];
                let mut b = vec![];
                let mut a_h = 0;
                let mut b_h = 0;
                for i in rs {
                    let s = input.A[d][i];
                    // 高さが低い方に入れる
                    let next_a_h = a_h + ceil_div(s, a_w);
                    let next_b_h = b_h + ceil_div(s, b_w);
                    if next_a_h < next_b_h {
                        a.push(i);
                        a_h = next_a_h;
                    } else {
                        b.push(i);
                        b_h = next_b_h;
                    }
                }
                r[d].push(a);
                r[d].push(b);
            }
            ws.remove(i);
            ws.push(a_w);
            ws.push(b_w);

            let new_score = eval(&ws, &r, input);
            if new_score < cur_score {
                eprintln!("split:   {} -> {}", cur_score, new_score);
                eprintln!("ws:      {:?}", ws);
                cur_score = new_score;
            } else {
                // ロールバック
                ws = _ws;
                r = _r;
            }
        } else if p < 0.6 {
            // 列iの幅を列jに移す
            let (i, j) = (rnd::gen_index(ws.len()), rnd::gen_index(ws.len()));
            if i == j {
                continue;
            }
            let ratio = rnd::gen_rangef(0.5, 0.99);
            let mv_w = (ws[i] as f64 * ratio).round() as i64;
            if ws[i] - mv_w < 1 {
                continue;
            }
            let _ws = ws.clone();
            let _r = r.clone();
            ws[i] -= mv_w;
            ws[j] += mv_w;
            let new_score = eval(&ws, &r, input);
            if new_score < cur_score {
                eprintln!("move:    {} -> {}", cur_score, new_score);
                eprintln!("ws:      {:?}", ws);
                cur_score = new_score;
            } else {
                // ロールバック
                ws = _ws;
                r = _r;
            }
        } else if p < 0.8 {
            // 領域を移動する
            let d = rnd::gen_index(input.D);
            let col1 = rnd::gen_index(ws.len());
            let col2 = rnd::gen_index(ws.len());
            if col1 == col2 {
                continue;
            }
            let cur_eval_d = eval_d(d, &ws, &r, input); // NOTE: eval_colにすればもっと早い
            let i = rnd::gen_index(r[d][col1].len());
            let r1 = r[d][col1].remove(i);
            r[d][col2].push(r1);
            let new_score = cur_score + eval_d(d, &ws, &r, input) - cur_eval_d;
            if new_score < cur_score {
                // eprintln!("mv_r:    {} -> {}", cur_score, new_score);
                // eprintln!("ws:      {:?}", ws);
                cur_score = new_score;
            } else {
                // ロールバック
                r[d][col2].pop();
                r[d][col1].push(r1);
            }
        } else {
            // 領域を移動する
            let d = rnd::gen_index(input.D);
            let col1 = rnd::gen_index(ws.len());
            let col2 = rnd::gen_index(ws.len());
            if col1 == col2 {
                continue;
            }
            let cur_eval_d = eval_d(d, &ws, &r, input); // NOTE: eval_colにすればもっと早い
            let i1 = rnd::gen_index(r[d][col1].len());
            let i2 = rnd::gen_index(r[d][col2].len());
            let r1 = r[d][col1].remove(i1);
            let r2 = r[d][col2].remove(i2);
            r[d][col2].push(r1);
            r[d][col1].push(r2);
            let new_score = cur_score + eval_d(d, &ws, &r, input) - cur_eval_d;
            if new_score < cur_score {
                // eprintln!("mv_r:    {} -> {}", cur_score, new_score);
                // eprintln!("ws:      {:?}", ws);
                cur_score = new_score;
            } else {
                // ロールバック
                r[d][col1].pop();
                r[d][col2].pop();
                r[d][col1].push(r1);
                r[d][col2].push(r2);
            }
        }

        iteration += 1;
    }

    eprintln!("iteration:   {}", iteration);
    eprintln!("score:       {}", eval(&ws, &r, input));

    for d in 0..input.D {
        for (col, rs) in r[d].iter().enumerate() {
            let mut height = 0;
            for &e in rs {
                height += ceil_div(input.A[d][e], ws[col]);
            }
            eprint!("{:4}", height);
        }
        eprintln!();
    }

    ws.sort_by(|a, b| b.cmp(a));
    eprintln!("ws:      {:?}", ws);

    let mut bins = vec![0];
    for w in ws {
        bins.push(bins.last().unwrap() + w);
    }

    bins
}

fn solve(input: &Input) -> Answer {
    let ans = Answer::new(input.D, input.N);
    let bins = create_bins(input);
    eprintln!("{:?}", bins);

    ans
}

fn main() {
    time::start_clock();
    let input = Input::read_input();
    let ans = solve(&input);
    ans.output();
}
