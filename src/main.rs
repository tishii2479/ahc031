mod def;
mod solver;
mod util;

use crate::def::*;
use crate::solver::*;
use crate::util::*;

const FIRST_TIME_LIMIT: f64 = 1.;

fn best_fit(ws: &Vec<i64>, input: &Input) -> Vec<Vec<Vec<usize>>> {
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
    }
    r
}

fn eval_d(d: usize, ws: &Vec<i64>, r: &Vec<Vec<Vec<usize>>>, input: &Input) -> i64 {
    let mut score = 0;
    for (col, rs) in r[d].iter().enumerate() {
        let mut height = 0;
        for &i in rs {
            height += ceil_div(input.A[d][i], ws[col]);
        }
        if height > input.W {
            score += (height - input.W) * ws[col] * 1000;
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

fn create_col_and_initial_r(input: &Input) -> (Vec<i64>, Vec<Vec<Vec<usize>>>) {
    let mut ws = vec![300, 300, 200, 200];

    // 初期状態の作成
    let mut r = best_fit(&ws, input);

    let mut cur_score = eval(&ws, &r, input);
    let mut iteration = 0;
    eprintln!("cur_score    = {}", cur_score);

    // while time::elapsed_seconds() < FIRST_TIME_LIMIT {
    for _ in 0..10000000 {
        let p = rnd::nextf();
        if p < 0.5 {
            // 領域を移動する
            let d = rnd::gen_index(input.D);
            let col1 = rnd::gen_index(ws.len());
            let col2 = rnd::gen_index(ws.len());
            if col1 == col2 || r[d][col1].len() <= 1 {
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
            // 領域をスワップする
            let d = rnd::gen_index(input.D);
            let col1 = rnd::gen_index(ws.len());
            let col2 = rnd::gen_index(ws.len());
            if col1 == col2 || r[d][col1].len() == 0 || r[d][col2].len() == 0 {
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
                // eprintln!("sw_r:    {} -> {}", cur_score, new_score);
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
    eprintln!("ws:          {:?}", ws);
    eprintln!("elapsed:     {:?}", time::elapsed_seconds());

    for d in 0..input.D {
        let mut height = vec![0; ws.len()];
        for col in 0..ws.len() {
            for &r_idx in r[d][col].iter() {
                height[col] += ceil_div(input.A[d][r_idx], ws[col]);
            }
        }
        eprintln!("height:  {:?}", height);
    }

    (ws, r)
}

fn main() {
    time::start_clock();
    let input = Input::read_input();
    let (ws, r) = create_col_and_initial_r(&input);
    let mut solver = Solver::new(ws.clone(), r.clone(), &input);
    let ans = solver.solve();
    ans.output();
}
