mod def;
mod solver;
mod util;

use crate::def::*;
use crate::solver::*;
use crate::util::*;

const FIRST_TIME_LIMIT: f64 = 0.5;

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

#[inline]
fn eval_height(w: i64, height: i64, max_height: i64, rs_count: usize) -> i64 {
    if height > max_height {
        (height - max_height) * w * 1_000
    } else if rs_count > 0 {
        // 含まれていない列はスコアに加算しない
        -(max_height - height) * w
    } else {
        1_000_000_000
    }
}

fn optimize_initial_r(ws: &Vec<i64>, input: &Input) -> (Vec<Vec<Vec<usize>>>, i64) {
    // 初期状態の作成
    let mut r = best_fit(&ws, input);
    let mut heights = vec![vec![0; ws.len()]; input.D];
    // TODO: let mut ceil_height = vec![vec![vec![0; ws.len()]; input.N]; input.D];

    let mut cur_score = 0;
    for d in 0..input.D {
        for (col, rs) in r[d].iter().enumerate() {
            heights[d][col] += rs
                .iter()
                .map(|&i| ceil_div(input.A[d][i], ws[col]))
                .sum::<i64>();
            cur_score += eval_height(ws[col], heights[d][col], input.W, rs.len());
        }
    }

    for _ in 0..30_000 {
        let d = rnd::gen_index(input.D);
        let (col1, col2) = (rnd::gen_index(ws.len()), rnd::gen_index(ws.len()));
        if col1 == col2 {
            continue;
        }

        // n:nスワップ
        let swap_count = rnd::gen_range(1, 4);
        let mut i1 = vec![];
        let mut i2 = vec![];
        let mut s1 = 0;
        let mut s2 = 0;
        for _ in 0..swap_count {
            if s1 <= s2 && i1.len() < r[d][col1].len() {
                let i = rnd::gen_index(r[d][col1].len());
                if i1.contains(&i) {
                    continue;
                }
                s1 += input.A[d][r[d][col1][i]];
                i1.push(i);
            } else if s2 <= s1 && i2.len() < r[d][col2].len() {
                let i = rnd::gen_index(r[d][col2].len());
                if i2.contains(&i) {
                    continue;
                }
                s2 += input.A[d][r[d][col2][i]];
                i2.push(i);
            }
        }
        // どちらかの列がなくなってしまうなら棄却
        if (r[d][col1].len() + i2.len() - i1.len() == 0)
            || (r[d][col2].len() + i1.len() - i2.len() == 0)
        {
            continue;
        }
        let cur_eval_col = eval_height(ws[col1], heights[d][col1], input.W, r[d][col1].len())
            + eval_height(ws[col2], heights[d][col2], input.W, r[d][col2].len());
        i1.sort_by(|a, b| b.cmp(a));
        i2.sort_by(|a, b| b.cmp(a));
        let rs1 = i1
            .iter()
            .map(|&i| r[d][col1].swap_remove(i))
            .collect::<Vec<usize>>();
        let rs2 = i2
            .iter()
            .map(|&i| r[d][col2].swap_remove(i))
            .collect::<Vec<usize>>();
        for &r1 in rs1.iter() {
            r[d][col2].push(r1);
            heights[d][col1] -= ceil_div(input.A[d][r1], ws[col1]);
            heights[d][col2] += ceil_div(input.A[d][r1], ws[col2]);
        }
        for &r2 in rs2.iter() {
            r[d][col1].push(r2);
            heights[d][col2] -= ceil_div(input.A[d][r2], ws[col2]);
            heights[d][col1] += ceil_div(input.A[d][r2], ws[col1]);
        }
        let new_eval_col = eval_height(ws[col1], heights[d][col1], input.W, r[d][col1].len())
            + eval_height(ws[col2], heights[d][col2], input.W, r[d][col2].len());
        let new_score = cur_score + new_eval_col - cur_eval_col;
        if new_score < cur_score {
            // eprintln!("[{:5}] sw_r: {} -> {}", t, cur_score, new_score);
            // eprintln!("ws:      {:?}", ws);
            cur_score = new_score;
        } else {
            for _ in 0..rs1.len() {
                r[d][col2].pop();
            }
            for _ in 0..rs2.len() {
                r[d][col1].pop();
            }
            // ロールバック
            for &r1 in rs1.iter() {
                r[d][col1].push(r1);
                heights[d][col1] += ceil_div(input.A[d][r1], ws[col1]);
                heights[d][col2] -= ceil_div(input.A[d][r1], ws[col2]);
            }
            for &r2 in rs2.iter() {
                r[d][col2].push(r2);
                heights[d][col1] -= ceil_div(input.A[d][r2], ws[col1]);
                heights[d][col2] += ceil_div(input.A[d][r2], ws[col2]);
            }
        }
    }

    (r, cur_score)
}

fn main() {
    time::start_clock();
    let input = Input::read_input();

    let mut start_cands = vec![];
    while time::elapsed_seconds() < FIRST_TIME_LIMIT {
        // TODO: bin_count=1は一回しかやらない
        let bin_count = rnd::gen_range(1, input.N.min(14));
        let mut bins = (0..bin_count - 1)
            .map(|_| rnd::gen_range(1, input.W as usize) as i64)
            .collect::<Vec<i64>>();
        bins.push(0);
        bins.push(input.W);
        bins.sort();
        let mut ws = (0..bins.len() - 1)
            .map(|i| bins[i + 1] - bins[i])
            .filter(|&x| x > 0)
            .collect::<Vec<i64>>();
        ws.sort();
        let (r, score) = optimize_initial_r(&ws, &input);
        eprintln!("score: {} {:?}", score, ws);

        start_cands.push((score, ws, r));
    }
    eprintln!("cand_count: {}", start_cands.len());

    start_cands.sort();
    let (_, ws, r) = start_cands[0].clone();
    eprintln!("ws({}): {:?}", ws.len(), ws);

    let mut solver = Solver::new(ws, r, &input);
    let ans = solver.solve();
    ans.output();
}
