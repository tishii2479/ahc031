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
        1_000_000
    }
}

fn optimize_initial_r(ws: &Vec<i64>, input: &Input) -> (Vec<Vec<Vec<usize>>>, i64, Vec<Vec<i64>>) {
    // 初期状態の作成
    let mut r = best_fit(&ws, input);
    let mut heights = vec![vec![0; ws.len()]; input.D];

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

    let mut i1 = Vec::with_capacity(3);
    let mut i2 = Vec::with_capacity(3);
    let mut r1 = Vec::with_capacity(3);
    let mut r2 = Vec::with_capacity(3);
    let iteration = input.N * input.D * 100; // :param (2_500..250_000)
    for _t in 0..iteration {
        let d = rnd::gen_index(input.D);
        let (col1, col2) = (rnd::gen_index(ws.len()), rnd::gen_index(ws.len()));
        if col1 == col2 {
            continue;
        }

        // n:nスワップ
        let swap_count = rnd::gen_range(1, 5); // :param
        i1.clear();
        i2.clear();
        r1.clear();
        r2.clear();
        let mut s1 = 0;
        let mut s2 = 0;
        for _ in 0..swap_count {
            if s1 <= s2 && i1.len() < r[d][col1].len() {
                let i = rnd::gen_index(r[d][col1].len());
                if i1.contains(&i) {
                    continue;
                }
                let r = r[d][col1][i];
                s1 += r;
                i1.push(i);
                r1.push(r);
            } else if s2 <= s1 && i2.len() < r[d][col2].len() {
                let i = rnd::gen_index(r[d][col2].len());
                if i2.contains(&i) {
                    continue;
                }
                let r = r[d][col2][i];
                s2 += r;
                i2.push(i);
                r2.push(r);
            }
        }
        // どちらかの列がなくなってしまうなら棄却
        let new_r_len1 = r[d][col1].len() + i2.len() - i1.len();
        let new_r_len2 = r[d][col2].len() + i1.len() - i2.len();
        if new_r_len1 == 0 || new_r_len2 == 0 {
            continue;
        }
        let cur_eval_col = eval_height(ws[col1], heights[d][col1], input.W, r[d][col1].len())
            + eval_height(ws[col2], heights[d][col2], input.W, r[d][col2].len());
        for &r in r1.iter() {
            heights[d][col1] -= ceil_div(input.A[d][r], ws[col1]);
            heights[d][col2] += ceil_div(input.A[d][r], ws[col2]);
        }
        for &r in r2.iter() {
            heights[d][col1] += ceil_div(input.A[d][r], ws[col1]);
            heights[d][col2] -= ceil_div(input.A[d][r], ws[col2]);
        }
        let new_eval_col = eval_height(ws[col1], heights[d][col1], input.W, new_r_len1)
            + eval_height(ws[col2], heights[d][col2], input.W, new_r_len2);
        let new_score = cur_score + new_eval_col - cur_eval_col;
        if new_score < cur_score {
            // eprintln!("[{:5}] sw_r: {} -> {}", _t, cur_score, new_score);
            i1.sort_by(|a, b| b.cmp(a));
            i2.sort_by(|a, b| b.cmp(a));
            for &i in i1.iter() {
                r[d][col1].swap_remove(i);
            }
            for &i in i2.iter() {
                r[d][col2].swap_remove(i);
            }
            for &r1 in r1.iter() {
                r[d][col2].push(r1);
            }
            for &r2 in r2.iter() {
                r[d][col1].push(r2);
            }
            cur_score = new_score;
        } else {
            for &r in r1.iter() {
                heights[d][col1] += ceil_div(input.A[d][r], ws[col1]);
                heights[d][col2] -= ceil_div(input.A[d][r], ws[col2]);
            }
            for &r in r2.iter() {
                heights[d][col1] -= ceil_div(input.A[d][r], ws[col1]);
                heights[d][col2] += ceil_div(input.A[d][r], ws[col2]);
            }
        }
    }

    (r, cur_score, heights)
}

fn optimize_start_cands(input: &Input) -> Vec<(i64, Vec<i64>, Vec<Vec<Vec<usize>>>)> {
    let mut start_cands = vec![];
    let mut max_bin_count = 1;

    while time::elapsed_seconds() < FIRST_TIME_LIMIT || start_cands.len() == 0 {
        let bin_count = rnd::gen_range(
            max_bin_count.max(3) - 2,
            (max_bin_count + 2).clamp(1, input.N) + 1,
        );
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

        // 一番小さい領域に入らないものがあれば棄却
        let mut is_valid = true;
        let min_region = ws[0] * input.W;
        for d in 0..input.D {
            if min_region < input.A[d][0] {
                is_valid = false;
            }
        }
        if !is_valid {
            continue;
        }

        let (r, score, heights) = optimize_initial_r(&ws, &input);

        let mut is_empty = false;
        for d in 0..input.D {
            for col in 0..ws.len() {
                if r[d][col].len() == 0 {
                    is_empty = true;
                }
            }
        }
        if is_empty {
            continue;
        }

        // eprintln!("score: {} {:?}", score, ws);
        start_cands.push((score, ws, r));

        let max_height = *heights
            .iter()
            .map(|v| v.iter().max().unwrap())
            .max()
            .unwrap();

        if max_height <= input.W {
            max_bin_count = max_bin_count.max(bin_count);
        }
    }

    eprintln!("cand_count:      {}", start_cands.len());
    eprintln!("max_bin_count:   {}", max_bin_count);
    start_cands.sort();

    start_cands
}

fn main() {
    time::start_clock();
    let input = Input::read_input();
    let start_cands = optimize_start_cands(&input);

    let (_, ws, r) = start_cands[0].clone();
    eprintln!("ws({}): {:?}", ws.len(), ws);

    let mut solver = Solver::new(ws, r, &input);
    let ans = solver.solve();
    ans.output();
}
