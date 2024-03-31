use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use crate::def::*;
use crate::util::*;

fn match_greedy(
    prev_h: &Vec<i64>,
    prev_rem: &Vec<Vec<(usize, i64)>>,
    next_h: &Vec<i64>,
    w: i64,
    groups: &mut Vec<((usize, usize), (usize, usize), i64)>,
) -> (usize, Vec<i64>) {
    let mut match_count = 0;
    let mut prev_pickup_rems = vec![0; prev_h.len()];

    let mut prev_rem_sum = 0;
    let mut next_rem_sum = w - next_h.iter().sum::<i64>();

    let mut prev_height = 0;
    let mut next_height = 0;
    let mut prev_l = 0;
    let mut next_l = 0;
    let mut prev_r = 0;
    let mut next_r = 0;

    groups.clear();

    while prev_r < prev_h.len() || next_r < next_h.len() {
        if (prev_r < prev_h.len() && prev_height <= next_height) || next_r == next_h.len() {
            // 使えるようになる余裕を回収する
            for &(r, rem) in prev_rem[prev_r].iter() {
                prev_pickup_rems[r] += rem;
                prev_rem_sum += rem;
            }

            // 回収する
            // 最終的にはprev_pickup_rem[:, prev_h.len() - 1]しか残らない
            if prev_r < prev_h.len() - 1 {
                prev_height += prev_pickup_rems[prev_r];
                prev_rem_sum -= prev_pickup_rems[prev_r];
                prev_pickup_rems[prev_r] = 0;
            }

            prev_height += prev_h[prev_r];
            prev_r += 1;
        } else {
            next_height += next_h[next_r];
            next_r += 1;
        }

        // 片方が進んでいなかったら、次に進める
        if prev_l == prev_r || next_l == next_r {
            continue;
        }
        // 片方が最後まで来ていたら、両方最後まで進める
        if (prev_r == prev_h.len() && next_r < next_h.len())
            || (prev_r < prev_h.len() && next_r == next_h.len())
        {
            continue;
        }

        if prev_height <= next_height && next_height <= prev_height + prev_rem_sum {
            let mut use_prev_rem = next_height - prev_height;
            prev_rem_sum -= use_prev_rem;
            for i in prev_r - 1..prev_h.len() {
                let r = prev_pickup_rems[i].min(use_prev_rem);
                prev_pickup_rems[i] -= r;
                use_prev_rem -= r;
            }
            groups.push(((prev_l, prev_r), (next_l, next_r), 0));
            prev_height = next_height;
            prev_l = prev_r;
            next_l = next_r;
            match_count += 1;
        } else if next_height <= prev_height && prev_height <= next_height + next_rem_sum {
            // nextを合わせられるなら合わせる
            let use_next_rem = prev_height - next_height;
            next_rem_sum -= use_next_rem;

            groups.push(((prev_l, prev_r), (next_l, next_r), use_next_rem));
            next_height = prev_height;
            prev_l = prev_r;
            next_l = next_r;
            match_count += 1;
        }
    }

    (match_count, prev_pickup_rems)
}

/// match_greedyで、vec![]を消したバージョン
/// 切り替えを減らした回数だけを返す
fn match_greedy_fast(
    prev_h: &Vec<i64>,
    prev_rem: &Vec<Vec<(usize, i64)>>,
    next_h: &Vec<i64>,
    next_height_sum: i64,
    w: i64,
    prev_pickup_rems: &mut Vec<i64>,
) -> usize {
    if prev_h.len() == 1 || next_h.len() == 1 {
        return 1;
    }
    let mut match_count = 0;
    for i in 0..prev_h.len() {
        prev_pickup_rems[i] = 0;
    }

    let mut prev_rem_sum = 0;
    let mut next_rem_sum = w - next_height_sum;

    let mut prev_height = 0;
    let mut next_height = 0;
    let mut prev_l = 0;
    let mut next_l = 0;
    let mut prev_r = 0;
    let mut next_r = 0;
    while prev_r < prev_h.len() || next_r < next_h.len() {
        if (prev_r < prev_h.len() && prev_height <= next_height) || next_r == next_h.len() {
            // 使えるようになる余裕を回収する
            for &(r, rem) in prev_rem[prev_r].iter() {
                prev_pickup_rems[r] += rem;
                prev_rem_sum += rem;
            }

            // 回収する
            // 最終的にはprev_pickup_rem[:, prev_h.len() - 1]しか残らない
            if prev_r < prev_h.len() - 1 {
                prev_height += prev_pickup_rems[prev_r];
                prev_rem_sum -= prev_pickup_rems[prev_r];
                prev_pickup_rems[prev_r] = 0;
            }

            prev_height += prev_h[prev_r];
            prev_r += 1;
        } else {
            next_height += next_h[next_r];
            next_r += 1;
        }

        // 片方が進んでいなかったら、次に進める
        // 片方が最後まで来ていたら、両方最後まで進める

        if prev_l == prev_r
            || next_l == next_r
            || (prev_r == prev_h.len() && next_r < next_h.len())
            || (prev_r < prev_h.len() && next_r == next_h.len())
        {
            continue;
        }

        if prev_height <= next_height && next_height <= prev_height + prev_rem_sum {
            let mut use_prev_rem = next_height - prev_height;
            prev_rem_sum -= use_prev_rem;
            for i in prev_r - 1..prev_h.len() {
                let r = prev_pickup_rems[i].min(use_prev_rem);
                prev_pickup_rems[i] -= r;
                use_prev_rem -= r;
            }
            prev_height = next_height;
            prev_l = prev_r;
            next_l = next_r;
            match_count += 1;
        } else if next_height <= prev_height && prev_height <= next_height + next_rem_sum {
            // nextを合わせられるなら合わせる
            let use_next_rem = prev_height - next_height;
            next_rem_sum -= use_next_rem;

            next_height = prev_height;
            prev_l = prev_r;
            next_l = next_r;
            match_count += 1;
        }
    }

    match_count
}

fn to_squeezed_height(
    heights: &mut Vec<i64>,
    base_heights: &Vec<i64>,
    height_sum: i64,
    r_idx: &Vec<usize>,
    d: usize,
    w: i64,
    input: &Input,
    space_for_r_idx: &mut Vec<(i64, usize)>,
) -> i64 {
    // 超過してしまう場合は、縮めることで損をしない領域（余りが大きい領域）を縮める
    let mut over_height = height_sum - input.W;
    let mut exceed_s = 0;
    heights.clear();
    for &h in base_heights.iter() {
        heights.push(h);
    }
    if over_height > 0 {
        space_for_r_idx.clear();
        for i in 0..heights.len() {
            let over_s = input.A[d][r_idx[i]] % w;
            space_for_r_idx.push((if over_s == 0 { w } else { over_s }, i));
        }
        space_for_r_idx.sort_unstable();
        while over_height > 0 {
            let (space, idx) = space_for_r_idx.remove(0);
            if heights[idx] > 1 {
                let sub = if space == w {
                    over_height.min(heights[idx] - 1)
                } else {
                    1
                };
                exceed_s += space * sub;
                heights[idx] -= sub;
                over_height -= sub;
            }
            space_for_r_idx.push((w, idx));
        }
    }

    exceed_s * EXCEED_COST
}

pub struct Solver<'a> {
    state: State,
    input: &'a Input,
}

impl<'a> Solver<'a> {
    pub fn new(ws: Vec<i64>, r: Vec<Vec<Vec<usize>>>, input: &'a Input) -> Solver {
        Solver {
            state: State::new(ws, r, input.W, input.D),
            input,
        }
    }

    pub fn solve(&mut self, time_limit: f64, param: &Param) -> Answer {
        self.state.setup_heights(self.input);

        let mut total_cost = 0;
        for d in 0..self.input.D {
            self.state.setup_prev_h_rem(d);
            if d != self.input.D - 1 {
                self.optimize_r(d, time_limit, param);
            }

            total_cost += self.state.to_next_d(d, self.input);
        }

        self.create_answer(total_cost)
    }

    /// d日目とd+1日目を同時に最適化する
    fn optimize_r(&mut self, d: usize, time_limit: f64, param: &Param) {
        self.state.setup_score(d, self.input);

        let is_last = d == self.input.D - 2;
        let start_time = time::elapsed_seconds();
        let duration = ((time_limit - start_time) / (self.input.D - d) as f64).max(1e-3)
            * if is_last { 2. } else { 1. };
        let end_time = start_time + duration;

        let mut iteration = 0;
        let mut adapt_in_swap = 0;
        let mut adapt_tr_move = 0;
        let mut adapt_tr_swap = 0;

        let start_temp = param.start_temp;
        let end_temp = param.end_temp;
        let mut action_ratio = vec![0.1, 0.4, 0.5]; // TODO: act_dごとに変える
        for i in 0..action_ratio.len() - 1 {
            action_ratio[i + 1] += action_ratio[i];
        }

        let mut swaps = Vec::with_capacity(5);
        let mut rs1 = Vec::with_capacity(5);
        let mut rs2 = Vec::with_capacity(5);

        // eprintln!("start_cur_score: {}", cur_score);
        while time::elapsed_seconds() < end_time {
            let act_d = d + (rnd::next() & 1);
            let progress = (time::elapsed_seconds() - start_time) / duration;
            let cur_temp = start_temp.powf(1. - progress) * end_temp.powf(progress);
            let threshold = -cur_temp * rnd::nextf().ln();
            let p = rnd::nextf();

            if p < action_ratio[0] {
                // 列内n回swap
                let col = rnd::gen_index(self.state.ws.len());
                if self.state.r[act_d][col].len() == 1 {
                    continue;
                }
                let swap_count = rnd::gen_range(1, self.state.r[act_d][col].len().clamp(1, 2) + 1);
                swaps.clear();
                for _ in 0..swap_count {
                    let (i, j) = (
                        rnd::gen_index(self.state.r[act_d][col].len()),
                        rnd::gen_index(self.state.r[act_d][col].len()),
                    );
                    if i != j {
                        swaps.push((i, j));
                    }
                }
                if swaps.len() == 0 {
                    continue;
                }

                for &(i, j) in swaps.iter() {
                    self.state.swap_r(act_d, col, i, j);
                }

                let new_score_col = self.state.eval_col(d, col, self.input, false);
                let score_diff = self.state.get_score_col_diff(col, new_score_col);
                if (score_diff as f64) <= threshold {
                    self.state.score_col[col] = new_score_col;
                    self.state.score += score_diff;
                    adapt_in_swap += 1;
                } else {
                    for &(i, j) in swaps.iter().rev() {
                        self.state.swap_r(act_d, col, i, j);
                    }
                }
            } else if p < action_ratio[1] {
                // 列間1個移動
                let col1 = rnd::gen_index(self.state.ws.len());
                if self.state.r[act_d][col1].len() <= 1 {
                    continue;
                }
                let col2 = rnd::gen_index(self.state.ws.len());
                if col1 == col2 {
                    continue;
                }
                let i1 = rnd::gen_index(self.state.r[act_d][col1].len());
                let i2 = rnd::gen_index(self.state.r[act_d][col2].len() + 1);

                let current_height_sum = self.state.height_sum[act_d][col2];
                self.state.move_r(act_d, (col1, i1), (col2, i2), self.input);
                let new_height_sum = self.state.height_sum[act_d][col2];

                // 高さが超過していて、さらに超過するなら棄却する
                if current_height_sum > self.input.W && new_height_sum > current_height_sum {
                    self.state.move_r(act_d, (col2, i2), (col1, i1), self.input);
                    continue;
                }

                let new_score_col1 = self.state.eval_col(d, col1, self.input, false);
                let new_score_col2 = self.state.eval_col(d, col2, self.input, false);
                let score_diff = self.state.get_score_col_diff(col1, new_score_col1)
                    + self.state.get_score_col_diff(col2, new_score_col2);

                if (score_diff as f64) <= threshold {
                    self.state.score_col[col1] = new_score_col1;
                    self.state.score_col[col2] = new_score_col2;
                    self.state.score += score_diff;
                    adapt_tr_move += 1;
                } else {
                    self.state.move_r(act_d, (col2, i2), (col1, i1), self.input);
                }
            } else {
                // 列間n:nスワップ
                let move_count = rnd::gen_range(2, 6); // :param
                let col1 = rnd::gen_index(self.state.ws.len());
                let col2 = rnd::gen_index(self.state.ws.len());
                if col1 == col2 {
                    continue;
                }
                let i1 = rnd::gen_index(self.state.r[act_d][col1].len());
                let i2 = rnd::gen_index(self.state.r[act_d][col2].len());
                let mut cnt1 = 0;
                let mut cnt2 = 0;
                let mut h1 = 0;
                let mut h2 = 0;

                // 低い方に足す
                for _ in 0..move_count {
                    if h1 <= h2 && i1 + cnt1 < self.state.r[act_d][col1].len() {
                        h1 += self.input.A[act_d][self.state.r[act_d][col1][i1 + cnt1]];
                        cnt1 += 1;
                    } else if h2 <= h1 && i2 + cnt2 < self.state.r[act_d][col2].len() {
                        h2 += self.input.A[act_d][self.state.r[act_d][col2][i2 + cnt2]];
                        cnt2 += 1;
                    }
                }
                rs1.clear();
                rs2.clear();

                let current_height_sum1 = self.state.height_sum[act_d][col1];
                let current_height_sum2 = self.state.height_sum[act_d][col2];
                for _ in 0..cnt1 {
                    rs1.push(self.state.remove_r(act_d, col1, i1));
                }
                for _ in 0..cnt2 {
                    rs2.push(self.state.remove_r(act_d, col2, i2));
                }
                for &r1 in rs1.iter().rev() {
                    self.state.insert_r(act_d, col2, i2, r1, self.input);
                }
                for &r2 in rs2.iter().rev() {
                    self.state.insert_r(act_d, col1, i1, r2, self.input);
                }
                let new_height_sum1 = self.state.height_sum[act_d][col1];
                let new_height_sum2 = self.state.height_sum[act_d][col2];

                // 高さが超過していて、さらに超過するなら棄却する
                if (current_height_sum1 > self.input.W && new_height_sum1 > current_height_sum1)
                    || (current_height_sum2 > self.input.W && new_height_sum2 > current_height_sum2)
                {
                    for _ in 0..cnt2 {
                        self.state.remove_r(act_d, col1, i1);
                    }
                    for _ in 0..cnt1 {
                        self.state.remove_r(act_d, col2, i2);
                    }
                    for &r1 in rs1.iter().rev() {
                        self.state.insert_r(act_d, col1, i1, r1, self.input);
                    }
                    for &r2 in rs2.iter().rev() {
                        self.state.insert_r(act_d, col2, i2, r2, self.input);
                    }
                    continue;
                }
                let new_score_col1 = self.state.eval_col(d, col1, self.input, false);
                let new_score_col2 = self.state.eval_col(d, col2, self.input, false);
                let score_diff = self.state.get_score_col_diff(col1, new_score_col1)
                    + self.state.get_score_col_diff(col2, new_score_col2);

                if (score_diff as f64) <= threshold {
                    self.state.score_col[col1] = new_score_col1;
                    self.state.score_col[col2] = new_score_col2;
                    self.state.score += score_diff;
                    adapt_tr_swap += 1;
                } else {
                    for _ in 0..cnt2 {
                        self.state.remove_r(act_d, col1, i1);
                    }
                    for _ in 0..cnt1 {
                        self.state.remove_r(act_d, col2, i2);
                    }
                    for &r1 in rs1.iter().rev() {
                        self.state.insert_r(act_d, col1, i1, r1, self.input);
                    }
                    for &r2 in rs2.iter().rev() {
                        self.state.insert_r(act_d, col2, i2, r2, self.input);
                    }
                }
            }
            iteration += 1;
        }

        eprintln!("score:       {:6}", self.state.score);
        eprintln!("duration:    {:.6}", duration);
        eprintln!("iteration:   {}", iteration);
        eprintln!("in-swap:     {}", adapt_in_swap);
        eprintln!("tr-move:     {}", adapt_tr_move);
        eprintln!("tr-swap:     {}", adapt_tr_swap);
        eprintln!("score:       {:?}", self.state.score_col);
    }

    fn create_answer(&mut self, total_cost: i64) -> Answer {
        let mut ans = Answer::new(self.input.D, self.input.N, total_cost);
        let mut width = 0;

        for col in 0..self.state.ws.len() {
            let w = self.state.ws[col];
            self.state.graphs[col].propagate_rem();

            for d in 0..self.input.D {
                let mut height = 0;
                for (i, &r_idx) in self.state.r[d][col].iter().enumerate() {
                    let node_idx = self.state.node_idx[d + 1][col][i];
                    let h = self.state.graphs[col].nodes[node_idx].height;
                    ans.p[d][r_idx] = (height, width, height + h, width + w);
                    height += h;
                }
            }
            width += w;
        }

        ans
    }
}

struct SharedVec {
    v: Vec<i64>,
    prev_rem: Vec<Vec<(usize, i64)>>,
    groups: Vec<((usize, usize), (usize, usize), i64)>,
    space_for_r_idx: Vec<(i64, usize)>,
}

struct State {
    score: i64,
    score_col: Vec<(i64, i64)>,
    ws: Vec<i64>,
    // r[d][col][i]
    r: Vec<Vec<Vec<usize>>>,
    // node_idx[d][col][i]
    node_idx: Vec<Vec<Vec<usize>>>,
    prev_h: Vec<Vec<i64>>,
    prev_rem: Vec<Vec<Vec<(usize, i64)>>>,
    heights: Vec<Vec<Vec<i64>>>,
    squeezed_heights: Vec<Vec<Vec<i64>>>,
    need_update_squeezed: Vec<Vec<bool>>,
    height_sum: Vec<Vec<i64>>,
    exceed_cost: Vec<Vec<i64>>,
    graphs: Vec<ColGraph>,
    shared: SharedVec,
}

impl State {
    fn new(ws: Vec<i64>, r: Vec<Vec<Vec<usize>>>, w: i64, d: usize) -> State {
        const MAX_N: usize = 50;
        let col_count = ws.len();
        let mut state = State {
            score: 0,
            score_col: vec![(0, 0); ws.len()],
            ws,
            r,
            prev_h: vec![],
            prev_rem: vec![],
            graphs: vec![ColGraph::new(w); col_count],
            node_idx: vec![vec![]; d + 1],
            heights: vec![vec![Vec::with_capacity(MAX_N * 2 / col_count); col_count]; d],
            squeezed_heights: vec![vec![Vec::with_capacity(MAX_N * 2 / col_count); col_count]; d],
            need_update_squeezed: vec![vec![false; col_count]; d],
            height_sum: vec![vec![0; col_count]; d],
            exceed_cost: vec![vec![0; col_count]; d],
            shared: SharedVec {
                v: vec![0; MAX_N],
                prev_rem: vec![vec![]; MAX_N],
                groups: Vec::with_capacity(MAX_N),
                space_for_r_idx: Vec::with_capacity(MAX_N),
            },
        };
        for col in 0..col_count {
            state.node_idx[0].push(vec![state.graphs[col].root_node]);
        }
        state
    }

    fn move_r(
        &mut self,
        d: usize,
        from: (usize, usize),
        to: (usize, usize),
        input: &Input,
    ) -> usize {
        let (from_col, from_idx) = from;
        let (to_col, to_idx) = to;
        let r_idx = self.remove_r(d, from_col, from_idx);
        self.insert_r(d, to_col, to_idx, r_idx, input);
        r_idx
    }

    fn remove_r(&mut self, d: usize, col: usize, i: usize) -> usize {
        let h = self.heights[d][col].remove(i);
        if i < self.squeezed_heights[d][col].len() {
            self.need_update_squeezed[d][col] = true;
        }
        self.height_sum[d][col] -= h;
        self.r[d][col].remove(i)
    }

    fn insert_r(&mut self, d: usize, col: usize, i: usize, r_idx: usize, input: &Input) {
        self.r[d][col].insert(i, r_idx);
        let h = ceil_div(input.A[d][r_idx], self.ws[col]);
        self.heights[d][col].insert(i, h);
        self.height_sum[d][col] += h;
        self.need_update_squeezed[d][col] = true;
    }

    fn swap_r(&mut self, d: usize, col: usize, i: usize, j: usize) {
        self.r[d][col].swap(i, j);
        self.heights[d][col].swap(i, j);
        if i < self.squeezed_heights[d][col].len() && j < self.squeezed_heights[d][col].len() {
            self.squeezed_heights[d][col].swap(i, j);
        }
    }

    fn eval_col(
        &mut self,
        d: usize,
        col: usize,
        input: &Input,
        force_allow_overflow: bool,
    ) -> (i64, i64) {
        // TODO: タイブレーク時には、余裕の残り具合も足す
        // TODO: 幅も考慮する
        // NOTE: 切り替え回数が変わらないなら、遷移した方が良い（スコアの差分はない方が良いのか？）

        // 現状超過していたら超過を許す
        let allow_overflow = force_allow_overflow || self.score_col[col].1 > 0;
        if !allow_overflow
            && (self.height_sum[d][col] > input.W || self.height_sum[d + 1][col] > input.W)
        {
            return (0, 1 << 40);
        }
        let exceed_cost1 = if self.height_sum[d][col] <= input.W {
            0
        } else if !self.need_update_squeezed[d][col] {
            self.exceed_cost[d][col]
        } else {
            self.exceed_cost[d][col] = to_squeezed_height(
                &mut self.squeezed_heights[d][col],
                &mut self.heights[d][col],
                self.height_sum[d][col],
                &self.r[d][col],
                d,
                self.ws[col],
                input,
                &mut self.shared.space_for_r_idx,
            );
            self.exceed_cost[d][col]
        };
        let exceed_cost2 = if self.height_sum[d + 1][col] <= input.W {
            0
        } else if !self.need_update_squeezed[d + 1][col] {
            self.exceed_cost[d + 1][col]
        } else {
            self.exceed_cost[d + 1][col] = to_squeezed_height(
                &mut self.squeezed_heights[d + 1][col],
                &mut self.heights[d + 1][col],
                self.height_sum[d + 1][col],
                &self.r[d + 1][col],
                d + 1,
                self.ws[col],
                input,
                &mut self.shared.space_for_r_idx,
            );
            self.exceed_cost[d + 1][col]
        };
        let next_h = if self.height_sum[d][col] <= input.W {
            &self.heights[d][col]
        } else {
            &self.squeezed_heights[d][col]
        };
        let next_next_h = if self.height_sum[d + 1][col] <= input.W {
            &self.heights[d + 1][col]
        } else {
            &self.squeezed_heights[d + 1][col]
        };
        let (match_count1, prev_pickup_rem1) = match_greedy(
            &self.prev_h[col],
            &self.prev_rem[col],
            &next_h,
            input.W,
            &mut self.shared.groups,
        );
        let switch_count1 = self.prev_h[col].len() + self.heights[d][col].len() - match_count1 * 2;

        for i in 0..self.heights[d][col].len() {
            self.shared.prev_rem[i].clear();
        }
        for &(_, (l, r), rem) in self.shared.groups.iter() {
            self.shared.prev_rem[l].push((r - 1, rem));
        }
        self.shared.prev_rem[0].push((
            self.heights[d][col].len() - 1,
            *prev_pickup_rem1.last().unwrap(),
        ));
        let match_count2 = match_greedy_fast(
            &next_h,
            &self.shared.prev_rem,
            &next_next_h,
            self.height_sum[d + 1][col],
            input.W,
            &mut self.shared.v,
        );
        let switch_count2 =
            self.heights[d][col].len() + self.heights[d + 1][col].len() - match_count2 * 2;

        // TODO: 調整
        let (w1, w2) = if d == 0 {
            (0, 2)
        } else if d == input.D - 2 {
            (2, 2)
        } else {
            (2, 1)
        };
        (
            ((switch_count1 * w1 + switch_count2 * w2) as i64 * self.ws[col]) / 2,
            exceed_cost1 + exceed_cost2,
        )
    }

    fn get_score_col_diff(&self, col: usize, new_score_col: (i64, i64)) -> i64 {
        new_score_col.0 + new_score_col.1 - self.score_col[col].0 - self.score_col[col].1
    }

    fn setup_score(&mut self, d: usize, input: &Input) {
        self.score_col = (0..self.ws.len())
            .map(|col| self.eval_col(d, col, input, true))
            .collect();
        self.score = self.score_col.iter().map(|x| x.0 + x.1).sum::<i64>();
    }

    fn setup_prev_h_rem(&mut self, d: usize) {
        // 事前計算
        let prev_h: Vec<Vec<i64>> = (0..self.ws.len())
            .map(|col| {
                self.node_idx[d][col]
                    .iter()
                    .map(|&i| self.graphs[col].nodes[i].height)
                    .collect()
            })
            .collect();
        let prev_rem: Vec<Vec<Vec<(usize, i64)>>> = (0..self.ws.len())
            .map(|col| self.graphs[col].gen_rem(&self.node_idx[d][col]))
            .collect();
        self.prev_h = prev_h;
        self.prev_rem = prev_rem;
    }

    fn setup_heights(&mut self, input: &Input) {
        for d in 0..input.D {
            for col in 0..self.ws.len() {
                for &r_idx in self.r[d][col].iter() {
                    let h = ceil_div(input.A[d][r_idx], self.ws[col]);
                    self.height_sum[d][col] += h;
                    self.heights[d][col].push(h);
                }
                if self.height_sum[d][col] > input.W {
                    self.exceed_cost[d][col] = to_squeezed_height(
                        &mut self.squeezed_heights[d][col],
                        &mut self.heights[d][col],
                        self.height_sum[d][col],
                        &self.r[d][col],
                        d,
                        self.ws[col],
                        input,
                        &mut self.shared.space_for_r_idx,
                    );
                }
            }
        }
    }

    fn to_next_d(&mut self, d: usize, input: &Input) -> i64 {
        let mut cost = 0;
        for col in 0..self.ws.len() {
            let exceed_cost = to_squeezed_height(
                &mut self.squeezed_heights[d][col],
                &mut self.heights[d][col],
                self.height_sum[d][col],
                &self.r[d][col],
                d,
                self.ws[col],
                input,
                &mut self.shared.space_for_r_idx,
            );
            let heights = if self.height_sum[d][col] <= input.W {
                &self.heights[d][col]
            } else {
                &self.squeezed_heights[d][col]
            };
            let mut groups = vec![];
            let (match_count, _) = match_greedy(
                &self.prev_h[col],
                &self.prev_rem[col],
                &heights,
                input.W,
                &mut groups,
            );
            let switch_count = if d == 0 {
                0
            } else {
                heights.len() + self.prev_h[col].len() - match_count * 2
            };
            cost += switch_count as i64 * self.ws[col];
            cost += exceed_cost;

            let mut next_nodes = vec![];
            for i in 0..groups.len() {
                let (prev_g, next_g, rem) = groups[i];
                let prev_nodes = &self.node_idx[d][col][prev_g.0..prev_g.1].to_vec();
                let next_h = heights[next_g.0..next_g.1].to_vec();
                next_nodes.extend(self.graphs[col].connect(prev_nodes, &next_h, rem));
            }
            self.node_idx[d + 1].push(next_nodes);
        }
        cost
    }
}

#[derive(Clone)]
struct ColGraph {
    nodes: Vec<Node>,
    root_node: usize,
}

impl ColGraph {
    fn new(w: i64) -> ColGraph {
        let mut graph = ColGraph {
            root_node: 0,
            nodes: vec![],
        };
        graph.root_node = graph.new_node(0, w);
        graph
    }

    fn new_node(&mut self, rem: i64, height: i64) -> usize {
        let mut node = Node::new();
        node.height = height;
        node.rem = rem;
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn connect(
        &mut self,
        prev_nodes: &Vec<usize>,
        next_heights: &Vec<i64>,
        rem: i64,
    ) -> Vec<usize> {
        let inter_node = self.new_node(rem, 0);
        for &par_node in prev_nodes.iter() {
            self.nodes[par_node].children.push(inter_node);
            self.nodes[inter_node].parents.push(par_node);
        }

        let mut next_nodes = vec![];
        for &h in next_heights.iter() {
            let next_node = self.new_node(0, h);
            self.nodes[inter_node].children.push(next_node);
            self.nodes[next_node].parents.push(inter_node);
            next_nodes.push(next_node);
        }

        self.backpropagate_rem(inter_node);

        next_nodes
    }

    /// remを親のノードから取ってくる
    /// 取ってきたら、経路のheightに足す
    fn backpropagate_rem(&mut self, inter_node: usize) {
        let prev_height_sum = self.nodes[inter_node]
            .parents
            .iter()
            .map(|&i| self.nodes[i].height)
            .sum::<i64>();
        let next_height_sum = self.nodes[inter_node]
            .children
            .iter()
            .map(|&i| self.nodes[i].height)
            .sum::<i64>();

        let mut need_rem = next_height_sum + self.nodes[inter_node].rem - prev_height_sum;

        let mut q = vec![inter_node];
        let mut seen = HashSet::new();
        seen.insert(inter_node);
        let mut rem_cand_node = vec![];
        while let Some(v) = q.pop() {
            if self.nodes[v].rem > 0 {
                rem_cand_node.push((
                    self.nodes[v].node_r,
                    self.nodes[v].node_l,
                    v,
                    self.nodes[v].rem,
                ));
            }
            for &u in self.nodes[v].parents.iter() {
                if !seen.insert(u) {
                    continue;
                }
                q.push(u);
            }
        }

        rem_cand_node.sort_by(|a, b| {
            if a.0 != b.0 {
                a.0.cmp(&b.0)
            } else {
                b.1.cmp(&a.1)
            }
        });
        let mut use_rem_nodes = HashMap::new();
        for (_, _, node_idx, rem) in rem_cand_node {
            let use_rem = rem.min(need_rem);
            use_rem_nodes.insert(node_idx, use_rem);
            need_rem -= use_rem;
            self.nodes[node_idx].rem -= use_rem;
        }
        assert_eq!(need_rem, 0);

        let mut q = vec![inter_node];
        let mut seen = HashSet::new();
        self.back_dfs(&mut q, &mut seen, &mut use_rem_nodes);
    }

    // need_remをparentsから貰いに行く
    // もらったら、経路上の全てのheightに足す
    fn back_dfs(
        &mut self,
        q: &mut Vec<usize>,
        seen: &mut HashSet<usize>,
        use_rem_nodes: &HashMap<usize, i64>,
    ) {
        let v = *q.last().unwrap();
        if let Some(&use_rem) = use_rem_nodes.get(&v) {
            for &u in q.iter() {
                if self.nodes[u].height > 0 {
                    self.nodes[u].height += use_rem;
                }
            }
        }
        let par = self.nodes[v].parents.clone();
        for &u in par.iter() {
            if !seen.insert(u) {
                continue;
            }
            q.push(u);
            self.back_dfs(q, seen, use_rem_nodes);
            q.pop();
        }
    }

    /// remを最初の子孫のheightに渡す
    /// 解を作成するとき用
    fn propagate_rem(&mut self) {
        let mut q = VecDeque::new();
        q.push_back(self.root_node);
        let mut seen = HashSet::new();
        seen.insert(self.root_node);
        while let Some(v) = q.pop_front() {
            if self.nodes[v].children.len() > 0 {
                let first_child = self.nodes[v].children[0];
                self.nodes[first_child].height += self.nodes[v].rem;
                self.nodes[first_child].rem += self.nodes[v].rem;
                self.nodes[v].rem = 0;
            }
            for &u in self.nodes[v].children.iter() {
                if !seen.insert(u) {
                    continue;
                }
                q.push_back(u);
            }
        }
    }

    fn gen_rem(&mut self, prev_nodes: &Vec<usize>) -> Vec<Vec<(usize, i64)>> {
        // node_l、node_rをリセットする
        let mut q = vec![self.root_node];
        let mut seen = HashSet::new();
        seen.insert(self.root_node);
        while let Some(v) = q.pop() {
            self.nodes[v].node_l = usize::MAX;
            self.nodes[v].node_r = usize::MIN;
            for &u in self.nodes[v].children.iter() {
                if !seen.insert(u) {
                    continue;
                }
                q.push(u);
            }
        }

        // prev_nodesから昇って、node_l、node_rを設定する
        for (i, &node_idx) in prev_nodes.iter().enumerate() {
            let mut q = vec![node_idx];
            let mut seen = HashSet::new();
            seen.insert(node_idx);
            while let Some(v) = q.pop() {
                self.nodes[v].node_l = self.nodes[v].node_l.min(i);
                self.nodes[v].node_r = self.nodes[v].node_r.max(i);
                for &u in self.nodes[v].parents.iter() {
                    if !seen.insert(u) {
                        continue;
                    }
                    q.push(u);
                }
            }
        }

        // rem[i][j] := iから使い始められて、jまでには消費する必要がある余裕
        let mut total_rem = vec![vec![0; prev_nodes.len()]; prev_nodes.len()];

        // 余裕があるノードを全て拾い、remに保管する
        let mut q = vec![self.root_node];
        let mut seen = HashSet::new();
        seen.insert(self.root_node);
        while let Some(v) = q.pop() {
            assert!(
                self.nodes[v].node_l <= self.nodes[v].node_r
                    && self.nodes[v].node_r < prev_nodes.len()
            );
            total_rem[self.nodes[v].node_l][self.nodes[v].node_r] += self.nodes[v].rem;
            for &u in self.nodes[v].children.iter() {
                if !seen.insert(u) {
                    continue;
                }
                q.push(u);
            }
        }

        let mut rem = vec![vec![]; prev_nodes.len()];
        for l in 0..prev_nodes.len() {
            for r in 0..prev_nodes.len() {
                if total_rem[l][r] > 0 {
                    rem[l].push((r, total_rem[l][r]));
                }
            }
        }

        rem
    }
}

#[derive(Clone, Debug)]
struct Node {
    parents: Vec<usize>,  // 親ノード
    children: Vec<usize>, // 子ノード
    rem: i64,             // 余裕
    height: i64,          // 高さ
    node_l: usize,        // 最初に使えるノード
    node_r: usize,        // 最後に使えるノード
}

impl Node {
    fn new() -> Node {
        Node {
            parents: vec![],
            children: vec![],
            rem: 0,
            height: 0,
            node_l: usize::MAX,
            node_r: usize::MIN,
        }
    }
}

#[test]
fn test_match_greedy() {
    fn create_rem_mat(len: usize, rem: Vec<(usize, usize, i64)>) -> Vec<Vec<(usize, i64)>> {
        let mut mat = vec![vec![]; len];
        for (l, r, val) in rem {
            mat[l].push((r, val));
        }
        mat
    }

    fn assert_match_result(
        prev_h: &Vec<i64>,
        next_h: &Vec<i64>,
        groups: &Vec<((usize, usize), (usize, usize), i64)>,
        prev_pickup_rem: &Vec<i64>,
        w: i64,
    ) {
        let used_next_rem = groups.iter().map(|x| x.2).sum::<i64>();
        let remain_prev_rem = prev_pickup_rem.iter().sum::<i64>();
        let next_h_sum = next_h.iter().sum::<i64>();
        assert_eq!(
            used_next_rem + remain_prev_rem + next_h_sum,
            w,
            "used_next_rem: {}, remain_prev_rem: {}, next_h_sum: {}",
            used_next_rem,
            remain_prev_rem,
            next_h_sum
        );
        assert_eq!(groups.iter().map(|x| x.0 .1).max().unwrap(), prev_h.len());
        assert_eq!(groups.iter().map(|x| x.1 .1).max().unwrap(), next_h.len());
    }

    const W: i64 = 100;

    // 1:1
    {
        let prev_h = vec![40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 60)]);
        let next_h = vec![80];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![80];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20)]);
        let next_h = vec![40];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 1:2
    {
        let prev_h = vec![40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 60)]);
        let next_h = vec![20, 30];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![60];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 40)]);
        let next_h = vec![20, 30];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![60];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 40)]);
        let next_h = vec![60, 20];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 2:1
    {
        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 40)]);
        let next_h = vec![10];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 40)]);
        let next_h = vec![80];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 40)]);
        let next_h = vec![30];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(1, 1, 40)]);
        let next_h = vec![30];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 40)]);
        let next_h = vec![30];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 2:2
    {
        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (1, 1, 30)]);
        let next_h = vec![20, 30];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![20, 30];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![20, 40];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        dbg!(&match_count, &groups, &prev_pickup_rem);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![20, 20];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![30, 20];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![50, 20];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![80, 10];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (1, 1, 30)]);
        let next_h = vec![80, 10];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![60, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 10)]);
        let next_h = vec![60, 10];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 3:1
    {
        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 20), (0, 2, 20)]);
        let next_h = vec![10];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 20), (2, 2, 20)]);
        let next_h = vec![10];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 2, 10)]);
        let next_h = vec![10];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 3:2
    {
        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (1, 2, 20)]);
        let next_h = vec![50, 40];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 20), (0, 2, 20)]);
        let next_h = vec![50, 40];
        let mut groups = vec![];
        let (match_count, prev_pickup_rem) =
            match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 3:3
    let prev_h = vec![20, 30, 40];
    let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 2, 10)]);
    let next_h = vec![20, 30, 40];
    let mut groups = vec![];
    let (match_count, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
    assert_eq!(match_count, 3);
    assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

    dbg!(&match_count, &groups, &prev_pickup_rem);
}

#[test]
fn test_stackgraph() {
    const W: i64 = 1000;
    let mut graph = ColGraph::new(W);
    let hs = vec![
        vec![205, 409, 121],
        vec![700, 122, 131],
        vec![661, 80],
        vec![984],
        vec![619],
    ];
    let mut node_idx = vec![vec![]; hs.len() + 1];
    let w = 10;
    let mut score = 0;

    node_idx[0] = vec![graph.root_node];

    for (d, h) in hs.into_iter().enumerate() {
        let prev_nodes = &node_idx[d];
        let prev_h: Vec<i64> = prev_nodes.iter().map(|&i| graph.nodes[i].height).collect();
        let prev_rem = graph.gen_rem(&prev_nodes);
        let next_h = h;
        let mut groups = vec![];
        let (match_count, _) = match_greedy(&prev_h, &prev_rem, &next_h, W, &mut groups);
        let switch_count = prev_h.len() + next_h.len() - match_count * 2;
        score += switch_count as i64 * w;
        // graph.return_rem(&mut prev_pickup_rem); // 不要？

        let mut next_nodes = vec![];
        for i in 0..groups.len() {
            let (prev_g, next_g, rem) = groups[i];
            let prev_nodes = &node_idx[d][prev_g.0..prev_g.1].to_vec();
            let next_h = &next_h[next_g.0..next_g.1].to_vec();
            next_nodes.extend(graph.connect(prev_nodes, &next_h, rem));
        }

        node_idx[d + 1] = next_nodes;
    }

    eprintln!("{}", score);
}
