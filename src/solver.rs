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
) -> (usize, Vec<((usize, usize), (usize, usize), i64)>, Vec<i64>) {
    let mut match_count = 0;
    let mut groups = vec![];
    let mut prev_pickup_rems = vec![0; prev_h.len()];

    let mut prev_rem_sum = 0;
    let mut next_rem_sum = w - next_h.iter().sum::<i64>();

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

    (match_count, groups, prev_pickup_rems)
}

/// match_greedyで、vec![]を消したバージョン
/// 切り替えを減らした回数だけを返す
fn match_greedy_fast(
    prev_h: &Vec<i64>,
    prev_rem: &Vec<Vec<(usize, i64)>>,
    next_h: &Vec<i64>,
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
    let mut next_rem_sum = w - next_h.iter().sum::<i64>();

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
    mut heights: Vec<i64>,
    r_idx: &Vec<usize>,
    d: usize,
    w: i64,
    input: &Input,
) -> (Vec<i64>, i64) {
    // 超過してしまう場合は、縮めることで損をしない領域（余りが大きい領域）を縮める
    let required_height = heights.iter().sum::<i64>();
    let mut over_height = (required_height - input.W).max(0);
    let mut exceed_cost = 0;
    if over_height > 0 {
        let mut space_for_r_idx = (0..heights.len())
            .map(|i| {
                let d = input.A[d][r_idx[i]] % w;
                (if d == 0 { w } else { d }, i)
            })
            .collect::<Vec<(i64, usize)>>();
        space_for_r_idx.sort();
        while over_height > 0 {
            let (space, idx) = space_for_r_idx.remove(0);
            if heights[idx] > 1 {
                exceed_cost += space;
                heights[idx] -= 1;
                over_height -= 1;
            }
            space_for_r_idx.push((w, idx));
        }
    }

    (heights, exceed_cost * EXCEED_COST)
}

pub struct Solver<'a> {
    state: State,
    input: &'a Input,
}

impl<'a> Solver<'a> {
    pub fn new(ws: Vec<i64>, r: Vec<Vec<Vec<usize>>>, input: &Input) -> Solver {
        Solver {
            state: State::new(ws, r, input.W, input.D),
            input,
        }
    }

    pub fn solve(&mut self) -> Answer {
        let mut total_cost = 0;

        for d in 0..self.input.D {
            self.state.setup_prev_h_rem(d);
            self.optimize_r(d);

            total_cost += self.state.to_next_d(d, self.input);
        }

        self.create_answer(total_cost)
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
                assert_eq!(height, self.input.W, "{} {}", d, col);
            }
            width += w;
        }

        ans
    }

    /// TODO: d=0は余裕だけを評価する
    fn optimize_r(&mut self, d: usize) {
        let mut cur_score_col: Vec<(i64, i64)> = (0..self.state.ws.len())
            .map(|col| self.state.eval_col(d, col, self.input, true))
            .collect();
        let mut _cur_score = cur_score_col.iter().map(|x| x.0 + x.1).sum::<i64>();

        let start_temp: f64 = 1e2; // :param
        let end_temp: f64 = 1e-1; // :param

        let start_time = time::elapsed_seconds();
        let duration = ((TIME_LIMIT - start_time) / (self.input.D - d) as f64).max(1e-3);
        let end_time = start_time + duration;

        let mut adapt_in_shuf = 0;
        let mut adapt_in_swap = 0;
        let mut adapt_tr_move = 0;
        let mut adapt_tr_swap = 0;

        let mut action_ratio = vec![0.2, 0.2, 0.2, 0.4];
        for i in 0..action_ratio.len() - 1 {
            action_ratio[i + 1] += action_ratio[i];
        }

        // eprintln!("start_cur_score: {}", cur_score);
        while time::elapsed_seconds() < end_time {
            let progress = (time::elapsed_seconds() - start_time) / duration;
            let cur_temp = start_temp.powf(1. - progress) * end_temp.powf(progress);
            let threshold = -cur_temp * rnd::nextf().ln();
            let p = rnd::nextf();

            if p < action_ratio[0] {
                // 列内シャッフル
                let col = rnd::gen_index(self.state.ws.len());
                if self.state.r[d][col].len() == 1 {
                    continue;
                }
                let mut new_r = self.state.r[d][col].clone();
                rnd::shuffle(&mut new_r);
                std::mem::swap(&mut self.state.r[d][col], &mut new_r);

                let new_score_col =
                    self.state
                        .eval_col(d, col, self.input, cur_score_col[col].1 > 0);
                let score_diff =
                    new_score_col.0 + new_score_col.1 - cur_score_col[col].0 - cur_score_col[col].1;
                if (score_diff as f64) <= threshold {
                    cur_score_col[col] = new_score_col;
                    _cur_score += score_diff;
                    adapt_in_shuf += 1;
                } else {
                    std::mem::swap(&mut self.state.r[d][col], &mut new_r);
                }
            } else if p < action_ratio[1] {
                // 列内n回swap
                // TODO: 隣接swap
                let col = rnd::gen_index(self.state.ws.len());
                if self.state.r[d][col].len() == 1 {
                    continue;
                }
                let swap_count = rnd::gen_range(1, self.state.r[d][col].len().clamp(1, 2) + 1);
                let swaps = (0..swap_count)
                    .map(|_| {
                        (
                            rnd::gen_index(self.state.r[d][col].len()),
                            rnd::gen_index(self.state.r[d][col].len()),
                        )
                    })
                    .filter(|(i, j)| i != j)
                    .collect::<Vec<(usize, usize)>>();
                if swaps.len() == 0 {
                    continue;
                }

                for &(i, j) in swaps.iter() {
                    self.state.r[d][col].swap(i, j);
                }
                let new_score_col =
                    self.state
                        .eval_col(d, col, self.input, cur_score_col[col].1 > 0);
                let score_diff =
                    new_score_col.0 + new_score_col.1 - cur_score_col[col].0 - cur_score_col[col].1;
                if (score_diff as f64) <= threshold {
                    cur_score_col[col] = new_score_col;
                    _cur_score += score_diff;
                    adapt_in_swap += 1;
                } else {
                    for &(i, j) in swaps.iter().rev() {
                        self.state.r[d][col].swap(i, j);
                    }
                }
            } else if p < action_ratio[2] {
                // 列間1個移動
                let col1 = rnd::gen_index(self.state.ws.len());
                if self.state.r[d][col1].len() <= 1 {
                    continue;
                }
                let col2 = rnd::gen_index(self.state.ws.len());
                if col1 == col2 {
                    continue;
                }
                let i1 = rnd::gen_index(self.state.r[d][col1].len());
                let r1 = self.state.r[d][col1].remove(i1);
                let i2 = rnd::gen_index(self.state.r[d][col2].len() + 1);
                self.state.r[d][col2].insert(i2, r1);
                let new_score_col1 =
                    self.state
                        .eval_col(d, col1, self.input, cur_score_col[col1].1 > 0);
                let new_score_col2 =
                    self.state
                        .eval_col(d, col2, self.input, cur_score_col[col2].1 > 0);
                let score_diff =
                    new_score_col1.0 + new_score_col1.1 + new_score_col2.0 + new_score_col2.1
                        - cur_score_col[col1].0
                        - cur_score_col[col1].1
                        - cur_score_col[col2].0
                        - cur_score_col[col2].1;

                if (score_diff as f64) <= threshold {
                    cur_score_col[col1] = new_score_col1;
                    cur_score_col[col2] = new_score_col2;
                    _cur_score += score_diff;
                    adapt_tr_move += 1;
                } else {
                    self.state.r[d][col2].remove(i2);
                    self.state.r[d][col1].insert(i1, r1);
                }
            } else {
                // 列間n:nスワップ
                let move_count = rnd::gen_range(2, 6); // :param
                let col1 = rnd::gen_index(self.state.ws.len());
                let col2 = rnd::gen_index(self.state.ws.len());
                if col1 == col2 {
                    continue;
                }
                let i1 = rnd::gen_index(self.state.r[d][col1].len());
                let i2 = rnd::gen_index(self.state.r[d][col2].len());
                let mut cnt1 = 0;
                let mut cnt2 = 0;
                let mut h1 = 0;
                let mut h2 = 0;

                // 低い方に足す
                for _ in 0..move_count {
                    if h1 <= h2 && i1 + cnt1 < self.state.r[d][col1].len() {
                        h1 += self.input.A[d][self.state.r[d][col1][i1 + cnt1]];
                        cnt1 += 1;
                    } else if h2 <= h1 && i2 + cnt2 < self.state.r[d][col2].len() {
                        h2 += self.input.A[d][self.state.r[d][col2][i2 + cnt2]];
                        cnt2 += 1;
                    }
                }
                let rs1 = (0..cnt1)
                    .map(|_| self.state.r[d][col1].remove(i1))
                    .collect::<Vec<usize>>();
                let rs2 = (0..cnt2)
                    .map(|_| self.state.r[d][col2].remove(i2))
                    .collect::<Vec<usize>>();
                for &r1 in rs1.iter().rev() {
                    self.state.r[d][col2].insert(i2, r1);
                }
                for &r2 in rs2.iter().rev() {
                    self.state.r[d][col1].insert(i1, r2);
                }
                let new_score_col1 =
                    self.state
                        .eval_col(d, col1, self.input, cur_score_col[col1].1 > 0);
                let new_score_col2 =
                    self.state
                        .eval_col(d, col2, self.input, cur_score_col[col2].1 > 0);
                let score_diff =
                    new_score_col1.0 + new_score_col1.1 + new_score_col2.0 + new_score_col2.1
                        - cur_score_col[col1].0
                        - cur_score_col[col1].1
                        - cur_score_col[col2].0
                        - cur_score_col[col2].1;

                if (score_diff as f64) <= threshold {
                    cur_score_col[col1] = new_score_col1;
                    cur_score_col[col2] = new_score_col2;
                    _cur_score += score_diff;
                    adapt_tr_swap += 1;
                } else {
                    for _ in 0..cnt2 {
                        self.state.r[d][col1].remove(i1);
                    }
                    for _ in 0..cnt1 {
                        self.state.r[d][col2].remove(i2);
                    }
                    for &r1 in rs1.iter().rev() {
                        self.state.r[d][col1].insert(i1, r1);
                    }
                    for &r2 in rs2.iter().rev() {
                        self.state.r[d][col2].insert(i2, r2);
                    }
                }
            }
        }
        eprintln!("in-shuf:     {}", adapt_in_shuf);
        eprintln!("in-swap:     {}", adapt_in_swap);
        eprintln!("tr-move:     {}", adapt_tr_move);
        eprintln!("tr-swap:     {}", adapt_tr_swap);
    }
}

struct State {
    ws: Vec<i64>,
    // r[d][col][i]
    r: Vec<Vec<Vec<usize>>>,
    // node_idx[d][col][i]
    node_idx: Vec<Vec<Vec<usize>>>,
    prev_h: Vec<Vec<i64>>,
    prev_rem: Vec<Vec<Vec<(usize, i64)>>>,
    graphs: Vec<ColGraph>,
    shared_v: Vec<i64>,
}

impl State {
    fn new(ws: Vec<i64>, r: Vec<Vec<Vec<usize>>>, w: i64, d: usize) -> State {
        const MAX_N: usize = 50;
        let col_count = ws.len();
        let mut state = State {
            ws,
            r,
            prev_h: vec![],
            prev_rem: vec![],
            graphs: vec![ColGraph::new(w); col_count],
            node_idx: vec![vec![]; d + 1],
            shared_v: vec![0; MAX_N],
        };
        for col in 0..col_count {
            state.node_idx[0].push(vec![state.graphs[col].root_node]);
        }
        state
    }

    fn eval_col(
        &mut self,
        d: usize,
        col: usize,
        input: &Input,
        allow_overflow: bool,
    ) -> (i64, i64) {
        // TODO: タイブレーク時には、余裕の残り具合も足す
        // TODO: 幅も考慮する
        // NOTE: 切り替え回数が変わらないなら、遷移した方が良い（スコアの差分はない方が良いのか？）
        let r_idx = &self.r[d][col];
        let heights = r_idx
            .iter()
            .map(|&r_idx| ceil_div(input.A[d][r_idx], self.ws[col]))
            .collect::<Vec<i64>>();
        if !allow_overflow && heights.iter().sum::<i64>() > input.W {
            return (0, 1 << 40);
        }
        let (next_h, exceed_cost) =
            to_squeezed_height(heights, &self.r[d][col], d, self.ws[col], input);
        let match_count = match_greedy_fast(
            &self.prev_h[col],
            &self.prev_rem[col],
            &next_h,
            input.W,
            &mut self.shared_v,
        );
        let switch_count = self.prev_h.len() + next_h.len() - match_count * 2;

        (switch_count as i64 * self.ws[col], exceed_cost)
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

    fn to_next_d(&mut self, d: usize, input: &Input) -> i64 {
        let mut cost = 0;
        for col in 0..self.ws.len() {
            let r_idx = &self.r[d][col];
            let heights = r_idx
                .iter()
                .map(|&r_idx| ceil_div(input.A[d][r_idx], self.ws[col]))
                .collect::<Vec<i64>>();
            let (next_h, exceed_cost) = to_squeezed_height(heights, &r_idx, d, self.ws[col], input);
            let (match_count, groups, _) =
                match_greedy(&self.prev_h[col], &self.prev_rem[col], &next_h, input.W);
            let switch_count = if d == 0 {
                0
            } else {
                next_h.len() + self.prev_h[col].len() - match_count * 2
            };
            cost += switch_count as i64 * self.ws[col];
            cost += exceed_cost;

            let mut next_nodes = vec![];
            for i in 0..groups.len() {
                let (prev_g, next_g, rem) = groups[i];
                let prev_nodes = &self.node_idx[d][col][prev_g.0..prev_g.1].to_vec();
                let next_h = &next_h[next_g.0..next_g.1].to_vec();
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
        let par = self.nodes[v].parents.clone(); // TODO: remove
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
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![80];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20)]);
        let next_h = vec![40];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 1:2
    {
        let prev_h = vec![40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 60)]);
        let next_h = vec![20, 30];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![60];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 40)]);
        let next_h = vec![20, 30];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![60];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 40)]);
        let next_h = vec![60, 20];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 2:1
    {
        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 40)]);
        let next_h = vec![10];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 40)]);
        let next_h = vec![80];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 40)]);
        let next_h = vec![30];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(1, 1, 40)]);
        let next_h = vec![30];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 40)]);
        let next_h = vec![30];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 2:2
    {
        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (1, 1, 30)]);
        let next_h = vec![20, 30];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![20, 30];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![20, 40];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        dbg!(&match_count, &groups, &prev_pickup_rem);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![20, 20];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![30, 20];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![50, 20];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![80, 10];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (1, 1, 30)]);
        let next_h = vec![80, 10];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![60, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 10)]);
        let next_h = vec![60, 10];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 3:1
    {
        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 20), (0, 2, 20)]);
        let next_h = vec![10];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 20), (2, 2, 20)]);
        let next_h = vec![10];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![20, 30, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 2, 10)]);
        let next_h = vec![10];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 3:2
    {
        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (1, 2, 20)]);
        let next_h = vec![50, 40];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);

        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 20), (0, 2, 20)]);
        let next_h = vec![50, 40];
        let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &prev_pickup_rem, W);
    }

    // 3:3
    let prev_h = vec![20, 30, 40];
    let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 2, 10)]);
    let next_h = vec![20, 30, 40];
    let (match_count, groups, prev_pickup_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
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

        let (match_count, groups, _) = match_greedy(&prev_h, &prev_rem, &next_h, W);
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
