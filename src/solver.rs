use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use crate::def::*;
use crate::util::*;

/// Returns:
/// match_count: i64 = 切り替えをした回数
/// groups: Vec<(((usize, usize), (usize, usize), i64)> = 対応するノードの添字、次以降に使える余裕
/// new_prev_rem: Vec<i64> = 前の余裕の消費具合
fn match_greedy(
    prev_h: &Vec<i64>,
    prev_rem: &Vec<Vec<i64>>,
    next_h: &Vec<i64>,
    w: i64,
) -> (
    usize,
    Vec<((usize, usize), (usize, usize), i64)>,
    Vec<Vec<i64>>,
) {
    let mut match_count = 0;
    let mut groups = vec![];
    let mut new_prev_rem = prev_rem.clone();

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
            for i in prev_r..prev_h.len() {
                prev_rem_sum += new_prev_rem[prev_r][i];
            }

            // 回収する
            if prev_r < prev_h.len() - 1 {
                for i in 0..=prev_r {
                    prev_height += new_prev_rem[i][prev_r];
                    prev_rem_sum -= new_prev_rem[i][prev_r];
                    new_prev_rem[i][prev_r] = 0;
                }
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
            // NOTE: 最終的な更新でだけする必要がある
            let mut use_prev_rem = next_height - prev_height;
            prev_rem_sum -= use_prev_rem;
            for i in 0..prev_h.len() {
                for j in i..prev_h.len() {
                    if i <= prev_r - 1 && prev_r - 1 <= j {
                        let r = new_prev_rem[i][j].min(use_prev_rem);
                        new_prev_rem[i][j] -= r;
                        use_prev_rem -= r;
                    }
                }
            }
            assert_eq!(use_prev_rem, 0, "{:?} {:?}", prev_rem, new_prev_rem);
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

    (match_count, groups, new_prev_rem)
}

#[test]
fn test_match_greedy() {
    fn create_rem_mat(len: usize, rem: Vec<(usize, usize, i64)>) -> Vec<Vec<i64>> {
        let mut mat = vec![vec![0; len]; len];
        for (l, r, val) in rem {
            mat[l][r] += val;
        }
        mat
    }

    fn assert_match_result(
        prev_h: &Vec<i64>,
        next_h: &Vec<i64>,
        groups: &Vec<((usize, usize), (usize, usize), i64)>,
        new_prev_rem: &Vec<Vec<i64>>,
        w: i64,
    ) {
        let used_next_rem = groups.iter().map(|x| x.2).sum::<i64>();
        let remain_prev_rem = new_prev_rem
            .iter()
            .map(|x| x.iter().sum::<i64>())
            .sum::<i64>();
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
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![80];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20)]);
        let next_h = vec![40];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);
    }

    // 1:2
    {
        let prev_h = vec![40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 60)]);
        let next_h = vec![20, 30];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![60];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 40)]);
        let next_h = vec![20, 30];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![60];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 40)]);
        let next_h = vec![60, 20];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);
    }

    // 2:1
    {
        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 40)]);
        let next_h = vec![10];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 40)]);
        let next_h = vec![80];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 40)]);
        let next_h = vec![30];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(1, 1, 40)]);
        let next_h = vec![30];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 40)]);
        let next_h = vec![30];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);
    }

    // 2:2
    {
        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (1, 1, 30)]);
        let next_h = vec![20, 30];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![20, 30];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![20, 40];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![20, 20];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![30, 20];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![50, 20];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (0, 1, 30)]);
        let next_h = vec![80, 10];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (1, 1, 30)]);
        let next_h = vec![80, 10];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![60, 30];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 10)]);
        let next_h = vec![60, 10];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);
    }

    // 3:1
    {
        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 20), (0, 2, 20)]);
        let next_h = vec![10];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 20), (2, 2, 20)]);
        let next_h = vec![10];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![20, 30, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 2, 10)]);
        let next_h = vec![10];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 1);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);
    }

    // 3:2
    {
        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 0, 20), (1, 2, 20)]);
        let next_h = vec![50, 40];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

        let prev_h = vec![10, 10, 40];
        let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 1, 20), (0, 2, 20)]);
        let next_h = vec![50, 40];
        let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        assert_eq!(match_count, 2);
        assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);
    }

    // 3:3
    let prev_h = vec![20, 30, 40];
    let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 2, 10)]);
    let next_h = vec![20, 30, 40];
    let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
    assert_eq!(match_count, 3);
    assert_match_result(&prev_h, &next_h, &groups, &new_prev_rem, W);

    dbg!(&match_count, &groups, &new_prev_rem);
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
        let start = time::elapsed_seconds();
        let mut a = 0.;

        let mut total_cost = 0;

        for d in 0..self.input.D {
            // 事前計算
            let prev_h: Vec<Vec<i64>> = (0..self.state.ws.len())
                .map(|col| {
                    self.state.node_idx[d][col]
                        .iter()
                        .map(|&i| self.state.trees[col].nodes[i].height)
                        .collect()
                })
                .collect();
            let prev_rem: Vec<Vec<Vec<i64>>> = (0..self.state.ws.len())
                .map(|col| self.state.trees[col].gen_rem(&self.state.node_idx[d][col]))
                .collect();

            a -= time::elapsed_seconds();
            self.optimize_r(&prev_h, &prev_rem, d);
            a += time::elapsed_seconds();

            // 事後計算
            for col in 0..self.state.ws.len() {
                let next_h = self.state.r[d][col]
                    .iter()
                    .map(|&r_idx| ceil_div(self.input.A[d][r_idx], self.state.ws[col]))
                    .collect();
                let (match_count, groups, _) =
                    match_greedy(&prev_h[col], &prev_rem[col], &next_h, self.input.W);
                let switch_count = if d == 0 {
                    0
                } else {
                    next_h.len() + prev_h[col].len() - match_count * 2
                };
                total_cost += switch_count as i64 * self.state.ws[col];

                let mut next_nodes = vec![];
                for i in 0..groups.len() {
                    let (prev_g, next_g, rem) = groups[i];
                    let prev_nodes = &self.state.node_idx[d][col][prev_g.0..prev_g.1].to_vec();
                    let next_h = &next_h[next_g.0..next_g.1].to_vec();
                    next_nodes.extend(self.state.trees[col].connect(prev_nodes, &next_h, rem));
                }
                self.state.node_idx[d + 1].push(next_nodes);
            }
        }
        let total = time::elapsed_seconds() - start;
        eprintln!("a: {:.5} ({:.5})", a, a / total);

        let mut ans = Answer::new(self.input.D, self.input.N, total_cost);
        let mut width = 0;
        for col in 0..self.state.ws.len() {
            let w = self.state.ws[col];
            self.state.trees[col].propagate_rem();
            for d in 0..self.input.D {
                let mut height = 0;
                for (i, &r_idx) in self.state.r[d][col].iter().enumerate() {
                    let node_idx = self.state.node_idx[d + 1][col][i];
                    let h = self.state.trees[col].nodes[node_idx].height;
                    ans.p[d][r_idx] = (height, width, height + h, width + w);
                    height += h;
                }
                assert_eq!(height, self.input.W, "{} {}", d, col);
            }
            width += w;
        }

        ans
    }

    /// TODO:
    /// - d=0だけ特別に処理する（d=1）からにする
    /// - d=0は余裕だけを評価する
    fn optimize_r(&mut self, prev_h: &Vec<Vec<i64>>, prev_rem: &Vec<Vec<Vec<i64>>>, d: usize) {
        let mut current_score_col: Vec<i64> = (0..self.state.ws.len())
            .map(|col| {
                self.state
                    .eval_col(d, col, &prev_h[col], &prev_rem[col], self.input)
            })
            .collect();
        let mut current_score = current_score_col.iter().sum::<i64>();

        for _ in 0..10000 {
            let p = rnd::nextf();
            if p < 0.5 {
                // 列内1:1swap
                let col = rnd::gen_index(self.state.ws.len());
                let (i, j) = (
                    rnd::gen_index(self.state.r[d][col].len()),
                    rnd::gen_index(self.state.r[d][col].len()),
                );
                if i == j {
                    continue;
                }
                self.state.r[d][col].swap(i, j);

                let new_score_col =
                    self.state
                        .eval_col(d, col, &prev_h[col], &prev_rem[col], self.input);
                let score_diff = new_score_col - current_score_col[col];
                if score_diff < 0 {
                    eprintln!("{} -> {}", current_score, current_score + score_diff);
                    current_score_col[col] = new_score_col;
                    current_score += score_diff;
                } else {
                    self.state.r[d][col].swap(i, j);
                }
            } else {
                // 列間1:1swap
                let (col1, col2) = (
                    rnd::gen_index(self.state.ws.len()),
                    rnd::gen_index(self.state.ws.len()),
                );
                if col1 == col2 {
                    continue;
                }
                let (i1, i2) = (
                    rnd::gen_index(self.state.r[d][col1].len()),
                    rnd::gen_index(self.state.r[d][col2].len()),
                );
                (self.state.r[d][col1][i1], self.state.r[d][col2][i2]) =
                    (self.state.r[d][col2][i2], self.state.r[d][col1][i1]);

                let new_score_col1 =
                    self.state
                        .eval_col(d, col1, &prev_h[col1], &prev_rem[col1], self.input);
                let new_score_col2 =
                    self.state
                        .eval_col(d, col2, &prev_h[col2], &prev_rem[col2], self.input);
                let score_diff = new_score_col1 + new_score_col2
                    - current_score_col[col1]
                    - current_score_col[col2];

                if score_diff < 0 {
                    eprintln!("{} -> {}", current_score, current_score + score_diff);
                    current_score_col[col1] = new_score_col1;
                    current_score_col[col2] = new_score_col2;
                    current_score += score_diff;
                } else {
                    (self.state.r[d][col1][i1], self.state.r[d][col2][i2]) =
                        (self.state.r[d][col2][i2], self.state.r[d][col1][i1]);
                }
            }
        }
    }
}

struct State {
    ws: Vec<i64>,
    // r[d][col][i]
    r: Vec<Vec<Vec<usize>>>,
    // node_idx[d][col][i]
    node_idx: Vec<Vec<Vec<usize>>>,
    trees: Vec<StackTree>,
}

impl State {
    fn new(ws: Vec<i64>, r: Vec<Vec<Vec<usize>>>, w: i64, d: usize) -> State {
        let col_count = ws.len();
        let mut state = State {
            ws,
            r,
            trees: vec![StackTree::new(w); col_count],
            node_idx: vec![vec![]; d + 1],
        };
        for col in 0..col_count {
            state.node_idx[0].push(vec![state.trees[col].root_node]);
        }
        state
    }

    fn eval_col(
        &self,
        d: usize,
        col: usize,
        prev_h: &Vec<i64>,
        prev_rem: &Vec<Vec<i64>>,
        input: &Input,
    ) -> i64 {
        let next_h = self.r[d][col]
            .iter()
            .map(|&r_idx| ceil_div(input.A[d][r_idx], self.ws[col]))
            .collect::<Vec<i64>>();

        if next_h.iter().sum::<i64>() > input.W {
            return 1 << 30;
        }

        let (match_count, _, _) = match_greedy(&prev_h, &prev_rem, &next_h, input.W);
        let switch_count = prev_h.len() + next_h.len() - match_count * 2;

        switch_count as i64 * self.ws[col]
    }
}

#[test]
fn test_stacktree() {
    const W: i64 = 1000;
    let mut tree = StackTree::new(W);
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

    node_idx[0] = vec![tree.root_node];

    for (d, h) in hs.into_iter().enumerate() {
        let prev_nodes = &node_idx[d];
        let prev_h: Vec<i64> = prev_nodes.iter().map(|&i| tree.nodes[i].height).collect();
        let prev_rem = tree.gen_rem(&prev_nodes);
        let next_h = h;

        let (match_count, groups, _) = match_greedy(&prev_h, &prev_rem, &next_h, W);
        let switch_count = prev_h.len() + next_h.len() - match_count * 2;
        score += switch_count as i64 * w;
        // tree.return_rem(&mut new_prev_rem); // 不要？

        let mut next_nodes = vec![];
        for i in 0..groups.len() {
            let (prev_g, next_g, rem) = groups[i];
            let prev_nodes = &node_idx[d][prev_g.0..prev_g.1].to_vec();
            let next_h = &next_h[next_g.0..next_g.1].to_vec();
            next_nodes.extend(tree.connect(prev_nodes, &next_h, rem));
        }

        node_idx[d + 1] = next_nodes;
    }

    eprintln!("{}", score);
}

#[derive(Clone)]
struct StackTree {
    nodes: Vec<Node>,
    root_node: usize,
}

impl StackTree {
    fn new(w: i64) -> StackTree {
        let mut tree = StackTree {
            root_node: 0,
            nodes: vec![],
        };
        tree.root_node = tree.new_node(0, w);
        tree
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
                rem_cand_node.push((self.nodes[v].node_r, v, self.nodes[v].rem));
            }
            for &u in self.nodes[v].parents.iter() {
                if !seen.insert(u) {
                    continue;
                }
                q.push(u);
            }
        }

        rem_cand_node.sort();
        let mut use_rem_nodes = HashMap::new();
        for (_, node_idx, rem) in rem_cand_node {
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

    fn gen_rem(&mut self, prev_nodes: &Vec<usize>) -> Vec<Vec<i64>> {
        // rem[i][j] := iから使い始められて、jまでには消費する必要がある余裕
        let mut rem = vec![vec![0; prev_nodes.len()]; prev_nodes.len()];

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

        // 余裕があるノードを全て拾い、remに保管する
        let mut q = vec![self.root_node];
        let mut seen = HashSet::new();
        seen.insert(self.root_node);
        while let Some(v) = q.pop() {
            rem[self.nodes[v].node_l][self.nodes[v].node_r] += self.nodes[v].rem;
            for &u in self.nodes[v].children.iter() {
                if !seen.insert(u) {
                    continue;
                }
                q.push(u);
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
