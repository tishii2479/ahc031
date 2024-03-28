mod def;
mod solver;
mod util;

use crate::def::*;
use crate::util::*;

const FIRST_TIME_LIMIT: f64 = 1.;

fn create_col_and_initial_r(input: &Input) -> (Vec<i64>, Vec<Vec<Vec<usize>>>) {
    let mut ws = vec![200; 5];
    let mut r = vec![vec![vec![]; ws.len()]; input.D];

    // 初期状態の作成
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
            for &i in rs {
                height += ceil_div(input.A[d][i], ws[col]);
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

    while time::elapsed_seconds() < FIRST_TIME_LIMIT {
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
    eprintln!("ws:          {:?}", ws);

    (ws, r)
}

fn greedy(
    prev_r: &Vec<usize>,
    prev_h: &Vec<i64>,
    prev_rem: &Vec<i64>,
    prev_g: &Vec<usize>,
    next_r: &Vec<usize>,
    h: i64,
    a: &Vec<Vec<i64>>,
    d: usize,
    w: i64,
) -> (Vec<i64>, Vec<i64>, Vec<usize>, usize) {
    let mut next_h = Vec::with_capacity(next_r.len());
    let mut next_rem = Vec::with_capacity(next_r.len());
    let mut next_g = Vec::with_capacity(next_r.len());

    // 切り替え回数
    // 重複した仕切りを使うと、切り替え回数を減らせる
    let mut switch_count = prev_r.len() + next_r.len() - 2;

    let mut prev_i = 0;
    let mut next_i = 0;
    let mut cur_prev_h = 0;
    let mut cur_next_h = 0;

    let mut cur_prev_g = prev_g[prev_i];
    let mut cur_next_g = 0;
    let mut cur_prev_rem = prev_rem[cur_prev_g];
    let mut cur_next_rem = h - next_r.iter().map(|&i| ceil_div(a[d][i], w)).sum::<i64>();

    while prev_i < prev_r.len() || next_i < next_r.len() {
        if cur_prev_h <= cur_next_h {
            if prev_i < prev_r.len() {
                cur_prev_h += prev_h[prev_i];
                prev_i += 1;
            }
            if prev_i < prev_g.len() && cur_prev_g != prev_g[prev_i] {
                // グループが変わった
                cur_prev_h += cur_prev_rem;
                cur_prev_g = prev_g[prev_i];
                cur_prev_rem = prev_rem[cur_prev_g];
            }
        } else {
            if next_i < next_r.len() {
                let h = ceil_div(a[d][next_r[next_i]], w);
                next_h.push(h);
                cur_next_h += h;
                next_i += 1;
            }
        }

        if cur_prev_h <= cur_next_h
            && cur_prev_h + cur_prev_rem >= cur_next_h
            && !(prev_i + 1 < prev_h.len() && cur_prev_h + prev_h[prev_i + 1] <= cur_next_h)
        {
            // 次の領域を消費してもnextに届かない場合はスルーする
            // cur_prev_remを消費してnextに合わせる場合
            let use_prev_rem = cur_next_h - cur_prev_h;
            while next_g.len() < next_i {
                next_g.push(cur_next_g);
            }
            next_rem.push(0);
            cur_next_g += 1;
            cur_prev_h += use_prev_rem;
            cur_prev_rem -= use_prev_rem;
            switch_count -= 2;
        } else if cur_next_h <= cur_prev_h
            && cur_next_h + cur_next_rem >= cur_prev_h
            && !(next_i + 1 < next_h.len() && cur_next_h + next_h[next_i + 1] <= cur_prev_h)
            && next_g.len() < next_i
        {
            // 次の領域を消費してもprevに届かない場合はスルーする
            // cur_next_remを消費してprevに合わせる場合
            let use_next_rem = cur_prev_h - cur_next_h;
            *next_h.last_mut().unwrap() += use_next_rem;
            while next_g.len() < next_i {
                next_g.push(cur_next_g);
            }
            next_rem.push(cur_next_rem);
            cur_next_g += 1;
            cur_next_h += use_next_rem;
            cur_next_rem -= use_next_rem;
            switch_count -= 2;
        }
    }
    if next_g.len() < next_i {
        // 最後のグループがまとまっていなければ残りを追加する
        next_rem.push(cur_next_rem);
    } else {
        // 最後のグループがまとまっていれば残りの余裕を加算する
        *next_rem.last_mut().unwrap() += cur_next_rem;
    }
    while next_g.len() < next_i {
        next_g.push(cur_next_g);
    }

    (next_h, next_rem, next_g, switch_count)
}

struct Node {
    is_joint: bool,
    val: i64,             // joint => rem, leaf => height
    child: Option<usize>, // joint => first-child, leaf => next-child
}

struct ColMatcher {
    node_pool: Vec<Node>,
    unused: Vec<usize>,
    cur_prev_height: i64,
    cur_next_height: i64,
    cur_next_rem: i64,
    cur_rem: Vec<i64>,
    cur_children: Vec<usize>,
    next_i: usize,
}

impl ColMatcher {
    fn new() -> ColMatcher {
        ColMatcher {
            node_pool: vec![],
            unused: vec![],
            cur_prev_height: 0,
            cur_next_height: 0,
            cur_next_rem: 0,
            cur_rem: vec![],
            cur_children: vec![],
            next_i: 0,
        }
    }

    fn f(&mut self, prev_root_node_idx: usize, next_h: &Vec<i64>, input: &Input) -> usize {
        self.cur_prev_height = 0;
        self.cur_next_height = 0;
        self.cur_next_rem = input.W - next_h.iter().sum::<i64>();
        self.cur_rem.clear();
        self.cur_children.clear();
        self.next_i = 0;
        self.dfs(prev_root_node_idx, next_h);
        prev_root_node_idx
    }

    fn dfs(&mut self, node_idx: usize, next_h: &Vec<i64>) {
        let node = &self.node_pool[node_idx];
        if node.is_joint {
            self.cur_rem.push(node.val);
            self.dfs(node.child.unwrap(), next_h);
            self.cur_prev_height += self.cur_rem.pop().unwrap();
            self.merge_if_possible(next_h);
        } else {
            self.cur_prev_height += node.val;
            if let Some(next_child) = node.child {
                self.merge_if_possible(next_h);
                self.dfs(next_child, next_h);
            } else {
                return;
            }
        }
    }

    fn new_node(&mut self, is_joint: bool, val: i64, child: Option<usize>) -> usize {
        if let Some(i) = self.unused.pop() {
            let node = &mut self.node_pool[i];
            node.is_joint = is_joint;
            node.val = val;
            node.child = child;
            return i;
        }
        self.node_pool.push(Node {
            is_joint,
            val,
            child,
        });
        self.node_pool.len() - 1
    }

    fn merge_if_possible(&mut self, next_h: &Vec<i64>) {
        let rem = self.cur_rem.iter().sum::<i64>();
        while self.next_i < next_h.len()
            && self.cur_next_height + next_h[self.next_i] < self.cur_prev_height
        {
            let new_node_idx = self.new_node(false, next_h[self.next_i], None);
            self.cur_children.push(new_node_idx);
            self.cur_next_height += next_h[self.next_i];
            self.next_i += 1;
        }

        if self.cur_prev_height <= self.cur_next_height
            && self.cur_prev_height + rem >= self.cur_next_height
        {
            // prevをnextに合わせる
            let mut use_rem = self.cur_next_height - self.cur_prev_height;
            let mut i = self.cur_rem.len() - 1;
            while use_rem > 0 {
                let c = use_rem.min(self.cur_rem[i]);
                self.cur_rem[i] -= c;
                use_rem -= c;
                i -= 1;
            }
            self.cur_prev_height = self.cur_next_height;
        } else if self.cur_next_height <= self.cur_prev_height
            && self.cur_next_height + self.cur_next_rem >= self.cur_prev_height
        {
            // nextをprevに合わせる
        }
    }
}

fn optimize_r(ws: Vec<i64>, mut r: Vec<Vec<Vec<usize>>>, input: &Input) {
    // 重さ順にソートしておく
    for d in 0..input.D {
        for rs in r[d].iter_mut() {
            rs.sort();
        }
    }

    let mut h = vec![vec![vec![]; ws.len()]; input.D];
    let mut rem = vec![vec![vec![]; ws.len()]; input.D];
    let mut g = vec![vec![vec![]; ws.len()]; input.D];
    for col in 0..ws.len() {
        h[0][col] = r[0][col]
            .iter()
            .map(|&i| ceil_div(input.A[0][i], ws[col]))
            .collect();
        let col_rem = input.W - h[0][col].iter().sum::<i64>();
        rem[0][col].push(col_rem);
        g[0][col] = vec![0; r[0][col].len()];
    }

    fn eval(
        d: usize,
        h: &Vec<Vec<Vec<i64>>>,
        rem: &Vec<Vec<Vec<i64>>>,
        r: &Vec<Vec<Vec<usize>>>,
        g: &Vec<Vec<Vec<usize>>>,
        ws: &Vec<i64>,
        input: &Input,
    ) -> i64 {
        let mut score = 0;
        for col in 0..ws.len() {
            let w = ws[col];
            let (next_h, next_rem, next_g, switch_count) = greedy(
                &r[d - 1][col],
                &h[d - 1][col],
                &rem[d - 1][col],
                &g[d - 1][col],
                &r[d][col],
                input.W,
                &input.A,
                d,
                w,
            );
            score += ws[col] * switch_count as i64;
        }
        score
    }

    let cur_score = eval(0, &h, &rem, &r, &g, &ws, input);

    for d in 1..input.D {
        for _ in 0..10000 {
            let p = rnd::nextf();
            if p < 0.2 {
                // 列内の入れ替え
                let col = rnd::gen_index(ws.len());
                let (i, j) = (
                    rnd::gen_index(r[d][col].len()),
                    rnd::gen_index(r[d][col].len()),
                );
                if i == j {
                    continue;
                }
                r[d][col].swap(i, j);
            }
        }
    }
}

fn solve(input: &Input) -> Answer {
    let (ws, r) = create_col_and_initial_r(input);
    // let ans = optimize_r(ws, r, input);

    let mut ans = Answer::new(input.D, input.N);
    for d in 0..input.D {
        let mut width = 0;
        for (col, rs) in r[d].iter().enumerate() {
            let mut height = 0;
            for &e in rs {
                let h = ceil_div(input.A[d][e], ws[col]);
                ans.p[d][e] = (height, width, height + h, width + ws[col]);
                height += h;
            }
            width += ws[col];
            eprint!("{:4}", height);
        }
        eprintln!();
    }

    ans
}

fn main() {
    time::start_clock();
    let input = Input::read_input();
    // let ans = solve(&input);
    // ans.output();

    let d = 1;
    let w = 100;
    let prev_r = vec![1, 2, 3, 4];
    let prev_h = vec![11666, 16167, 16717, 46095];
    let prev_h: Vec<i64> = prev_h.iter().map(|&s| ceil_div(s, w)).collect();
    let prev_rem = vec![input.W - prev_h.iter().sum::<i64>()];
    let prev_g = vec![0, 0, 0, 0];
    let next_r = vec![0, 1, 2, 3];
    let (next_h, next_rem, next_g, switch_count) = greedy(
        &prev_r, &prev_h, &prev_rem, &prev_g, &next_r, input.W, &input.A, d, w,
    );
    dbg!(next_h, next_rem, next_g, switch_count);
}
