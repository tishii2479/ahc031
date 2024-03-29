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

        if prev_l == prev_r || next_l == next_r {
            continue;
        }
        if (prev_r == prev_h.len() && next_r < next_h.len())
            || (prev_r < prev_h.len() && next_r == next_h.len())
        {
            continue;
        }

        eprintln!(
            "a: pl: {}, pr: {}, nl: {}, nr: {}, ph: {}, nh: {}, pr_sum: {}, nr_sum: {}",
            prev_l, prev_r, next_l, next_r, prev_height, next_height, prev_rem_sum, next_rem_sum
        );
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

    let prev_h = vec![20, 30, 40];
    let prev_rem = create_rem_mat(prev_h.len(), vec![(0, 2, 10)]);
    let next_h = vec![10];
    let (match_count, groups, new_prev_rem) = match_greedy(&prev_h, &prev_rem, &next_h, W);
    assert_eq!(match_count, 1);
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

    pub fn solve(&mut self) {
        for d in 0..self.input.D {
            let col = 0;
            let next_h: Vec<i64> = self.state.r[d][col]
                .iter()
                .map(|&r_idx| ceil_div(self.input.A[d][r_idx], self.state.ws[col]))
                .collect();
            eprintln!("{:?}", next_h);
        }
        for d in 0..self.input.D {
            self.state.setup_prev_rems(d);
            for col in 0..self.state.ws.len() {
                let next_h = self.state.r[d][col]
                    .iter()
                    .map(|&r_idx| ceil_div(self.input.A[d][r_idx], self.state.ws[col]))
                    .collect();
                let (match_count, groups, mut new_prev_rem) = match_greedy(
                    &self.state.prev_h[col],
                    &self.state.prev_rem[col],
                    &next_h,
                    self.input.W,
                );
                let switch_count = self.state.prev_h[col].len() + next_h.len() - match_count * 2;
                self.state.score += switch_count as i64 * self.state.ws[col];
                self.state.trees[col].return_rem(&mut new_prev_rem);

                let mut next_nodes = vec![];
                for i in 0..groups.len() {
                    let (prev_g, next_g, rem) = groups[i];
                    let prev_nodes = &self.state.node_idx[d][col][prev_g.0..prev_g.1].to_vec();
                    let next_h = &next_h[next_g.0..next_g.1].to_vec();
                    next_nodes.extend(self.state.trees[col].connect(prev_nodes, &next_h, rem));
                }
                assert_eq!(
                    next_nodes.len(),
                    self.state.r[d][col].len(),
                    "{:?} {:?} {:?}",
                    groups,
                    &self.state.prev_h[col],
                    next_h
                );
                self.state.node_idx[d + 1].push(next_nodes);
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
    // prev_h[col][i]
    prev_h: Vec<Vec<i64>>,
    // prev_h[col][l][r]
    prev_rem: Vec<Vec<Vec<i64>>>,
    score: i64,
}

impl State {
    fn new(ws: Vec<i64>, r: Vec<Vec<Vec<usize>>>, w: i64, d: usize) -> State {
        let col_count = ws.len();
        let mut state = State {
            ws,
            r,
            trees: vec![StackTree::new(w); col_count],
            node_idx: vec![vec![]; d + 1],
            prev_h: vec![vec![]; col_count],
            prev_rem: vec![vec![]; col_count],
            score: 0,
        };
        for col in 0..col_count {
            state.node_idx[0].push(vec![state.trees[col].root_node]);
        }
        state
    }

    fn setup_prev_rems(&mut self, d: usize) {
        for col in 0..self.ws.len() {
            let prev_nodes = &self.node_idx[d][col];
            let prev_h: Vec<i64> = prev_nodes
                .iter()
                .map(|&i| self.trees[col].nodes[i].height)
                .collect();
            let prev_rem = self.trees[col].gen_rem(&prev_nodes);
            self.prev_h[col] = prev_h;
            self.prev_rem[col] = prev_rem;
        }
    }
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
        node._rem = rem;
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
        dbg!(&prev_nodes, &next_heights, rem);
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

        next_nodes
    }

    fn gen_rem(&mut self, prev_nodes: &Vec<usize>) -> Vec<Vec<i64>> {
        // ISSUE: めっちゃ遅い

        // rem[i][j] := iから使い始められて、jまでには消費する必要がある余裕
        let mut rem = vec![vec![0; prev_nodes.len()]; prev_nodes.len()];

        // node_l、node_rをリセットする
        let mut q = vec![self.root_node];
        while let Some(v) = q.pop() {
            self.nodes[v].node_l = usize::MAX;
            self.nodes[v].node_r = usize::MIN;
            for &u in self.nodes[v].children.iter() {
                q.push(u);
            }
        }

        // node_idxから上方向に昇って、node_l、node_rを設定する
        for (i, &node_idx) in prev_nodes.iter().enumerate() {
            let mut q = vec![node_idx];
            while let Some(v) = q.pop() {
                // dbg!(&v, &self.nodes[v]);
                self.nodes[v].node_l = self.nodes[v].node_l.min(i);
                self.nodes[v].node_r = self.nodes[v].node_r.max(i);
                for &u in self.nodes[v].parents.iter() {
                    q.push(u);
                }
            }
        }

        // 余裕があるノードを全て拾い、remに保管する
        let mut q = vec![self.root_node];
        while let Some(v) = q.pop() {
            // dbg!(&v, &self.nodes[v]);
            rem[self.nodes[v].node_l][self.nodes[v].node_r] += self.nodes[v].rem;
            self.nodes[v].rem = 0;
            for &u in self.nodes[v].children.iter() {
                q.push(u);
            }
        }

        rem
    }

    fn return_rem(&mut self, rem: &mut Vec<Vec<i64>>) {
        let mut q = vec![self.root_node];
        while let Some(v) = q.pop() {
            let add_rem = rem[self.nodes[v].node_l][self.nodes[v].node_r].min(self.nodes[v]._rem);
            self.nodes[v].rem += add_rem;
            rem[self.nodes[v].node_l][self.nodes[v].node_r] -= add_rem;
            for &u in self.nodes[v].children.iter() {
                q.push(u);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Node {
    parents: Vec<usize>,  // 親ノード
    children: Vec<usize>, // 子ノード
    _rem: i64,            // 余裕（保管用）
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
            _rem: 0,
            rem: 0,
            height: 0,
            node_l: usize::MAX,
            node_r: usize::MIN,
        }
    }
}
