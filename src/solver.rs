use crate::def::*;
use crate::util::*;

const POOL_SIZE: usize = 1000;
const ROOT_NODE: usize = 0;

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
    let mut match_count = prev_h.len() + next_h.len();
    let mut groups = vec![];
    let mut next_rem = prev_rem.clone();

    let mut next_rem_sum = w - next_h.iter().sum::<i64>();
    let mut rems = vec![0; prev_h.len()];

    let mut prev_height = 0;
    let mut next_height = 0;
    let mut prev_l = 0;
    let mut next_l = 0;
    let mut prev_r = 0;
    let mut next_r = 0;
    while prev_r < prev_h.len() || next_r < next_h.len() {
        if prev_r < prev_h.len() && prev_height < next_height {
            // 使えるようになる余裕を回収する
            for i in prev_r..prev_h.len() {
                rems[i] += prev_rem[prev_r][i];
            }

            // 回収する
            prev_height += rems[prev_r];
            rems[prev_r] = 0;
            prev_r += 1;
        } else {
            next_height += next_h[next_r];
            next_r += 1;
        }

        if prev_height == 0 || next_height == 0 {
            continue;
        }

        if prev_height <= next_height && next_height <= prev_height + rems.iter().sum::<i64>() {
            // prevを合わせられるなら合わせる
            let mut use_prev_rem = next_height - prev_height;
            for i in 0..rems.len() {
                let r = rems[i].min(use_prev_rem);
                rems[i] -= r;
                use_prev_rem -= r;
            }
            groups.push(((prev_l, prev_r), (next_l, next_r), 0));
            prev_l = prev_r;
            next_l = next_r;
            match_count -= 2;
        } else if next_height <= prev_height && prev_height <= next_height + next_rem_sum {
            // nextを合わせられるなら合わせる
            let use_next_rem = prev_height - next_height;
            next_rem_sum -= use_next_rem;
            groups.push(((prev_l, prev_r), (next_l, next_r), use_next_rem));
            prev_l = prev_r;
            next_l = next_r;
            match_count -= 2;
        }
    }

    (match_count, groups, next_rem)
}

struct Solver<'a> {
    state: State,
    input: &'a Input,
}

impl<'a> Solver<'a> {
    fn new(ws: Vec<i64>, r: Vec<Vec<Vec<usize>>>, input: &Input) -> Solver {
        Solver {
            state: State::new(ws, r, POOL_SIZE, input.W),
            input,
        }
    }

    fn solve(&mut self) {
        for d in 0..self.input.D {
            self.state.setup_prev_rems(d, self.input);
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
                self.state.score += match_count as i64 * self.state.ws[col];
                self.state.trees[col].return_rem(&mut new_prev_rem);

                let mut next_nodes = vec![];
                for i in 0..groups.len() {
                    let (prev_g, next_g, rem) = groups[i];
                    let prev_nodes = &self.state.node_idx[d][col][prev_g.0..prev_g.1].to_vec();
                    let next_h = &next_h[next_g.0..next_g.1].to_vec();
                    next_nodes.extend(self.state.trees[col].connect(prev_nodes, &next_h, rem));
                }
                self.state.node_idx[col].push(next_nodes);
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
    prev_h: Vec<Vec<i64>>,
    prev_rem: Vec<Vec<Vec<i64>>>,
    score: i64,
}

impl State {
    fn new(ws: Vec<i64>, r: Vec<Vec<Vec<usize>>>, pool_size: usize, w: i64) -> State {
        let col = ws.len();
        let mut state = State {
            ws,
            r,
            trees: vec![StackTree::new(w); col],
            node_idx: vec![vec![]; col],
            prev_h: vec![vec![]; col],
            prev_rem: vec![vec![]; col],
            score: 0,
        };
        for i in 0..col {
            state.node_idx[i].push(vec![state.trees[i].root_node]);
        }
        state
    }

    fn setup_prev_rems(&mut self, d: usize, input: &Input) {
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
            rem[self.nodes[v].node_l][self.nodes[v].node_l] += self.nodes[v].rem;
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

#[derive(Clone)]
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
