fn solve(input: &Input) -> Answer {
    let bins = create_bins(input);
    let mut ans = Answer::new(input.D, input.N);

    for d in 0..input.D {
        let mut height = vec![0; bins.len() - 1];
        for i in (0..input.N).rev() {
            let mut best_remain = 1 << 30;
            let mut best_j = !0;
            for j in 0..bins.len() - 1 {
                if height[j] >= input.W {
                    continue;
                }
                let w = bins[j + 1] - bins[j];
                let remain = w * (input.W - height[j]);
                if input.A[d][i] <= remain && remain < best_remain {
                    best_remain = remain;
                    best_j = j;
                }
            }

            assert!(best_j < bins.len() - 1, "{} {} {:?}", d, i, height);
            let w = bins[best_j + 1] - bins[best_j];
            let h = (input.A[d][i] + w - 1) / w;
            ans.p[d][i] = (
                height[best_j],
                bins[best_j],
                height[best_j] + h,
                bins[best_j + 1],
            );
            height[best_j] += h;
        }
    }

    ans
}

fn to_ans(bins: &Vec<i64>, slides: &Vec<Vec<i64>>, ops: &Vec<Ops>, input: &Input) -> Answer {
    let mut ans = Answer::new(input.D, input.N);
    let mut slides = slides.clone();

    let mut v = Vec::with_capacity(input.N);
    for d in 0..input.D {
        v.clear();
        for &op in ops[d].del.iter() {
            assert!(remove_sorted(&mut slides[op.bin], op.p).is_some());
        }
        for &op in ops[d].add.iter() {
            assert!(insert_sorted(&mut slides[op.bin], op.p).is_some());
        }
        for i in 0..bins.len() - 1 {
            let w = bins[i + 1] - bins[i];
            for j in 0..slides[i].len() - 1 {
                let h = slides[i][j + 1] - slides[i][j];
                v.push((
                    w * h,
                    (slides[i][j], bins[i], slides[i][j + 1], bins[i + 1]),
                ));
            }
        }
        v.sort();
        for i in 0..input.N {
            ans.p[d][i] = v[i].1;
        }
    }
    ans
}

fn create_bins(input: &Input) -> Vec<i64> {
    vec![0, 400, 700, 800, 900, 950, input.W]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Op {
    bin: usize,
    p: i64,
}

#[derive(Clone, Debug)]
struct Ops {
    add: Vec<Op>,
    del: Vec<Op>,
}

fn eval(bins: &Vec<i64>, slides: &Vec<Vec<i64>>, ops: &Vec<Ops>, input: &Input) -> i64 {
    let mut score = 0;
    let mut slides = slides.clone();
    let mut v = Vec::with_capacity(input.N);

    for d in 0..input.D {
        for &op in ops[d].del.iter() {
            // 操作によって、重複した操作が発生していれば中止する
            if remove_sorted(&mut slides[op.bin], op.p).is_none() {
                return 1 << 30;
            }
        }
        for &op in ops[d].add.iter() {
            // 操作によって、重複した操作が発生していれば中止する
            if insert_sorted(&mut slides[op.bin], op.p).is_none() {
                return 1 << 30;
            }
        }

        // O(D N log N)
        // NOTE: vを保持すればsortは消せる
        // NOTE: maxを保持していれば、違反の検証はできる
        v.clear();
        for i in 0..bins.len() - 1 {
            let w = bins[i + 1] - bins[i];
            for j in 0..slides[i].len() - 1 {
                let h = slides[i][j + 1] - slides[i][j];
                v.push((w * h) as i64);
            }
        }
        v.sort();
        for i in 0..input.N {
            score += (input.A[d][i] - v[i]).max(0) * 100;
        }
    }
    score
}

fn solve(input: &Input) -> Answer {
    let mut bins = create_bins(input);
    let mut ops = vec![
        Ops {
            add: vec![],
            del: vec![]
        };
        input.D
    ];
    let mut slides = vec![vec![0, input.W]; bins.len() - 1];

    let mut rect_count = bins.len() - 1;
    while rect_count < input.N {
        let bin = rnd::gen_index(bins.len() - 1);
        let y = rnd::gen_range(1, input.W as usize) as i64;
        if insert_sorted(&mut slides[bin], y).is_some() {
            rect_count += 1;
        }
    }

    let mut cur_score = eval(&bins, &slides, &ops, input);
    let mut iteration = 0;

    'action: while elapsed_seconds() < TIME_LIMIT {
        let p = rnd::nextf();
        if p < 0.5 {
            let op_d = rnd::gen_range(1, input.D);
            let mut new_score = 0;
            let mut slides = slides.clone();
            let mut v = Vec::with_capacity(input.N);
            let mut op = None;
            for d in 0..input.D {
                for &op in ops[d].del.iter() {
                    // 操作によって、重複した操作が発生していれば中止する
                    if remove_sorted(&mut slides[op.bin], op.p).is_none() {
                        continue 'action;
                    }
                }
                for &op in ops[d].add.iter() {
                    // 操作によって、重複した操作が発生していれば中止する
                    if insert_sorted(&mut slides[op.bin], op.p).is_none() {
                        continue 'action;
                    }
                }
                if d == op_d {
                    op = {
                        let (bin_add, bin_del) = (
                            rnd::gen_index(bins.len() - 1),
                            rnd::gen_index(bins.len() - 1),
                        );
                        if slides[bin_del].len() == 2 {
                            // 削除する仕切りがないか、同じビンを指していれば中止
                            continue 'action;
                        }
                        let slide_add_p = rnd::gen_range(1, input.W as usize) as i64;
                        let slide_del_p =
                            slides[bin_del][rnd::gen_range(1, slides[bin_del].len() - 1)];
                        if remove_sorted(&mut slides[bin_del], slide_del_p).is_none() {
                            continue 'action;
                        }
                        if insert_sorted(&mut slides[bin_add], slide_add_p).is_none() {
                            // slide_addがslides[bin_add]に既に存在していれば中止
                            continue 'action;
                        };
                        Some((
                            Op {
                                bin: bin_add,
                                p: slide_add_p,
                            },
                            Op {
                                bin: bin_del,
                                p: slide_del_p,
                            },
                        ))
                    };
                }

                // O(D N log N)
                // NOTE: vを保持すればsortは消せる
                // NOTE: maxを保持していれば、違反の検証はできる
                v.clear();
                for i in 0..bins.len() - 1 {
                    let w = bins[i + 1] - bins[i];
                    for j in 0..slides[i].len() - 1 {
                        let h = slides[i][j + 1] - slides[i][j];
                        v.push((w * h) as i64);
                    }
                }
                v.sort();
                if v.len() < input.N {
                    dbg!(&slides, &ops, &ops[d], op, op_d, d);
                }
                for i in 0..input.N {
                    new_score += (input.A[d][i] - v[i]).max(0) * 100;
                }
            }
            if new_score < cur_score {
                // 採用
                eprintln!(
                    "op: {} -> {}, d: {}, op: {:?}",
                    cur_score, new_score, op_d, op
                );
                cur_score = new_score;
                if let Some((op_add, op_del)) = op {
                    if let Some(i) = ops[op_d].add.iter().position(|x| x == &op_del) {
                        ops[op_d].add.remove(i);
                        ops[op_d].add.push(op_add);
                    } else if let Some(i) = ops[op_d].del.iter().position(|x| x == &op_add) {
                        ops[op_d].del.remove(i);
                        ops[op_d].del.push(op_del);
                    } else {
                        ops[op_d].add.push(op_add);
                        ops[op_d].del.push(op_del);
                    }
                }
            } else {
                // ロールバック
            }
        } else if p < 0.7 {
            let (bin_add, bin_del) = (
                rnd::gen_index(bins.len() - 1),
                rnd::gen_index(bins.len() - 1),
            );
            if slides[bin_del].len() == 2 {
                // 削除する仕切りがないか、同じビンを指していれば中止
                continue 'action;
            }
            let slide_add_p = rnd::gen_range(1, input.W as usize) as i64;
            let slide_del_p = slides[bin_del][rnd::gen_range(1, slides[bin_del].len() - 1)];
            if insert_sorted(&mut slides[bin_add], slide_add_p).is_none() {
                // slide_addがslides[bin_add]に既に存在していれば中止
                continue 'action;
            };
            remove_sorted(&mut slides[bin_del], slide_del_p);
            let new_score = eval(&bins, &slides, &ops, input);
            if new_score < cur_score {
                // 採用
                eprintln!("sli: {} -> {}", cur_score, new_score);
                cur_score = new_score;
            } else {
                // ロールバック
                remove_sorted(&mut slides[bin_add], slide_add_p);
                insert_sorted(&mut slides[bin_del], slide_del_p);
            }
        } else {
            let bin = rnd::gen_range(1, bins.len() - 1);
            let prev_bin_p = bins[bin];
            let bin_p = rnd::gen_range(bins[bin - 1] as usize + 1, bins[bin + 1] as usize) as i64;
            bins[bin] = bin_p;
            let new_score = eval(&bins, &slides, &ops, input);
            if new_score < cur_score {
                // 採用
                eprintln!(
                    "bin: {} -> {}, bin: {}, {} -> {}",
                    cur_score, new_score, bin, prev_bin_p, bin_p
                );
                cur_score = new_score;
            } else {
                // ロールバック
                bins[bin] = prev_bin_p;
            }
        }
        iteration += 1;
    }

    eprintln!("iteration =  {}", iteration);

    to_ans(&bins, &slides, &ops, input)
}

fn create_bins(input: &Input) -> Vec<i64> {
    let mut ws = vec![];
    for d in 0..input.D {
        let mut height = vec![0; ws.len()];
        for i in (0..input.N).rev() {
            eprintln!("ws:      {:?}", ws);
            eprintln!("height:  {:?}", height);
            let a = input.A[d][i];
            {
                let mut best_j = !0;
                let mut best_rem = 1 << 30;
                for (j, w) in ws.iter().enumerate() {
                    let h = (a + w - 1) / w;
                    let rem = (input.W - height[j] - h) * w;
                    if rem >= 0 && rem <= best_rem {
                        best_j = j;
                        best_rem = rem;
                    }
                }
                // 残りが最も少ない列（ギリギリ入る列）に入れる
                if best_j < ws.len() {
                    height[best_j] += ceil_div(a, ws[best_j]);
                    continue;
                }
            }
            assert!(
                height.iter().map(|h| (h > &input.W) as i64).sum::<i64>() == 0,
                "{:?}",
                height
            );

            // 入らない場合
            {
                // 列の幅を調整するか、新しい列を作る
                let mut best_j = !0;
                let mut best_w = 0;
                for (j, &w) in ws.iter().enumerate() {
                    let h = (a + w - 1) / w;
                    if w <= best_w {
                        continue;
                    }
                    if height[j] == 0 {
                        // 空の列がある場合は、最も幅が大きいところに入れる
                        best_j = j;
                        best_w = w;
                    } else if height[j] + h < input.W + 50 {
                        // もう少しで入るところがあれば、幅を広げて入れる
                        best_j = j;
                        best_w = w;
                    }
                }
                if best_j < ws.len() {
                    let total_s = height[best_j] * ws[best_j] + a;
                    let new_w = ceil_div(total_s, input.W);
                    ws[best_j] = new_w;
                    height[best_j] = ceil_div(total_s, new_w);
                } else {
                    let new_w = ceil_div(a, input.W).max(100);
                    ws.push(new_w);
                    height.push(ceil_div(a, new_w));
                }
            }
        }
        eprintln!("ws:      {:?}", ws);
        eprintln!("height:  {:?}", height);
        eprintln!();
    }
    ws.sort_by(|a, b| b.cmp(a));
    eprintln!("ws: {:?}", ws);

    let mut bins = vec![0];
    for w in ws {
        bins.push(bins.last().unwrap() + w);
    }
    bins
}

struct NodePool {
    stack: RefCell<Vec<Node>>,
}

impl NodePool {
    fn new(size: usize) -> NodePool {
        let stack = vec![Node::new(); size];
        NodePool {
            stack: RefCell::new(stack),
        }
    }

    fn get(&mut self) -> NodeHandle {
        if let Some(mut node) = self.stack.borrow_mut().pop() {
            node.init();
            return NodeHandle {
                stack: &self.stack,
                node: Some(node),
            };
        }
        let node = Node::new();
        NodeHandle {
            stack: &self.stack,
            node: Some(node),
        }
    }
}

struct NodeHandle<'a> {
    stack: &'a RefCell<Vec<Node>>,
    node: Option<Node>,
}

impl<'a> NodeHandle<'a> {
    fn drop(&mut self) {
        let node = self.node.take().unwrap();
        self.stack.borrow_mut().push(node);
    }
}

// impl<'a> ops::Deref for NodeHandle<'a> {
//     type Target = Node;
//     fn deref(&self) -> &Node {
//         self.node.as_ref().unwrap()
//     }
// }

// impl<'a> ops::DerefMut for NodeHandle<'a> {
//     fn deref_mut(&mut self) -> &mut Node {
//         self.node.as_mut().unwrap()
//     }
// }

#[derive(Clone)]
struct Node {
    par: Option<usize>,
    children: Vec<Node>,
    height: i64,
    _rem: i64,
    cur_rem: i64,
}

impl Node {
    fn new() -> Node {
        Node {
            par: None,
            children: vec![],
            height: 0,
            _rem: 0,
            cur_rem: 0,
        }
    }

    fn init(&mut self) {
        self.par = None;
        self.children.clear();
        self.height = 0;
        self._rem = 0;
        self.cur_rem = 0;
    }
}
