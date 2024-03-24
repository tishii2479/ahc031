mod util;

use crate::util::*;
use proconio::input;
use util::time::elapsed_seconds;

const TIME_LIMIT: f64 = 2.9;

#[allow(non_snake_case)]
struct Input {
    W: i64,
    D: usize,
    N: usize,
    A: Vec<Vec<i64>>,
}

impl Input {
    #[allow(non_snake_case)]
    fn read_input() -> Input {
        input! {
            W: i64, D: usize, N: usize,
            A: [[i64; N]; D]
        }
        Input { W, D, N, A }
    }
}

struct Answer {
    p: Vec<Vec<(i64, i64, i64, i64)>>,
}

impl Answer {
    fn new(d: usize, n: usize) -> Answer {
        Answer {
            p: vec![vec![(0, 0, 0, 0); n]; d],
        }
    }

    fn output(&self) {
        for vec in self.p.iter() {
            for p in vec.iter() {
                println!("{} {} {} {}", p.0, p.1, p.2, p.3);
            }
        }
    }
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
                        if slides[bin_del].len() == 2 || bin_add == bin_del {
                            // 削除する仕切りがないか、同じビンを指していれば中止
                            None
                        } else {
                            let slide_add_p = rnd::gen_range(1, input.W as usize) as i64;
                            let slide_del_idx = rnd::gen_range(1, slides[bin_del].len() - 1);
                            if insert_sorted(&mut slides[bin_add], slide_add_p).is_none() {
                                // slide_addがslides[bin_add]に既に存在していれば中止
                                None
                            } else {
                                let slide_del_p = slides[bin_del].remove(slide_del_idx);
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
                            }
                        }
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
                eprintln!("op: {} -> {} : {} {:?}", cur_score, new_score, op_d, op);
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
            if slides[bin_del].len() == 2 || bin_add == bin_del {
                // 削除する仕切りがないか、同じビンを指していれば中止
                continue;
            }
            let slide_add_p = rnd::gen_range(1, input.W as usize) as i64;
            let slide_del_idx = rnd::gen_range(1, slides[bin_del].len() - 1);
            if insert_sorted(&mut slides[bin_add], slide_add_p).is_none() {
                // slide_addがslides[bin_add]に既に存在していれば中止
                continue;
            };
            let slide_del_p = slides[bin_del].remove(slide_del_idx);
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
                eprintln!("bin: {} -> {}", cur_score, new_score);
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

fn main() {
    let input = Input::read_input();
    let ans = solve(&input);
    ans.output();
}
