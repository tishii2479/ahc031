pub mod rnd {
    static mut S: usize = 88172645463325252;

    #[inline]
    #[allow(unused)]
    pub fn next() -> usize {
        unsafe {
            S = S ^ S << 7;
            S = S ^ S >> 9;
            S
        }
    }

    #[inline]
    #[allow(unused)]
    pub fn nextf() -> f64 {
        (next() & 4294967295) as f64 / 4294967296.
    }

    #[inline]
    #[allow(unused)]
    pub fn gen_rangef(low: f64, high: f64) -> f64 {
        nextf() * (high - low) + low
    }

    #[inline]
    #[allow(unused)]
    pub fn gen_range(low: usize, high: usize) -> usize {
        assert!(low < high);
        (next() % (high - low)) + low
    }

    #[inline]
    #[allow(unused)]
    pub fn gen_index(len: usize) -> usize {
        next() % len
    }

    #[allow(unused)]
    pub fn shuffle<I>(vec: &mut Vec<I>) {
        for i in 0..vec.len() {
            let j = gen_range(0, vec.len());
            vec.swap(i, j);
        }
    }
}

pub mod time {
    static mut START: f64 = -1.;

    #[allow(unused)]
    pub fn start_clock() {
        let _ = elapsed_seconds();
    }

    #[inline]
    #[allow(unused)]
    pub fn elapsed_seconds() -> f64 {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        unsafe {
            if START < 0. {
                START = t;
            }
            #[cfg(feature = "local")]
            {
                return (t - START) * 1.5;
            }
            t - START
        }
    }
}

#[inline]
pub fn ceil_div(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// aに最も近い要素のインデックスを返す
#[inline]
pub fn closest_index(a: usize, v: &Vec<usize>) -> usize {
    let mut i = 0;
    let n = v.len();
    for j in 0..n {
        if v[j].abs_diff(a) < v[i].abs_diff(a) {
            i = j;
        }
    }
    i
}
