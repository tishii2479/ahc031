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
    pub fn gen_range(low: usize, high: usize) -> usize {
        assert!(low < high);
        (next() % (high - low)) + low
    }

    #[inline]
    #[allow(unused)]
    pub fn gen_index(len: usize) -> usize {
        next() % len
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

/// 同じ値が存在していれば挿入せずにNoneを返す
/// 挿入できたらそのインデックスを返す
#[inline]
pub fn insert_sorted(v: &mut Vec<i64>, e: i64) -> Option<usize> {
    match v.binary_search(&e) {
        Ok(_) => None,
        Err(pos) => {
            v.insert(pos, e);
            Some(pos)
        }
    }
}

/// 値が存在しなければNoneを返す
/// 削除できたら削除した値を返す
#[inline]
pub fn remove_sorted(v: &mut Vec<i64>, e: i64) -> Option<i64> {
    match v.binary_search(&e) {
        Ok(pos) => Some(v.remove(pos)),
        Err(_) => None,
    }
}
