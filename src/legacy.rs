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
