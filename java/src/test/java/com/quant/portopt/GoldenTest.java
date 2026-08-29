package com.quant.portopt;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;
import org.junit.Test;

/**
 * Golden-value suite: recomputes every case of ../data/golden/golden.json from
 * the bundled data and asserts agreement within each case's tolerance — the
 * same contract every language port must satisfy.
 */
public class GoldenTest {

    private static void assertVector(String caseName, String prefix, double[] got) {
        double[] want = TestData.expectVector(caseName, prefix);
        double tol = TestData.tol(caseName);
        assertEquals(caseName + ": vector length", want.length, got.length);
        for (int i = 0; i < want.length; i++) {
            assertEquals(caseName + ": " + prefix + "_" + i, want[i], got[i], tol);
        }
    }

    @Test
    public void goldenCaseCount() {
        assertTrue("expected at least 20 golden cases", TestData.golden().size() >= 20);
    }

    @Test
    public void sampleMomentsDaily() {
        String c = "sample_moments_daily";
        double tol = TestData.tol(c);
        double[] mu = Estimators.sampleMean(TestData.returns());
        double[][] cov = Estimators.sampleCov(TestData.returns());
        assertEquals(TestData.expectScalar(c, "mean_0"), mu[0], tol);
        assertEquals(TestData.expectScalar(c, "mean_4"), mu[4], tol);
        assertEquals(TestData.expectScalar(c, "mean_7"), mu[7], tol);
        assertEquals(TestData.expectScalar(c, "cov_0_0"), cov[0][0], tol);
        assertEquals(TestData.expectScalar(c, "cov_0_1"), cov[0][1], tol);
        assertEquals(TestData.expectScalar(c, "cov_4_4"), cov[4][4], tol);
        assertEquals(TestData.expectScalar(c, "cov_6_7"), cov[6][7], tol);
    }

    @Test
    public void shrinkageConstCorrDaily() {
        String c = "shrinkage_const_corr_daily";
        double tol = TestData.tol(c);
        Estimators.ShrinkageResult res = Estimators.shrinkCovariance(TestData.returns());
        assertEquals(TestData.expectScalar(c, "intensity"), res.intensity(), tol);
        assertEquals(TestData.expectScalar(c, "rbar"), res.rbar(), tol);
        assertEquals(TestData.expectScalar(c, "shrunk_0_0"), res.sigma()[0][0], tol);
        assertEquals(TestData.expectScalar(c, "shrunk_0_1"), res.sigma()[0][1], tol);
        assertEquals(TestData.expectScalar(c, "shrunk_2_5"), res.sigma()[2][5], tol);
        assertEquals(TestData.expectScalar(c, "shrunk_7_7"), res.sigma()[7][7], tol);
    }

    @Test
    public void twoFundScalarsAnnualized() {
        String c = "two_fund_scalars_annualized";
        double tol = TestData.tol(c);
        Markowitz.TwoFundScalars sc =
                Markowitz.twoFundScalars(TestData.muAnnualized(), TestData.sigmaAnnualized());
        assertEquals(TestData.expectScalar(c, "a"), sc.a(), tol);
        assertEquals(TestData.expectScalar(c, "b"), sc.b(), tol);
        assertEquals(TestData.expectScalar(c, "c"), sc.c(), tol);
        assertEquals(TestData.expectScalar(c, "d"), sc.d(), tol);
    }

    @Test
    public void minvarUnconstrained() {
        String c = "minvar_unconstrained";
        double tol = TestData.tol(c);
        double[] w = Markowitz.minVarianceWeights(TestData.sigmaAnnualized());
        assertVector(c, "w", w);
        double ret = Linalg.dot(TestData.muAnnualized(), w);
        double vol = Math.sqrt(Linalg.quadForm(w, TestData.sigmaAnnualized()));
        assertEquals(TestData.expectScalar(c, "ret"), ret, tol);
        assertEquals(TestData.expectScalar(c, "vol"), vol, tol);
    }

    @Test
    public void tangencyUnconstrained() {
        String c = "tangency_unconstrained";
        double tol = TestData.tol(c);
        double rf = TestData.inputScalar(c, "rf");
        double[] w = Markowitz.tangencyWeights(TestData.muAnnualized(),
                TestData.sigmaAnnualized(), rf);
        assertVector(c, "w", w);
        double sharpe = (Linalg.dot(TestData.muAnnualized(), w) - rf)
                / Math.sqrt(Linalg.quadForm(w, TestData.sigmaAnnualized()));
        assertEquals(TestData.expectScalar(c, "sharpe"), sharpe, tol);
    }

    @Test
    public void frontierPointT080() {
        String c = "frontier_point_t080";
        double tol = TestData.tol(c);
        double target = TestData.inputScalar(c, "target");
        double[] w = Markowitz.frontierWeights(TestData.muAnnualized(),
                TestData.sigmaAnnualized(), target);
        assertVector(c, "w", w);
        assertEquals(TestData.expectScalar(c, "variance"),
                Linalg.quadForm(w, TestData.sigmaAnnualized()), tol);
    }

    @Test
    public void minvarLongOnlyPgd() {
        assertVector("minvar_longonly_pgd", "w",
                Markowitz.minVarianceLongOnly(TestData.sigmaAnnualized()));
    }

    @Test
    public void frontierConstrainedT080() {
        String c = "frontier_constrained_t080";
        double[] w = Markowitz.constrainedFrontierWeights(
                TestData.muAnnualized(), TestData.sigmaAnnualized(),
                TestData.inputScalar(c, "target"), TestData.inputScalar(c, "penalty"),
                null, null);
        assertVector(c, "w", w);
    }

    @Test
    public void choleskyAnnualizedCov() {
        String c = "cholesky_annualized_cov";
        double tol = TestData.tol(c);
        double[][] low = Linalg.cholesky(TestData.sigmaAnnualized());
        assertEquals(TestData.expectScalar(c, "l_0_0"), low[0][0], tol);
        assertEquals(TestData.expectScalar(c, "l_7_0"), low[7][0], tol);
        assertEquals(TestData.expectScalar(c, "l_7_7"), low[7][7], tol);
    }

    @Test
    public void simplexProjectionHand() {
        String c = "simplex_projection_hand";
        double[] v = {TestData.inputScalar(c, "v_0"), TestData.inputScalar(c, "v_1"),
                TestData.inputScalar(c, "v_2")};
        assertVector(c, "w", Markowitz.projectSimplexBox(v, TestData.inputScalar(c, "total")));
    }

    @Test
    public void simplexProjectionBox() {
        String c = "simplex_projection_box";
        double[] v = {TestData.inputScalar(c, "v_0"), TestData.inputScalar(c, "v_1"),
                TestData.inputScalar(c, "v_2"), TestData.inputScalar(c, "v_3")};
        double[] lb = {TestData.inputScalar(c, "lb")};
        double[] ub = {TestData.inputScalar(c, "ub")};
        assertVector(c, "w",
                Markowitz.projectSimplexBox(v, TestData.inputScalar(c, "total"), lb, ub));
    }

    @Test
    public void impliedPi() {
        String c = "implied_pi";
        double[] pi = BlackLitterman.impliedEquilibriumReturns(
                TestData.sigmaAnnualized(), TestData.wMkt(), TestData.inputScalar(c, "delta"));
        assertVector(c, "pi", pi);
    }

    @Test
    public void blOmegaDefault() {
        String c = "bl_omega_default";
        double tol = TestData.tol(c);
        BlackLitterman.ViewMatrices vm =
                BlackLitterman.viewsToMatrices(TestData.views(), TestData.assetNames());
        double[][] om = BlackLitterman.defaultOmega(TestData.sigmaAnnualized(), vm.p(),
                TestData.inputScalar(c, "tau"), vm.confidences());
        assertEquals(TestData.expectScalar(c, "omega_0_0"), om[0][0], tol);
        assertEquals(TestData.expectScalar(c, "omega_1_1"), om[1][1], tol);
    }

    @Test
    public void blPosteriorReturnsAndWeights() {
        BlackLitterman.ViewMatrices vm =
                BlackLitterman.viewsToMatrices(TestData.views(), TestData.assetNames());
        double[] pi = BlackLitterman.impliedEquilibriumReturns(
                TestData.sigmaAnnualized(), TestData.wMkt(),
                (Double) TestData.viewsDoc().get("delta"));
        double tau = (Double) TestData.viewsDoc().get("tau");
        BlackLitterman.Result res = BlackLitterman.blPosterior(
                TestData.sigmaAnnualized(), pi, vm.p(), vm.q(), tau, vm.confidences());
        assertVector("bl_posterior_returns", "mu", res.muPost());
        double rf = TestData.inputScalar("bl_posterior_weights", "rf");
        assertVector("bl_posterior_weights", "w",
                Markowitz.tangencyWeights(res.muPost(), res.sigmaPost(), rf));
    }

    @Test
    public void inverseVol() {
        assertVector("inverse_vol", "w",
                RiskParity.inverseVolWeights(TestData.sigmaAnnualized()));
    }

    @Test
    public void ercWeights() {
        String c = "erc_weights";
        double[] w = RiskParity.ercWeights(TestData.sigmaAnnualized());
        assertVector(c, "w", w);
        double[] rc = RiskParity.riskContributions(w, TestData.sigmaAnnualized());
        double lo = rc[0];
        double hi = rc[0];
        for (double v : rc) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        assertEquals(TestData.expectScalar(c, "rc_spread"), hi - lo, TestData.tol(c));
    }

    @Test
    public void riskBudgetPinned() {
        String c = "risk_budget_pinned";
        double[] budgets = new double[8];
        for (int i = 0; i < 8; i++) {
            budgets[i] = TestData.inputScalar(c, "b_" + i);
        }
        assertVector(c, "w", RiskParity.riskBudgetWeights(TestData.sigmaAnnualized(), budgets));
    }

    @Test
    public void backtestEqualWeight() {
        String c = "backtest_equal_weight";
        double tol = TestData.tol(c);
        double rf = TestData.inputScalar(c, "rf");
        int window = (int) TestData.inputScalar(c, "window");
        int step = (int) TestData.inputScalar(c, "step");
        Backtest.Result res = Backtest.runBacktest(TestData.returns(), h -> {
            double[] w = new double[h[0].length];
            java.util.Arrays.fill(w, 1.0 / h[0].length);
            return w;
        }, rf, window, step);
        assertEquals(TestData.expectScalar(c, "ann_return"), res.annReturn(), tol);
        assertEquals(TestData.expectScalar(c, "ann_vol"), res.annVol(), tol);
        assertEquals(TestData.expectScalar(c, "sharpe"), res.sharpe(), tol);
        assertEquals(TestData.expectScalar(c, "max_drawdown"), res.maxDrawdown(), tol);
    }

    @Test
    public void backtestErc() {
        String c = "backtest_erc";
        double tol = TestData.tol(c);
        Backtest.Result res = Backtest.runBacktest(TestData.returns(),
                h -> RiskParity.ercWeights(Estimators.sampleCov(h)),
                TestData.inputScalar(c, "rf"),
                (int) TestData.inputScalar(c, "window"),
                (int) TestData.inputScalar(c, "step"));
        assertEquals(TestData.expectScalar(c, "sharpe"), res.sharpe(), tol);
        assertEquals(TestData.expectScalar(c, "max_drawdown"), res.maxDrawdown(), tol);
    }

    @Test
    public void turnoverMinvarLongOnly() {
        String c = "turnover_minvar_longonly";
        double tol = TestData.tol(c);
        Backtest.Result res = Backtest.runBacktest(TestData.returns(),
                h -> Markowitz.minVarianceLongOnly(
                        Estimators.annualizeCov(Estimators.sampleCov(h), 252)),
                0.0,
                (int) TestData.inputScalar(c, "window"),
                (int) TestData.inputScalar(c, "step"));
        assertEquals(TestData.expectScalar(c, "avg_turnover"), res.avgTurnover(), tol);
        assertEquals(TestData.expectScalar(c, "n_rebalances"),
                res.rebalanceDates().size(), tol);
    }

    @Test
    public void viewsAreTheBundledPair() {
        // Guards the fixture: the golden BL cases assume exactly these two views.
        List<BlackLitterman.View> views = TestData.views();
        assertEquals(2, views.size());
        assertEquals("absolute", views.get(0).type());
        assertEquals("EM_EQ", views.get(0).asset());
        assertEquals("relative", views.get(1).type());
        assertEquals("US_EQ", views.get(1).longAsset());
        assertEquals("EU_EQ", views.get(1).shortAsset());
    }
}
