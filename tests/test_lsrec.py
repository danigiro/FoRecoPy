import numpy as np
import forecopy as rpy
import unittest

class TestCS(unittest.TestCase):
    def test_tools(self):
        agg_mat = np.array([1,1,1,1,1,1,0,0,0,0,1,1]).reshape(3,4) 
        cons_mat = np.array([1,0,0,-1,-1,-1,-1,0,1,0,-1,-1,0,0,0,0,1,0,0,-1,-1]).reshape(3,7)
        tag = rpy.cstools(agg_mat=agg_mat)
        tcm = rpy.cstools(cons_mat=cons_mat)

        with self.subTest():
            self.assertTrue(np.all(tag.agg_mat==tcm.agg_mat))
        with self.subTest():
            self.assertTrue(np.all(tag.cons_mat()==tcm.cons_mat()))
        with self.subTest():
            self.assertTrue(np.all(tag.strc_mat()==tcm.strc_mat()))
        with self.subTest():
            self.assertTrue(np.all(tag.dim==tcm.dim))
    

    def test_solver_cov(self):
        np.random.seed(123)

        # Simulation parameters for base and residuals
        h = 2    # Forecast horizons
        N = 100  # Number of residuals
        agg_mat = np.array([1,1,1,1,1,1,0,0,0,0,1,1]).reshape(3,4) # Aggregation matrix
        cons_mat = np.array([1,0,0,-1,-1,-1,-1,0,1,0,-1,-1,0,0,0,0,1,0,0,-1,-1]).reshape(3,7)
        bts_mean = np.repeat(5, agg_mat.shape[1]) # Bottom time series' mean
        mean = np.concatenate([agg_mat @ bts_mean, bts_mean]) # All time series' mean

        # Simulated base forecasts
        base = np.random.normal(
            loc = np.concatenate([mean for i in range(h)]),
            size = sum(agg_mat.shape)*h
            ).reshape([h,sum(agg_mat.shape)])

        # Simulated residuals 
        res = np.random.normal(
            loc = np.concatenate([mean for i in range(N)]),
            size = sum(agg_mat.shape)*N
            ).reshape([N,sum(agg_mat.shape)]) - mean
        
        for i in ['ols', 'str', 'wls', 'shr', 'sam']:
            r1 = rpy.csrec(
                base=base, res=res, comb=i, approach='proj', agg_mat=agg_mat
                )
            r2 = rpy.csrec(
                base=base, res=res, comb=i, approach='strc', agg_mat=agg_mat
                )
            r3 = rpy.csrec(
                base=base, res=res, comb=i, approach='proj', cons_mat=cons_mat
                )
            r4 = rpy.csrec(
                base=base, res=res, comb=i, approach='strc', cons_mat=cons_mat
                )
            
            r5 = rpy.csrec(
                base=base, res=res, comb=i, approach='proj', 
                cons_mat=cons_mat, solver='linearx'
                )
            r6 = rpy.csrec(
                base=base, res=res, comb=i, approach='strc', 
                cons_mat=cons_mat, solver='linearx'
                )

            r7 = rpy.csrec(
                base=base, res=res, comb=i, approach='proj_tol', 
                cons_mat=cons_mat
                )
            r8 = rpy.csrec(
                base=base, res=res, comb=i, approach='strc_tol', 
                cons_mat=cons_mat
                )
            r9 = rpy.csrec(
                base=base, res=res, comb=i, approach='proj_tol', 
                cons_mat=cons_mat, solver='linearx'
                )
            r10 = rpy.csrec(
                base=base, res=res, comb=i, approach='strc_tol', 
                cons_mat=cons_mat, solver='linearx'
                )

            with self.subTest():
                self.assertTrue(np.mean((r1-r2)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r3)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r4)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r5)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r6)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r7)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r8)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r9)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r10)**2)<1e-6)


class TestTE(unittest.TestCase):
    def test_tools(self):
        agg_order = 4
        agg_mat = np.array([1,1,1,1,1,1,0,0,0,0,1,1]).reshape(3,4) 
        tcs = rpy.cstools(agg_mat=agg_mat)
        tte = rpy.tetools(agg_order=agg_order)

        with self.subTest():
            self.assertTrue(np.all(tcs.agg_mat==tte._agg_mat))
        with self.subTest():
            self.assertTrue(np.all(tcs.cons_mat()==tte.cons_mat()))
        with self.subTest():
            self.assertTrue(np.all(tcs.strc_mat()==tte.strc_mat()))
        with self.subTest():
            self.assertTrue(tcs.dim==(tte.kt, tte.ks, tte.m))
        with self.subTest():
            self.assertTrue(tte.p==3)
    

    def test_solver_cov(self):
        np.random.seed(123)
        # Simulation parameters for base and residuals
        h = 2    # Forecast horizons for the lowest frequency time series
        N = 100  # Number of residuals for the lowest frequency time series
        agg_order = 12 # Max. aggregation order
        strc_mat_te = rpy.tetools(agg_order=agg_order).strc_mat()
        mean = np.array(strc_mat_te @ np.repeat(3, strc_mat_te.shape[1]))
        agg_order_set = rpy.tetools(agg_order=agg_order).kset

        # Simulated base forecasts
        base = np.random.normal(
            loc = np.repeat(mean, h),
            size = strc_mat_te.shape[0]*h
            )

        # Simulated residuals 
        res = np.random.normal(
            loc = np.repeat(mean, N),
            size = strc_mat_te.shape[0]*N
            ) - np.repeat(mean, N)
        
        for i in ['ols', 'str', 'wlsv', 'shr', 'sam']:
            r1 = rpy.terec(
                base=base, res=res, comb=i, approach='proj', 
                agg_order=agg_order
                )
            r2 = rpy.terec(
                base=base, res=res, comb=i, approach='strc', 
                agg_order=agg_order
                )
            r3 = rpy.terec(
                base=base, res=res, comb=i, approach='proj', 
                agg_order=agg_order_set
                )
            r4 = rpy.terec(
                base=base, res=res, comb=i, approach='strc', 
                agg_order=agg_order_set
                )
            
            r5 = rpy.terec(
                base=base, res=res, comb=i, approach='proj', 
                agg_order=agg_order, solver='linearx'
                )
            r6 = rpy.terec(
                base=base, res=res, comb=i, approach='strc', 
                agg_order=agg_order, solver='linearx'
                )

            r7 = rpy.terec(
                base=base, res=res, comb=i, approach='proj_tol', 
                agg_order=agg_order
                )
            r8 = rpy.terec(
                base=base, res=res, comb=i, approach='strc_tol', 
                agg_order=agg_order
                )
            r9 = rpy.terec(
                base=base, res=res, comb=i, approach='proj_tol', 
                agg_order=agg_order, solver='linearx', tol=1e-9
                )
            r10 = rpy.terec(
                base=base, res=res, comb=i, approach='strc_tol', 
                agg_order=agg_order, solver='linearx'
                )

            with self.subTest():
                self.assertTrue(np.mean((r1-r2)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r3)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r4)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r5)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r6)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r7)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r8)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r9)**2)<1e-6)
            with self.subTest():
                self.assertTrue(np.mean((r1-r10)**2)<1e-6)

if __name__ == '__main__':
    unittest.main()