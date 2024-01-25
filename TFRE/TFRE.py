"""
A Tuning-Free Robust and Efficient (TFRE) Approach to High-dimensional Regression

"""
 
import numpy as np
import matplotlib.pyplot as plt  
from . import QICD

class TFRE:     
    """
    A class used to perform TFRE regrssions
    
    Returns
    -------
    model : TFRE.model class
        The class used to record the regression details.
    TFRE_Lasso : TFRE.Lasso class
        The class used to record the results of the TFRE regrssion with Lasso penalty. 
    TFRE_scad : TFRE.SCAD class
        The class used to record the results of the TFRE regrssion with SCAD 
        penalty. ``None`` if ``second_stage`` is not ``scad``. 
    TFRE_mcp : TFRE.MCP class
        The class used to record the results of the TFRE regrssion with MCP 
        penalty. ``None`` if ``second_stage`` is not ``mcp``.
        
    
    """
    class model:
        """
        a class used to record the regression details.
        
        Returns
        -------
        X : np.ndarray([n,p])
            Input matrix of the regression.
        y : np.ndarray([n,])
            Response vector of the regression. 
        incomplete : bool
            If ``True``, the *Incomplete U-statistics* resampling technique would 
            be applied in computation. If ``False``, the complete U-statistics 
            would be used in computation. 
        second_stage : str
            Penalty function for the second stage model. One of ``"scad"``, 
            ``"mcp"`` and ``"none"``. 
            
        """
        def __init__(self,X,y,incomplete,second_stage):
            self.X = X
            self.y = y
            self.incomplete = incomplete
            self.second_stage = second_stage
        
    
    class Lasso:
        """
        a class used to record the results of the TFRE regrssion with Lasso penalty. 
        
        Returns
        -------
        beta_TFRE_Lasso : np.ndarray([p+1,])
            The estimated coefficient vector of the TFRE Lasso regression. The 
            first element is the estimated intercept.
        tfre_lambda : np.ndarray([1,])
            The estimated tuning parameter of the TFRE Lasso regression.
            
        """
        def __init__(self,beta_TFRE_Lasso,tfre_lambda):  
            self.beta_TFRE_Lasso = beta_TFRE_Lasso
            self.tfre_lambda = tfre_lambda 
    
    
    class SCAD:
        """
        a class used to record the results of the TFRE regrssion with SCAD penalty. 
        ``None`` if ``second_stage`` is not ``scad``. 
        
        Returns
        -------
        Beta_TFRE_scad : np.ndarray([k,p+1])
            The estimated coefficient matrix of the TFRE SCAD regression. The 
            diminsion is k x (p+1) with the first column to be the intercepts, 
            where k is the length of ``eta_list`` vector.
        df_TFRE_scad : np.ndarray([k,])
            The number of nonzero coefficients (intercept excluded) for each 
            value in ``eta_list``. 
        eta_list : np.ndarray([k,])
            The tuning parameter vector used in the TFRE SCAD regressions.
        hbic : np.ndarray([k,])
            A numerical vector of HBIC values for the TFRE SCAD model corresponding 
            to each value in ``eta_list``. 
        eta_min : float
            The eta value which yields the smallest HBIC value in the TFRE SCAD
            regression.
        beta_TFRE_scad_min : np.ndarray([p+1,])
            The estimated coefficient vector which employs ``eta_min`` as the 
            eta value in the TFRE SCAD regression. 
            
        """ 
        def __init__(self,Beta_TFRE_scad,df_TFRE_scad,eta_list,hbic,min_ind):
            self.Beta_TFRE_scad = Beta_TFRE_scad
            self.df_TFRE_scad = df_TFRE_scad
            self.eta_list = eta_list
            self.hbic = hbic
            self.eta_min = eta_list[min_ind]
            self.beta_TFRE_scad_min = Beta_TFRE_scad[min_ind,]
 
            
    class MCP:
        """
        a class used to record the results of the TFRE regrssion with MCP penalty. 
        ``None`` if ``second_stage`` is not ``mcp``.
        
        Returns
        -------
        Beta_TFRE_mcp : np.ndarray([k,p+1])
            The estimated coefficient matrix of the TFRE MCP regression. The 
            diminsion is k x (p+1) with the first column to be the intercepts, 
            where k is the length of ``eta_list`` vector.
        df_TFRE_mcp : np.ndarray([k,])
            The number of nonzero coefficients (intercept excluded) for each 
            value in ``eta_list``. 
        eta_list : np.ndarray([k,])
            The tuning parameter vector used in the TFRE MCP regressions.
        hbic : np.ndarray([k,])
            A numerical vector of HBIC values for the TFRE MCP model corresponding 
            to each value in ``eta_list``. 
        eta_min : float
            The eta value which yields the smallest HBIC value in the TFRE MCP
            regression.
        beta_TFRE_mcp_min : np.ndarray([p+1,])
            The estimated coefficient vector which employs ``eta_min`` as the 
            eta value in the TFRE MCP regression.  
            
        """ 
        def __init__(self,Beta_TFRE_mcp,df_TFRE_mcp,eta_list,hbic,min_ind):
            self.Beta_TFRE_mcp = Beta_TFRE_mcp
            self.df_TFRE_mcp = df_TFRE_mcp
            self.eta_list = eta_list
            self.hbic = hbic
            self.eta_min = eta_list[min_ind]
            self.beta_TFRE_mcp_min = Beta_TFRE_mcp[min_ind,] 
            
            
    def est_lambda(self, X = None, alpha0 = 0.1, const_lambda = 1.01, times = 500):
        """Estimate the tuning parameter for a TFRE Lasso regression given the covariate matrix X.
         
        Parameters
        ----------
        X : np.ndarray([n,p])
            Input matrix of the regression.
        alpha0 : float, optional, default = 0.1
            The level to estimate the tuning parameter.  
        const_lambda : float, optional, default = 1.01
            The constant to estimate the tuning parameter, should be greater than 1.
        times : int, optional, default = 500
            The size of simulated samples to estimate the tuning parameter. 
  
 
        Returns
        -------
        : float
            The estimated tuning parameter of the TFRE Lasso regression given X. 
             
        Examples
        --------
        >>> import numpy as np
        >>> from TFRE import TFRE
        >>> n = 100
        >>> p = 400
        >>> X = np.random.normal(0,1,size=(n,p))
        >>> obj = TFRE()
        >>> obj.est_lambda(X)
        [0.43150559039112646]
            
        """
        if X is None or type(X) is not np.ndarray:
            print("""Error in est_lambda():\nPlease supply the covariate matrix X to estimate the tuning parameter for TFRE Lasso""")
            return 

        n = X.shape[0]
        epsilon_tfre = np.random.choice(np.arange(1, n+1), size=(times, n))
        xi = 2 * epsilon_tfre - (n + 1)
        S = (-2 / n / (n - 1)) * X.T.dot(xi.T)
        res = np.apply_along_axis(lambda t: np.max(np.abs(t)), axis=0, arr=S)
 
        
        return np.quantile(res,1-alpha0)*const_lambda
    
    
    def __p_diff(self, theta, second_stage, lamb, a = 3.7):
        theta = np.abs(theta)
        if second_stage == "scad":
            less = np.less.outer(theta, lamb)  
            y = np.maximum(np.add.outer(-theta, a * lamb),0)
            res = np.multiply(less,lamb) + np.divide(np.multiply(y, (1 - less)), (a - 1))

        else:
            res = np.minimum(np.add.outer(-theta / a, lamb), lamb)
            
        return np.maximum(res,1e-4)
 
    
    def __hbic_tfre_second(self, newx, newy, n, beta_int, second_stage, 
                         lambda_list, a, thresh, maxin, maxout, const):
        penalty = self.__p_diff(beta_int.reshape(-1),second_stage,lambda_list,a) 
        Beta = QICD.fit(newx, newy, penalty, beta_int, thresh, maxin, maxout)
        df = np.apply_along_axis(lambda t: np.sum(np.abs(t) > 1e-06), axis=1, arr=Beta)
        hbic1 = np.apply_along_axis(lambda t: np.log(np.sum(np.abs(newy - np.dot(newx, t)))), axis=1, arr=Beta) 
        hbic2 = np.log(newx.shape[1]) * df * np.log(np.log(n)) / n / const
        
        hbic = hbic1 + hbic2
         
         
        return hbic, Beta 
     
        
    def fit(self, X = None, y = None, alpha0 = 0.1, const_lambda = 1.01, 
            times = 500, incomplete = True, const_incomplete = 10, 
            thresh = 1e-06, maxin = 100, maxout = 20, second_stage = "scad",
            a = 3.7, eta_list = None, const_hbic = 6): 
        """Fit a TFRE regression model with Lasso, SCAD or MCP regularization.
 
        Parameters
        ----------
        X : np.ndarray([n,p])
            Input matrix of the regression.
        y : np.ndarray([n,])
            Response vector of the regression.  
        alpha0 : float, optional, default = 0.1
            The level to estimate the tuning parameter.  
        const_lambda : float, optional, default = 1.01
            The constant to estimate the tuning parameter, should be greater than 1.
        times : int, optional, default = 500
            The size of simulated samples to estimate the tuning parameter. 
        incomplete : bool, optional, defaule = ``True``
            If ``True``, the *Incomplete U-statistics* resampling technique would 
            be applied in computation. If ``False``, the complete U-statistics 
            would be used in computation. 
        const_incomplete : int, optional, default = 10
            The constant for the *Incomplete U-statistics* technique. If `
            `incomplete = TRUE``, ``const_incomplete`` x n samples will be 
            randomly selected in the coefficient estimation.
        thresh : float, optional, default = 1e-6
            Convergence threshold for QICD algorithm.
        maxin : int, optional, default = 100
            Maximum number of inner coordiante descent iterations in QICD algorithm.
        maxout : int, optional, default = 20
            Maximum number of outter Majoriaztion Minimization step (MM) iterations 
            in QICD algorithm. 
        second_stage : str, optional, default = ``"scad"``
            Penalty function for the second stage model. One of ``"scad"``, 
            ``"mcp"`` and ``"none"``. 
        a : float, optional, default = 3.7, suggested by Fan and Li (2001)
            an unknown parameter in SCAD and MCP penalty functions.
        eta_list : float, optional, default = 3.7, suggested by Fan and Li (2001)
            A numerical vector for the tuning parameters to be used in the TFRE S
            CAD or MCP regression. Cannot be ``None`` if ``second_stage = "scad"``
            or ``"mcp"``.
        const_hbic : int, optional,  default = 6
            The constant to be used in calculating HBIC in the TFRE SCAD regression.
      
        Returns
        -------
        self: :class:`TFRE` class
            a fitted :class:`TFRE` class with attributes "model", "TFRE_Lasso", 
            "TFRE_scad" (if ``second_stage = "scad"``), and "TFRE_mcp"(if ``second_stage = "mcp"``).
    
        Examples
        --------
        >>> import numpy as np
        >>> from TFRE import TFRE
        >>> n = 100
        >>> p = 400
        >>> X = np.random.normal(0,1,size=(n,p))
        >>> beta =  np.append([1.5,-1.25,1,-0.75,0.5],np.zeros(p-5))
        >>> y = X.dot(beta) + np.random.normal(0,1,n)
        >>> 
        >>> obj = TFRE()
        >>> obj.fit(X,y,eta_list=np.arange(0.09,0.51,0.03))
        <TFRE.TFRE.TFRE at 0x148111e10>
        
        """
        if X is None or y is None:
            print("""Error in fit():\nPlease supply the data (X, y) for the regression""")
            return 
        
        if X.shape[0] is not y.shape[0]:
            print("""Error in fit():\nThe row number of X and the length of y should be consistent""")
            return 
        
        if second_stage!="none" and eta_list is None:
            print("""Error in fit():\nPlease supply the tuning parameter list for the TFRE SCAD or MCP regression""")
            return 
         
        self.model = self.model(X,y,incomplete,second_stage)
        #{"X": X, "y": y, "incomplete": incomplete, "second_stage": second_stage}
      
        n = len(y)
        p = X.shape[1]
        xbar = X.mean(0)
        ybar = y.mean()
        lam_lasso = np.array([self.est_lambda(X, alpha0, const_lambda, times)])
        initial = np.zeros(p)

          
        def numpy_pairwise_combinations(x):
            idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
            return x[idx[:,0],]-x[idx[:,1],]
        
        x_diff = numpy_pairwise_combinations(X)
        y_diff = numpy_pairwise_combinations(y)
        n_diff = len(y_diff)
        
        if incomplete:
            ind = np.random.choice(n_diff,const_incomplete*n,replace=False) 
            newx = x_diff[ind,]
            newy = y_diff[ind]
        else:
            newx = x_diff
            newy = y_diff 
      
        beta_Lasso = QICD.fit(newx, newy, np.full((p, 1), lam_lasso), initial, thresh, maxin, maxout)
        intercpt_Lasso = ybar - beta_Lasso.dot(xbar) 
        beta_TFRE_Lasso = np.append(intercpt_Lasso, beta_Lasso) 
          
        self.TFRE_Lasso = self.Lasso(beta_TFRE_Lasso,lam_lasso) 
        
        if second_stage == "scad":
            hbic, beta_scad = self.__hbic_tfre_second(newx, newy, n, beta_Lasso.reshape(-1), 
                                                      second_stage, eta_list, a, 
                                                      thresh, maxin, maxout, const_hbic) 
            intercpt_scad = ybar - beta_scad.dot(xbar) 
            Beta_TFRE_scad = np.column_stack((intercpt_scad,beta_scad))  
            min_ind = hbic.argmin(0)
            df_TFRE_scad = np.count_nonzero(beta_scad, axis=1) 
            self.TFRE_scad = self.SCAD(Beta_TFRE_scad,df_TFRE_scad,eta_list,hbic,min_ind) 
              
        elif second_stage == "mcp":
            hbic, beta_mcp = self.__hbic_tfre_second(newx, newy, n, beta_Lasso.reshape(-1), 
                                                     second_stage, eta_list, a, 
                                                     thresh, maxin, maxout, const_hbic) 
            intercpt_mcp = ybar - beta_mcp.dot(xbar)  
            Beta_TFRE_mcp = np.column_stack((intercpt_mcp,beta_mcp))  
            min_ind = hbic.argmin(0) 
            df_TFRE_mcp = np.count_nonzero(beta_mcp, axis=1) 
            self.TFRE_mcp = self.MCP(Beta_TFRE_mcp,df_TFRE_mcp,eta_list,hbic,min_ind) 
           
        elif second_stage != "none":
            print("""Warning in fit():\n'second_stage' should be one of 'none', 'scad' and 'mcp'""")
  
        return self
  
             
    def predict(self, newX, s = None): 
        """Make predictions from a fitted :class:`TFRE` class for new X values.
        
        Parameters
        ----------
        newX : np.ndarray([:math:`n_0`,p])
            Matrix of new values for X at which predictions are to be made.
        s : str
            Regression model to use for prediction. Should be one of ``"1st"`` 
            and ``"2nd"``.
 
        Returns
        -------
        : np.ndarray([:math:`n_0`,])
            The vector of predictions for the new X values given the fitted 
            ``TFRE`` class. 
         
        Notes
        ----- 
        If ``second_stage = None``, ``s`` cannot be ``"2nd"``. If ``second_stage = None``
        and ``s = "2nd"``, the function will return the predictions based on the 
        TFRE Lasso regression. If ``second_stage = "scad"`` or ``"mcp"``, and 
        ``s = "2nd"``, the function will return the predictions based on the T
        FRE SCAD or MCP regression with the smallest HBIC.
        
        Examples
        --------
        >>> import numpy as np
        >>> from TFRE import TFRE
        >>> n = 100
        >>> p = 400
        >>> X = np.random.normal(0,1,size=(n,p))
        >>> beta =  np.append([1.5,-1.25,1,-0.75,0.5],np.zeros(p-5))
        >>> y = X.dot(beta) + np.random.normal(0,1,n)
        >>> 
        >>> obj = TFRE()
        >>> obj.fit(X,y,eta_list=np.arange(0.09,0.51,0.03))
        >>> 
        >>> newX = np.random.normal(0,1,size=(10,p))
        >>> obj.predict(newX,"2nd")
        array([ 2.61684897,  2.66548778, -0.13456993, -0.67466848,  3.92941648,
                1.21428428, -1.66033086, -2.13238483,  0.95340816, -2.32122001])
        
        """
        if newX is None or type(newX) is not np.ndarray:
            print("""Error in predict():\nPlease supply a numpy.ndarray for 'newX'""")
            return 
        
        if not hasattr(self,"TFRE_Lasso"):
            print("""Error in predict():\nPlease supply a valid 'TFRE' object""")
            return 
        
        p = self.model.X.shape[1]
        if newX.shape[1] != p:
            print("""Error in predict():\nThe number of variables in 'newX' must be""", p)
            return 
        
        if s=="1st":
            pred = newX.dot(self.TFRE_Lasso.beta_TFRE_Lasso[1:]) + self.TFRE_Lasso.beta_TFRE_Lasso[0] 
        elif(s=="2nd"):
            if self.model.second_stage == "scad":
                pred = newX.dot(self.TFRE_scad.beta_TFRE_scad_min[1:]) + self.TFRE_scad.beta_TFRE_scad_min[0] 
            elif self.model.second_stage == "mcp":
                pred = newX.dot(self.TFRE_mcp.beta_TFRE_mcp_min[1:]) + self.TFRE_mcp.beta_TFRE_mcp_min[0] 
            else:
                pred = newX.dot(self.TFRE_Lasso.beta_TFRE_Lasso[1:]) + self.TFRE_Lasso.beta_TFRE_Lasso[0] 
                print("""Warning in predict():\nThe object doesn't include a second stage model. Return the predicted values according to the TFRE Lasso regression""") 
        else:
            print("""Error in predict():\ns should be one of '1st' and '2nd'""")  
            return
        
        return pred
    
    
    def coef(self, s = None):  
        """Extract coefficients from a fitted :class:`TFRE` class. 
         
        Parameters
        ---------- 
        s : str
            Regression model to use for coefficient extraction. Should be one 
            of ``"1st"`` and ``"2nd"``.
 
        Returns
        -------
        : np.ndarray([p+1,])
            The coefficient vector from the fitted ``TFRE`` class, with the first 
            element as the intercept.

         
        Notes
        ----- 
        If ``second_stage = None``, ``s`` cannot be ``"2nd"``. If ``second_stage = None`` 
        and ``s = "2nd"``, the function will return the coefficient vector from 
        the TFRE Lasso regression. If ``second_stage = "scad"`` or ``"mcp"``, and
        ``s = "2nd"``, the function will return the coefficient vector from the 
        TFRE SCAD or MCP regression with the smallest HBIC.
        
        Examples
        --------
        >>> import numpy as np
        >>> from TFRE import TFRE
        >>> n = 100
        >>> p = 400
        >>> X = np.random.normal(0,1,size=(n,p))
        >>> beta =  np.append([1.5,-1.25,1,-0.75,0.5],np.zeros(p-5))
        >>> y = X.dot(beta) + np.random.normal(0,1,n)
        >>> 
        >>> obj = TFRE()
        >>> obj.fit(X,y,eta_list=np.arange(0.09,0.51,0.03))
        >>> 
        >>> obj.coef("1st")[:10]
        array([-0.12943468,  1.21390299, -0.82102807,  0.56632981, -0.20740154,
                0.        ,  0.        ,  0.        ,  0.        ,  0.        ])
        >>> obj.coef("2nd")[:10]
        array([-0.13552865,  1.63426996, -1.13200778,  1.1699545 , -0.47397631,
                0.17350995,  0.        ,  0.        ,  0.        ,  0.        ])
        
        """
        if not hasattr(self,"TFRE_Lasso"):
            print("""Error in coef():\nPlease supply a valid 'TFRE' object""")
            return 
        
        if s=="1st":
            coef = self.TFRE_Lasso.beta_TFRE_Lasso
        elif(s=="2nd"):
            if self.model.second_stage == "scad":
                coef = self.TFRE_scad.beta_TFRE_scad_min
            elif self.model.second_stage == "mcp":
                coef = self.TFRE_mcp.beta_TFRE_mcp_min
            else:
                coef = self.TFRE_Lasso.beta_TFRE_Lasso
                print("""Warning in coef():\nThe object doesn't include a second stage model. Return the coefficient vector from the TFRE Lasso regression""") 
        else:
            print("""Error in coef():\ns should be one of '1st' and '2nd'""")  
            return
        
        return coef


    def plot(self):  
        """Plot the second stage model curve for a fitted :class:`TFRE` class.
        
        Returns
        -------
        : Figure
            This function plots the HBIC curve and the model size curve as a 
            function of the ``eta`` values used, from a fitted TFRE SCAD or MCP 
            model.

         
        Notes
        ----- 
        In the output plot, the red line represents the HBIC curve as a function
        of ``eta`` values, the blue line represents the number of nonzero coefficients
        as a function of ``eta`` values, and the purple vertical dashed line denotes
        the model selected with the smallest HBIC.
        
        This function cannot plot the object if ``second_stage = None``.
        
        Examples
        --------
        >>> import numpy as np
        >>> from TFRE import TFRE
        >>> n = 100
        >>> p = 400
        >>> X = np.random.normal(0,1,size=(n,p))
        >>> beta =  np.append([1.5,-1.25,1,-0.75,0.5],np.zeros(p-5))
        >>> y = X.dot(beta) + np.random.normal(0,1,n)
        >>>  
        >>> obj = TFRE()
        >>> obj.fit(X,y,eta_list=np.arange(0.09,0.51,0.03))
        >>> obj.plot() 
        
        .. image::  ../plot.png
        
        """
        
        if not hasattr(self,"TFRE_Lasso"):
            print("""Error in coef():\nPlease supply a valid 'TFRE' object""")
            return 
        
        if self.model.second_stage == "scad":
            fig = plt.figure(figsize=(10,8))
            ax1 = fig.add_subplot(222)
            _lines1_, = ax1.plot(self.TFRE_scad.eta_list,self.TFRE_scad.hbic,
                                 color = "r", label="HBIC") 
            ax1.set_xlabel("eta value", fontsize = 14)
            ax1.set_ylabel("HBIC",color = "r", fontsize = 14)
            ax1.tick_params(axis="y", labelcolor = "r")
            
            ax2 = ax1.twinx()
            _lines2_, = ax2.plot(self.TFRE_scad.eta_list,self.TFRE_scad.df_TFRE_scad,
                                 color = "b", label="df")  
            ax2.set_ylabel("df",color = "b", fontsize = 14)
            ax2.tick_params(axis="y", labelcolor = "b")
            
            plt.axvline(x = self.TFRE_scad.eta_min, color = 'tab:purple', 
                        linestyle = "dashed")
            y0 = self.TFRE_scad.df_TFRE_scad[np.argmin(self.TFRE_scad.hbic)]
            plt.plot([self.TFRE_scad.eta_min, max(self.TFRE_scad.eta_list)], [y0, y0], linestyle='dashed', color='tab:purple')
            
            lns = [_lines1_, _lines2_]
            labels = [l.get_label() for l in lns]
            plt.legend(lns, labels, loc = "right", fontsize = 14, frameon = False)
            
            plt.tight_layout()
            plt.show()
            
        elif self.model.second_stage == "mcp":
            fig = plt.figure(figsize=(12,8))
            ax1 = fig.add_subplot(222)
            _lines1_, = ax1.plot(self.TFRE_mcp.eta_list,self.TFRE_mcp.hbic,
                                 color = "r", label="HBIC") 
            ax1.set_xlabel("eta value", fontsize = 14)
            ax1.set_ylabel("HBIC",color = "r", fontsize = 14)
            ax1.tick_params(axis="y", labelcolor = "r")
            
            ax2 = ax1.twinx()
            _lines2_, = ax2.plot(self.TFRE_mcp.eta_list,self.TFRE_mcp.df_TFRE_mcp,
                                 color = "b", label="df")  
            ax2.set_ylabel("df",color = "b", fontsize = 14)
            ax2.tick_params(axis="y", labelcolor = "b")
            
            plt.axvline(x = self.TFRE_mcp.eta_min, color = 'tab:purple', 
                        linestyle = "dashed")
            y0 = self.TFRE_mcp.df_TFRE_mcp[np.argmin(self.TFRE_scad.hbic)]
            plt.plot([self.TFRE_mcp.eta_min, max(self.TFRE_mcp.eta_list)], [y0, y0], linestyle='dashed', color='tab:purple')
            
            lns = [_lines1_, _lines2_]
            labels = [l.get_label() for l in lns]
            plt.legend(lns, labels, loc = "right", fontsize = 14, frameon = False)
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("""Error in plot():\nPlease supply a valid 'TFRE' object with a second stage model""")  
            return
             
        return fig
 