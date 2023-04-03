.. TFRE documentation master file, created by
   sphinx-quickstart on Sat Apr  1 21:51:40 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. 


TFRE: A Tuning-Free Robust and Efficient Approach to High-dimensional Regression
=================================================================================

.. toctree::
   :maxdepth: 2
 
   ./docs/TFRE.rst


[Wang2020]_ proposed the TFRE Lasso estimator for high-dimensional linear regressions with heavy-tailed errors as below: $$\\widehat{\\boldsymbol{\\beta}}(\\lambda^{*}) = \\arg\\min_{\\boldsymbol{\\beta}}\\frac{1}{n(n-1)}{\\sum\\sum}_{i\\neq j}\\left|(Y_i-\\boldsymbol{x}_i^T\\boldsymbol{\\beta})-(Y_j-\\boldsymbol{x}_j^T\\boldsymbol{\\beta})\\right| + \\lambda^{*}\\sum_{k=1}^p|\\beta_k|,$$
where :math:`\lambda^{*}` is the tuning parameter which can be estimated independent of errors. In [Wang2020]_, the following tuning parameter is suggested $$\\lambda^{*} = const_{\\lambda} * G^{-1}_{||\\boldsymbol{S}_n||_\\infty}(1-\\alpha_0), $$
where :math:`\boldsymbol{S}_n = -2[n(n-1)]^{-1}\sum_{j=1}^n\boldsymbol{x}_j[2r_j-(n+1)]`, :math:`r_1,\ldots,r_n` follows the uniform distribution on the permutations of the integers :math:`\{1,\ldots,n\}`, and :math:`G^{-1}_{||\boldsymbol{S}_n||_\infty}(1-\alpha_0)` denotes the :math:`(1-\alpha_0)`-quantile of the distribution of :math:`||\boldsymbol{S}_n||_\infty`.
 
In this package, the TFRE Lasso model is fitted by QICD algorithm proposed in [PengWang2015]_. To overcome the computational barrier arising from the U-statistics structure of the aforementioned loss function, we apply the *Incomplete U-statistics* resampling technique which was first proposed in [Clemencon2016]_.

[Wang2020]_ also proposed a second-stage enhancement by using the TFRE Lasso estimator :math:`\widehat{\boldsymbol{\beta}}(\lambda^{*})` as an initial estimator. It is defined as:
$$\\widetilde{\\boldsymbol{\\beta}}^{(1)} = \\arg\\min_{\\boldsymbol{\\beta}}\\frac{1}{n(n-1)}{\\sum\\sum}_{i\\neq j}\\left|(Y_i-\\boldsymbol{x}_i^T\\boldsymbol{\\beta})-(Y_j-\\boldsymbol{x}_j^T\\boldsymbol{\\beta})\\right| + \\sum_{k=1}^pp_{\\eta}'( | \\widehat{\\beta}_{k} (\\lambda^{*}) | )|\\beta_k|,$$ where :math:`p'_{\eta}(\cdot)` denotes the derivative of some nonconvex penalty function  :math:`p_{\eta}(\cdot)`, :math:`\eta > 0` is a tuning parameter. This function implements the second-stage enhancement with two popular nonconvex penalty functions: SCAD and MCP. The modified high-dimensional BIC criterion in [Wang2020]_ is employed for selecting :math:`\eta`. Define: $$HBIC(\\eta) = \\log\\left\\{{\\sum\\sum}_{i\\neq j}\\left|(Y_i-\\boldsymbol{x}_i^T\\widetilde{\\boldsymbol{\\beta}}_{\\eta})-(Y_j-\\boldsymbol{x}_j^T\\widetilde{\\boldsymbol{\\beta}}_{\\eta})\\right|\\right\\} + | A_{\\eta} | \\frac{\\log\\log n}{n* const\\_hbic}\\log p,$$
where :math:`\widetilde{\boldsymbol{\beta}}_{\eta}` denotes the second-stage estimator with the tuning parameter value :math:`\eta`, and :math:`|A_{\eta}|` denotes the cardinality of the index set of the selected model. In this package, we select the value of :math:`\eta` that minimizes HBIC(:math:`\eta`).
 



Indices and tables
==================

* :ref:`genindex`  
* :ref:`search`


Reference
==========

.. [Wang2020] Lan Wang, Bo Peng, Jelena Bradic, Runze Li & Yunan Wu (2020) A Tuning-free Robust and Efficient Approach to High-dimensional Regression, Journal of the American Statistical Association, 115:532, 1700-1714, `DOI: 10.1080/01621459.2020.1840989 <https://doi.org/10.1080/01621459.2020.1840989>`_.     

.. [PengWang2015] Bo Peng & Lan Wang (2015) An Iterative Coordinate Descent Algorithm for High-Dimensional Nonconvex Penalized Quantile Regression, Journal of Computational and Graphical Statistics, 24:3, 676-694, `DOI: 10.1080/10618600.2014.913516 <https://doi.org/10.1080/10618600.2014.913516>`_.   
.. [Clemencon2016] Stephan Clemencon, Igor Colin, & Aurelien Bellet, (2016). Scaling-up empirical risk minimization: optimization of incomplete U-statistics. The Journal of Machine Learning Research, 17(1), 2682-2717.