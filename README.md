# TFRE_python: A Tuning-Free Robust and Efficient Approach to High-dimensional Regression 
 This Python package provides functions to estimate the coefficients in high-dimensional linear regressions via a tuning-free and robust approach.  The method was published in Lan Wang, Bo Peng, Jelena Bradic, Runze Li and Yunan Wu (2020) A tuning-free robust and efficient approach to high-dimensional regression. Journal of the American Statistical Association, 115, 1700-1714 (JASA’s discussion paper). See also Lan Wang, Bo Peng, Jelena Bradic, Runze Li and Yunan Wu (2020), Rejoinder to “A tuning-free robust and efficient approach to high-dimensional regression". Journal of the American Statistical Association, 115, 1726-1729.

You can preview the package documentation [here](https://rawcdn.githack.com/yunanwu123/TFRE_python/58029648199f9f4db1fc257bedacb0f2774102b0/doc/_build/html/index.html).

To install the package, please run one of following commands in Terminal: 
```{python} 
pip install git+https://github.com/yunanwu123/TFRE_python
pip install -i https://test.pypi.org/simple/ TFRE
pip install TFRE
```
This package requires the C++ template library [eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page). Please download it before installation.

## Reference

Wang, L., Peng, B., Bradic, J., Li, R. and Wu, Y. (2020), ***A Tuning-free Robust and Efficient Approach to High-dimensional Regression**, Journal of the American Statistical Association, 115:532, 1700-1714*, [doi:10.1080/01621459.2020.1840989](https://doi.org/10.1080/01621459.2020.1840989).

Wang, L., Peng, B., Bradic, J., Li, R. and Wu, Y. (2020), ***Rejoinder to 'A Tuning-Free Robust and Efficient Approach to High-Dimensional Regression'**, Journal of the American Statistical Association, 115:532, 1726-1729*, [doi:10.1080/01621459.2020.1843865](https://doi.org/10.1080/01621459.2020.1843865).

Peng, B. and Wang, L. (2015), ***An Iterative Coordinate Descent Algorithm for High-Dimensional Nonconvex Penalized Quantile Regression**, Journal of Computational and Graphical Statistics, 24:3, 676-694*, [doi:10.1080/10618600.2014.913516](https://doi.org/10.1080/10618600.2014.913516).

Clémençon, S., Colin, I., and Bellet, A. (2016), ***Scaling-up empirical risk minimization: optimization of incomplete u-statistics**, The Journal of Machine Learning Research, 17(1):2682–2717*, URL: [https://jmlr.org/papers/v17/15-012.html](https://jmlr.org/papers/v17/15-012.html).

Fan, J. and Li, R. (2001), ***Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties**, Journal of the American Statistical Association, 96:456, 1348-1360*, [doi:10.1198/016214501753382273](https://doi.org/10.1198/016214501753382273). 
