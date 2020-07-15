This Matlab toolbox implements the BISN algorithm proposed in [1]. Please 
check the following steps before using the toolbox.

1. Comiling the mex code
The C++ code is written based on the template-based C++ library Armadillo. 
To achieve the best performance, it is better to link the C++ code with
BLAS and LAPACK in Intel MKL in Linux, since currently openmp 3.1 is not 
supported in Windows.

The mex code for Matlab 2017b and above in both windows and ubuntu OS have 
been provided. 

In the case the mex code are obselete, you can compile the original C++ 
code into the mex code following the instructions below.

Before mex the C++ code, please download and install Intek MKL from
https://software.intel.com/en-us/mkl

In particualr for ubuntu OS, Intel MLK can be downloaded and installed by 
running the following commands in the terminal:

cd /tmp

sudo wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'

sudo apt-get update

sudo apt-get install intel-mkl-64bit-2018.2-046

To mex the C++ code in Windows using the msvc compiler, please run 
win_msvc_IntelMKL_mex.m in Matlab.

To mex the C++ code in Linux using the g++ compiler, please run 
linux_gpp_IntelMKL_mex.m in Matlab.

You may need to change directory of Intel MKL to your own installation 
directory in the above m files.



2. Call function BISN_integrated.m

We can simply call the function as:

K = BISN_integrated(XDat);

where XDat is a n x p matrix with n observations for each of the p 
variables. Missing data can be represented by NaN in XDat. The resulting 
K matrix will be a sparse matrix. 

You need to reduce the step size eta (see below) if the algorithm 
divergences (e.g., some very large values suddenly appears). By default, 
we set eta = 300.

options.eta = 100;
K = BISN_integrated(XDat, options);

On the other hand, you may consider increasing the step size if the 
algorithm doesn't diverge and you want to speed up the convergence.

Instead of simply estmating L and D from the data (which is given by the 
function BISN.cpp), BISN_integrated.m further thresholding 
lambda / (1 + lambda) using the method in Section V in [2] to yield a 
sparse graph in an automated manner. However, due to the mean-filed 
approximation used in BISN, lambda of elements in the bottom-right corner 
are typically not well estimated. More specifically, since 
K_{jk} = [LDL^T]_{jk}, the elements in  the bottom-right corner of K are 
the sum of a larger set of elements in L and D than the elements in the 
top-left corner. Due to the mean field approximation, the estimates 
of <K_{jk}^2> is typically corrupted by the variances of elements in L and D. 
As lambda is a function of <K_{jk}^2>, it cannot be well estimated either.
To settle this problem, we run BISN again by reversely ordering the data 
(i.e., setting options.backward_pass = 1) and then average the resulting 
lambda with that from the forward pass. Note that options.backward_pass = 1 
by default.

On the other hand, after estimating the sparse precision matrix, 
BISN_integrated.m can further reestimate the non-zero elements in the 
precision via maximum likelihood by setting options.prm_learning = 1. BISN 
can reliably estimate the non-zero elements when the sample size is 
relatively large, but it is recommended to reestimate the non-zero elements 
when the sample size is small. To do so, we can call the function as:

options.prm_learning = 1;
K = BISN_integrated(XDat, options);

An example of testing BISN_integrated on synthetic data w/o missing data 
can be found in example.m. To apply BISN to your own data, you may store 
your data in a nxp matrix as above, normalize the data, and call BISN as 
in example.m.


[1] H. Yu, S. Wu, L. Xin, and J. Dauwels. Fast Bayesian Inference of Sparse 
    Networks with Automatic Sparsity Determination. Journal of Machine 
    Learning Research, 2020.
[2] H. Yu, L. Xin, and J. Dauwels. Variational wishart approximation for 
    graphical model selection: Monoscale and multiscale models. IEEE 
    Transactions on Signal Processing, 2019.