■ Variational Bayesian sparse regression (VBSR) toolbox

In this VBSR toolbox, we implemented several sparse regression algorithms based on Variational Bayesian (VB) method with ARD (Automatic Relevance Determination) prior. In order to apply time series data, we also implemented time embedding representation for output prediction from input time series. Our main concern in this toolbox is to implement efficient methods which can deal with very high input dimension over 10,000. 

■ Implemented algorithms

・The standard VB method ('Sparse full-covariance method') calculates an inverse of regularized input covariance matrix. When the input dimension becomes over several thousands, computational time becomes very long because computational time for inverse calculation increases as cubic power of input dimension. 

・'Sparse space-covariance method' uses sparse constraints only on the space dimension and it does not impose sparse constraints on temporal dimension using time embedding (: temporal dimension is not pruned). This algorithm calculates inverse of spatial covariance matrix and much faster than the standard method for time embedding representation.

・'Sparse stepwise method' maximize the free energy coordinate by coordinate. This method does not use inverse covariance matrix. This method also introduce sparse condition only for spatial dimension but not for temporal dimension.

・'Sparse sequential method' is a incremental method proposed by Tipping and Faul (2003). It increases effective input dimension one by one. This method introduce sparse condition both for space & time dimension. However, this method becomes slow for more than 10,000 total dimension.

■ Download
VBSR toolbox ver 1.0
mex_prog.zip

■ Installation
1. unzip the zipped file to appropriate place

2. Make MEX file
 run 'mex_compile' at the directory 'mex_prog'

3. set path to the unzipped directory of this toolbox
example script: set_path.m
 
4.Run the Test program 
testjob:

■ How to use
Please see 'Usage-predict.txt' and 'testjob.m' for more detail on how to use this toolbox.

■ Time delay embedding 
Please see 'Read_embed.txt' for time delay embedding representation. In the previous version, time delay embedding was done before estimation. When input dimension becomes very large, this requires huge amount of memory and time. Therefore, embedding was done inside the estimation program using MEX-program. 

■ Introduction of sparse estimation
SparseEstimation_intro.pdf

■ Reference
Sato M., (2001).
On-line model selection based on the variational Bayes. 
Neural Computation, 13, 1649-1681.

Toda, A., Imamizu, H., Sato, M., Wada, Y., Kawato, M., (2007). 
Reconstruction of temporal movement from single-trial non-invasive brain activity: A hierarchical Bayesian method. 
The 14th International Conference on Neural Information Processing (ICONIP2007).

Isao Nambu, Rieko Osu, Masa-aki Sato, Soichi Ando, Mitsuo Kawato, Eiichi Naito, (2009).
Single-trial reconstruction of finger-pinch forces from human motor-cortical activation measured by near-infrared spectroscopy (NIRS)
NeuroImage 47, 628-637.

Tipping, M. E. and A. C. Faul, (2003). 
Fast marginal likelihood maximisation for sparse Bayesian models. 
In C. M. Bishop and B. J. Frey (Eds.), Proceedings of the Ninth International Workshop on Artificial Intelligence and Statistics, Key West, FL, Jan 3-6. 

■ Environment
The codes in the toolbox were written for MATLAB ver.6.5 or later. 

■ Copyright
VBSR toolbox is free but copyright software, distributed under the terms of the GNU General Public License as published by the Free Software Foundation . Further details on "GPL" can be found at http://www.gnu.org/copyleft/. No formal support or maintenance is provided or implied.

■ Author
Masa-aki Sato
ATR Computational Neuroscience Laboratories
Department of Computational Brain Imaging
