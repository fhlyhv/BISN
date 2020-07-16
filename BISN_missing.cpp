#include <mex.h>
#include "armaMex.hpp"
#include <armadillo>
#include <time.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <boost/math/special_functions/expint.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/round.hpp>
#define POSITIVE_EPS 0.0001
#define epsilon1 1e-30
#define epsilon2 1e-7
//#pragma comment(lib, "libmwservices.lib")
//extern bool ioFlush(void);
// use half Cauchy prior on the global shrinkage parameter omega

double Lentz_Algorithm(double const x)
{
    double f_prev = epsilon1, C_prev = epsilon1, D_prev = 0, delta = 2+epsilon2, D_curr, C_curr, f_curr;
    double j = 1.0, tmp1, tmp2;
    while (delta-1>=epsilon2 || 1-delta >= epsilon2)
    {
        j++;
        tmp1 = x+2*j-1;
        tmp2 = pow(j-1,2);
        D_curr = 1/(tmp1-tmp2*D_prev);
        C_curr = tmp1-tmp2/C_prev;
        delta = C_curr*D_curr;
        f_curr = f_prev*delta;
        f_prev = f_curr;
        C_prev = C_curr;
        D_prev = D_curr;
    }
    return 1/(x+1+f_curr);
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
    /* Input Interface*/
    mat XDat = armaGetPr(prhs[0]);
    uvec row_missing = conv_to<uvec>::from(armaGetPr(prhs[1])) - 1;
    umat id_missing = conv_to<umat>::from(armaGetPr(prhs[2])) - 1;
    uword maxIter, s, p = XDat.n_cols,n = XDat.n_rows;
    double tol, eta, r;
    
    if (nrhs>3)
    {
        eta = armaGetDouble(prhs[3]);
        eta = eta/(p+eta);
    }
    else eta = 300.0/(p+300.0);//0.05; //4*sqrt(double(p)/(p^2-1));
    
    if (nrhs>4) maxIter = (uword)armaGetDouble(prhs[4]);
    else maxIter = 10000;
    
    if (nrhs>5) tol = armaGetDouble(prhs[5]);
    else tol = 0.01;
    
    if (nrhs>6) r = armaGetDouble(prhs[6]);
    else
    {
        r = 0.5; //(1-400.0/double(p))/(1+400.0/double(p));
        //if(r<0.05) r = 0.05;
    }
    
    if (nrhs>7) s = (uword)armaGetDouble(prhs[7]);
    else s = boost::math::round((double)p/(1e-3*(p-1)+1)); //(pow((double)p,1.5)/(5e-2*(p-1)+sqrt(double(p))));
    
    
    
    
    uword n_missing = row_missing.n_elem, i;
    field<uvec> col_missing(n_missing);
    uvec id_tmp(1, fill::ones);
    for (i = 0; i < n_missing; i ++) {
        col_missing(i) = id_missing(find(id_missing.col(0) == row_missing(i)),id_tmp);
                //col(2).elem(find(id_missing.col(1) == row_missing(i)));
    }
    
    
	
    /* Initialization */
    double timeBegin = clock();
	mat nS(p, p);
    vec dnS;
    uword pe = p*(p-1)/2;

    uvec idl(pe);
    uvec idu(pe);
    uvec idr(pe);
    uvec idc(pe);
    uword k = 0, j, jp, kappa;
    for (j = 0,jp = 0;j<p;j++,jp+=p)
    {
        for (i=j+1;i<p;i++)
        {
            idl(k) = jp+i;
            idu(k) = i*p+j;
            idr(k) = i;
            idc(k) = j;
            k++;
        }

    }

    double psr = (double)p/s;
    uvec idp(p), id0(s), idd;
    vec nd2pp = (double)n/2+linspace<vec>(p,1,p);

    mat ML(p,p,fill::eye);
    mat ML2(p,p,fill::eye);
    mat ML2pVL(p,p,fill::eye);
    mat VL(p,p,fill::zeros);
    mat LAMBDA(p,p,fill::zeros);

    vec h(pe,fill::zeros),zeta(pe),mL(pe,fill::zeros);
    zeta.fill(10.0);

    vec alpha(p,fill::ones);
    vec beta(p,fill::ones);

    double a = (double)pe/2, b = a/50;

    vec mLold = h/zeta;
    vec lambdaold(pe,fill::ones);

    mat VL2(p,p,fill::zeros);
    for (i=1;i<p;i++)
    {
        VL2(span(i,p-1),i) += i;
        VL2(i,span(i,p-1)) += i;
    }
	VL2 *= 0.01;

	vec d(pe,1);
	d.fill(0.5);
    
    
    

    vec mD, mD2, mD2pvD, vD, vL, lambda(pe), gmL, gvL, gmD, gvD, c5(p), c6, mLnew, lambdanew, alphatmp, betatmp, dtmp;//mL,
    mat c1, c2, c3, c4, LDL, c2pc3, S_cond, K_tmp1, K_tmp2, LD, nS0 = XDat.t()*XDat;
    double omega, difmL, diflambda, difmax;
    vec d1h(pe,fill::zeros), d1zeta(pe,fill::zeros), d1alpha(p,fill::zeros), d1beta(p,fill::zeros), d1d(pe,fill::zeros);
    double d2h = 0, d2zeta = 0, d2alpha = 0, d2beta = 0, d2d = 0, d1b = 0, d2b = 0;
    double tau = 6e2; //, tauzeta = tauh, taualpha = tauh, taubeta = tauh, taub = tauh, taud = tauh;
    double rho, gb, btmp;//1/eta;rho_ub, c7=(5*eta>0.25)?0.25:5*eta, 
    vec gh, gzeta, galpha, gbeta, gd;
    arma_rng::set_seed(0);

    /* KL proximal variational inference */
    mexPrintf("Start Running BISN ...\n");
    mexEvalString("drawnow;");
    for (kappa=1;kappa<=maxIter;kappa++)
    {
        mD = alpha/beta;
        mD2 = square(mD);
        vD = mD/beta;
        mD2pvD = mD2+vD;

        vL = 1/zeta;
        VL.elem(idl) = vL;
        mL =  h%vL;
        ML.elem(idl) = mL;
        ML2.elem(idl) = square(mL);
        ML2pVL.elem(idl) = ML2.elem(idl)+vL;

        omega = a/b;
        for (i=0;i<pe;i++) lambda(i)=(d(i)>10)?Lentz_Algorithm(d(i)):-boost::math::expint(-d(i));
		lambda.elem(find(d<=10)) %= exp(d.elem(find(d<=10)));
        lambda = 1/(d%lambda)-1;
//         lambda.elem(find(d<0)).fill(1e6);
//         lambda.elem(find(lambda>1e6)).fill(1e6);
        LAMBDA.elem(idl) = omega*lambda;
        LAMBDA.elem(idu) = LAMBDA.elem(idl);
        
        LD = ML;
        LD.each_row() %= mD.t();
        nS.zeros();
        for (i = 0; i < n_missing; i ++) {
            id_tmp.fill(row_missing(i));
            S_cond = LD.rows(col_missing(i)) * ML.rows(col_missing(i)).t();
            S_cond = (S_cond + S_cond.t()) / 2;
            S_cond = inv_sympd(S_cond);
            XDat(id_tmp, col_missing(i)).zeros();
            XDat(id_tmp, col_missing(i)) = - XDat.row(row_missing(i)) * LD \
                    * ML.rows(col_missing(i)).t() * S_cond;
            nS(col_missing(i), col_missing(i)) += S_cond + \
                    XDat(id_tmp, col_missing(i)).t() * XDat(id_tmp, col_missing(i));
            S_cond.clear();
        }
        nS += nS0;
        dnS = nS.diag();

        if (kappa==1)
        {
            c1 = nS;
            c2 = -VL;
            c2.each_row() += sum(VL);
            c2 *= LAMBDA(1); 
            //c2 = LAMBDA(1)*(repmat(sum(VL),p,1)-VL);
            c3 = LAMBDA;
            c4 = (VL+VL2)*mD2pvD(0);
			VL2.clear();
        }
        else
        {   
            LDL = LD.rows(id0)*ML.t();
            c1 *= r;
            c1.rows(id0) += psr*(nS.rows(id0)+LDL%LAMBDA.rows(id0))*ML;

            c2 *= r;
            c2.rows(id0) += psr*LAMBDA.rows(id0)*VL;

            c3 *= r;
            c3.rows(id0) += psr*LAMBDA.rows(id0)*ML2;
            
            K_tmp1 = ML2pVL.rows(id0);
            K_tmp1.each_row() %= mD2pvD.t();
            K_tmp2 = ML2.rows(id0);
            K_tmp2.each_row() %= mD2.t();
            c4 *=r;
            c4.rows(id0) += psr*(K_tmp1*ML2pVL.t()-K_tmp2*ML2.t()+square(LDL));
        }

        c2pc3 = c2+c3;
        gmL = -c1.elem(idl)%mD.elem(idc)-ML.elem(idl)%(mD2pvD.elem(idc)%c2.elem(idl)+vD.elem(idc)%c3.elem(idl));
        gvL = dnS.elem(idr)%mD.elem(idc)+c2pc3.elem(idl)%mD2pvD.elem(idc);
        gh = gmL+mL%gvL - h;
        gvL.elem(find(gvL<0)).zeros();
        gzeta = gvL - zeta;
        d1h = (1-1/tau)*d1h+gh/tau;
        d2h = (1-1/tau)*d2h+mean(square(gh))/tau;
        d1zeta = (1-1/tau)*d1zeta + gzeta/tau;
        d2zeta = (1-1/tau)*d2zeta + mean(square(gzeta))/tau;

        gmD = (VL.t()*dnS+trans(sum(ML%c1))+trans(sum(VL%(c2pc3+c3)))%mD)/2;
        gvD = trans(sum(ML2pVL%c2pc3))/4;
        for (i=0;i<p;i++) c5(i) = boost::math::trigamma(alpha(i));
        c6 = mD/(alpha%c5-1);
        alphatmp = nd2pp+c6/beta%gvD;
        alphatmp.elem(find(alphatmp<0)).zeros();
        betatmp = gmD+(1/beta+c5%c6)%gvD;
        betatmp.elem(find(betatmp<0)).zeros();
        galpha = alphatmp - alpha;
        gbeta = betatmp - beta;
        d1alpha = (1-1/tau)*d1alpha + galpha/tau;
        d2alpha = (1-1/tau)*d2alpha + mean(square(galpha))/tau;
        d1beta = (1-1/tau)*d1beta + gbeta/tau;
        d2beta = (1-1/tau)*d2beta + mean(square(gbeta))/tau;

        dtmp = omega/2*c4.elem(idl);
        dtmp.elem(find(dtmp<0)).zeros();
        gd = dtmp - d;
        d1d = (1-1/tau)*d1d + gd/tau;
        d2d = (1-1/tau)*d2d + mean(square(gd))/tau;

        btmp = sum(lambda%c4.elem(idl))/2;
        if (btmp<0) btmp = 0;
        gb = btmp - b;
        d1b = (1-1/tau)*d1b + gb/tau;
        d2b = (1-1/tau)*d2b + pow(gb,2)/tau;

        rho = (mean(square(d1h))+mean(square(d1zeta))+p/pe*(mean(square(d1alpha))+mean(square(d1beta)))+mean(square(d1d))+pow(d1b,2)/pe)/(d2h+d2zeta+p/pe*(d2alpha+d2beta)+d2d+d2b/pe);
        if (rho>eta) rho = eta;



        /*idd = find(gvL<0);
        if (!idd.is_empty())
        {
            rho_ub = -c7*max(zeta.elem(idd)/gzeta.elem(idd));
            idd.clear();
            if (rho>rho_ub) rho = rho_ub; //rho*pow(0.5,ceil(log2(rho/rho_ub)));
        }



        idd = find(alphatmp<0);
        if (!idd.is_empty())
        {
            rho_ub = -c7*max(alpha.elem(idd)/galpha.elem(idd));
            idd.clear();
            if (rho>rho_ub) rho = rho_ub; //rho*pow(0.5,ceil(log2(rho/rho_ub)));
        }



        idd = find(betatmp<0);
        if (!idd.is_empty())
        {
            rho_ub = -c7*max(beta.elem(idd)/gbeta.elem(idd));
            idd.clear();
            if (rho>rho_ub) rho = rho_ub; //rho*pow(0.5,ceil(log2(rho/rho_ub)));
        }



        idd = find(dtmp<0);
        if (!idd.is_empty())
        {
            rho_ub = -c7*max(d.elem(idd)/gd.elem(idd));
            idd.clear();
            if (rho>rho_ub) rho = rho_ub; //rho*pow(0.5,ceil(log2(rho/rho_ub)));
        }



        if (btmp<0)
        {
            rho_ub = -c7*b/gb;
            if (rho>rho_ub) rho = rho_ub;
        }*/
        tau = (1-rho)*tau + 1;
        h += rho*gh;
//         gvL = zeta;
        zeta += rho*gzeta;
//         zeta.elem(find(zeta<1e-6)) = 0.5*gvL.elem(find(zeta<1e-6));
//         alphatmp = alpha;
        alpha += rho*galpha;
//         alpha.elem(find(alpha<1e-6)) = 0.5*alphatmp.elem(find(alpha<1e-6));
//         betatmp = beta;
        beta += rho*gbeta;
//         beta.elem(find(beta<1e-6)) = 0.5*betatmp.elem(find(beta<1e-6));
//         dtmp = d;
        d += rho*gd;
//         d.elem(find(d<1e-6)) = 0.5*dtmp.elem(find(d<1e-6));
//         btmp = b;
        b += rho*gb;
//         if (b<1e-6) b = btmp/2;
        //if (b>a) b = a;


        if (kappa%100 == 0)
        {
            mLnew = h/zeta;
            lambdanew = lambda;
            difmL = sqrt(mean(square(mLnew-mLold))/mean(square(mLold)));
            diflambda = max(abs(lambdanew-lambdaold));
            difmax = max(abs(mLnew-mLold));
            mexPrintf("#no. of iterations = %d, difmL = %f, difmax = %f, diflambda = %f\n",kappa,difmL,difmax,diflambda);
            mexEvalString("drawnow;");
            if (difmL<tol)
                break;
            else
            {
                mLold = mLnew;
                lambdaold = lambdanew;
            }
        }

        idp = linspace<uvec>(0, p-1, p);
        for (i=0;i<s;i++)
        {
            k = randi<uword>(distr_param(0,p-1-i));
            id0(i) = idp(k);
            if (k!=p-1-i) idp.subvec(k,p-2-i) = idp.subvec(k+1,p-1-i);
        }
        
        LDL = LD.rows(id0)*ML.t();
        c1.rows(id0) -= psr*(nS.rows(id0)+LDL%LAMBDA.rows(id0))*ML;
        c2.rows(id0) -= psr*LAMBDA.rows(id0)*VL;
        c3.rows(id0) -= psr*LAMBDA.rows(id0)*ML2;
        
        K_tmp1 = ML2pVL.rows(id0);
        K_tmp1.each_row() %= mD2pvD.t();
        K_tmp2 = ML2.rows(id0);
        K_tmp2.each_row() %= mD2.t();
        c4.rows(id0) -= psr*(K_tmp1*ML2pVL.t()-K_tmp2*ML2.t()+square(LDL));
    }




    ML.elem(idl) = h/zeta;
    VL.elem(idl) = 1/zeta;
    mD = alpha/beta;
    vD = mD/beta;
    double ElapsedTime = (clock()-timeBegin)/CLOCKS_PER_SEC;
    if (kappa < maxIter || difmL<tol) mexPrintf("BISN converges, elapsed time is %f seconds.\n",ElapsedTime);
    else mexPrintf("BISN reaches the maximum number of iterations, elapsed time is %f seconds.\n",ElapsedTime);

    /* Output Interface */
    if (nlhs>3)
    {
        plhs[0] = armaCreateMxMatrix(p,p,mxDOUBLE_CLASS,mxREAL);
        armaSetPr(plhs[0],ML);
        plhs[1] = armaCreateMxMatrix(p,p,mxDOUBLE_CLASS,mxREAL);
        armaSetPr(plhs[1],VL);
        plhs[2] = armaCreateMxMatrix(p,1,mxDOUBLE_CLASS,mxREAL);
        armaSetPr(plhs[2],mD);
        plhs[3] = armaCreateMxMatrix(p,1,mxDOUBLE_CLASS,mxREAL);
        armaSetPr(plhs[3],vD);
        if (nlhs>4)
        {
            omega = a/b;
            plhs[4] = mxCreateDoubleScalar(omega);
        }
        if (nlhs>5)
        {
            //lambda = c/d;
            for (i=0;i<pe;i++) lambda(i)=(d(i)>10)?Lentz_Algorithm(d(i)):-boost::math::expint(-d(i));
            lambda.elem(find(d<=10)) %= exp(d.elem(find(d<=10)));
            lambda = 1/(d%lambda)-1;
            plhs[5] = armaCreateMxMatrix(pe,1,mxDOUBLE_CLASS,mxREAL);
            armaSetPr(plhs[5],lambda);
        }
        if (nlhs>6) plhs[6] = mxCreateDoubleScalar(ElapsedTime);
    }
    else
    {
        mexErrMsgIdAndTxt("BINS:output","Expected at least four output arguments");
    }

}
