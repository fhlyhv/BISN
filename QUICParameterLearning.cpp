#include <armadillo>
#include "armaMex.hpp"
using namespace arma;


void mexFunction(int nlhs,mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
    
    mat K0 = armaGetPr(prhs[0]);
    mat S = armaGetPr(prhs[1]); 
    uvec idr = conv_to<uvec>::from(armaGetPr(prhs[2])) - 1; 
    uvec idc = conv_to<uvec>::from(armaGetPr(prhs[3])) - 1;  
    uword max_outer_iter, max_inner_iter;
    if (nrhs > 4) max_outer_iter = (uword)armaGetDouble(prhs[4]);
    else max_outer_iter = 100;
    if (nrhs > 5) max_inner_iter = (uword)armaGetDouble(prhs[5]);
    else max_inner_iter = 100;
    
    uword p = K0.n_cols, iter_outer, iter_inner, p_od = idr.n_elem, i;
    uvec idl = idc * p + idr, idu = idr * p + idc, idd = linspace<uvec>(0, p - 1, p), ida = join_cols(idd * p + idd, idl);
    vec Sida = S(ida), Sidu = S(idu), Did(p + p_od), gradK, Kd = K0.diag();
    mat U(p, p, fill::zeros), W, Kh(p, p, fill::zeros);
    double a, b, mu, diffD = 0;
    
    
    double objh, xi0 = 1, xi, sum_grad2;
    while (! K0.is_sympd()) {
        K0 *= 0.9;
        K0.diag() = Kd;
    }
    double obj0 = - 2*accu(log(mat(chol(K0)).diag())) + accu(Sida % K0(ida)) + accu(Sidu % K0(idu));
    
    for (iter_outer = 0; iter_outer < max_outer_iter; iter_outer ++) {
        W = inv_sympd(K0);
        gradK = Sida - W(ida);
        Did.zeros();
        U.zeros();
        for (iter_inner = 0; iter_inner < max_inner_iter; iter_inner ++) {
            for (i = 0; i < p; i++) {
                a = pow(W(i, i), 2);
                b = gradK(i) + accu(W.col(i) % U.col(i));
                mu = - b / a;
                Did(i) += mu;
                U.row(i) += mu * W.row(i);
                diffD += fabs(mu);
            }
            
            for (i = 0; i < p_od; i ++) {
                a = pow(W(idl(i)), 2) + W(idr(i), idr(i)) * W(idc(i), idc(i));
                b = gradK(p + i) + accu(W.col(idr(i)) % U.col(idc(i)));
                mu = - b / a;
                Did(p + i) += mu;
                U.row(idr(i)) += mu * W.row(idc(i));
                U.row(idc(i)) += mu * W.row(idr(i));
                diffD += fabs(mu);
            }
            
            if (diffD < 0.05 * accu(abs(Did))) break;
            else diffD = 0;
        }
        sum_grad2 = accu(Did % gradK);
        xi = xi0;
        while (true) {
            Kh(ida) = K0(ida) + xi * Did;
            Kh(idu) = Kh(idl);
            if (Kh.is_sympd()) {
                objh = - 2*accu(log(mat(chol(Kh)).diag())) + accu(Sida % Kh(ida)) + accu(Sidu % Kh(idu));
                if (objh <= obj0 + 1e-3 * xi * sum_grad2)
                    break;
                else
                    xi /= 2;
            } else
                xi /=2;
        }
        // printf("xi = %e\n", xi);
        if (abs(Kh - K0).max() < 1e-10 && fabs(obj0 - objh ) < 1e-10) {
            break;
        } else {
            obj0 = objh;
            K0 = Kh;
        }
    }
    
    plhs[0] = armaCreateMxMatrix(p,p,mxDOUBLE_CLASS,mxREAL);
    armaSetPr(plhs[0],Kh);
}