/* ORIGINAL OPTIMIZATION PROBLEM
 * solved using QP
 *
 * the problem is
 *   Minimize  (1/2) <w,w>>
 *   subj. to  y_n (<w,x_n> + b) >= 1  ; for n = 1 ... N
 */


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <numeric>

#include <CGAL/basic.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
// choose exact floating-point type
#include <CGAL/Gmpzf.h>
using ET = CGAL::Gmpzf;

using namespace std;

tuple<vector<vector<CGAL::Gmpzf>>,vector<CGAL::Gmpzf>> parse_data(istream &in) {
    vector<vector<CGAL::Gmpzf>> xs;
    vector<CGAL::Gmpzf>         ys;

    string line;
    while(getline(in,line)) {
        xs.push_back({});
        auto & x = xs.back();

        istringstream iss{line};
        string tmp;
        while(getline(iss,tmp,',')) {
            istringstream iss_(tmp);
            CGAL::Gmpzf gmpzf;
            iss_ >> gmpzf;
            x.push_back(gmpzf);
        }
        ys.push_back(x.back());
        x.pop_back();
    }

    return {xs,ys};
}

int main(int argc, char *argv[]) {
    string filename{argv[1]};

    ifstream in{filename};

    int                         d;
    int                         N;
    vector<vector<CGAL::Gmpzf>> x;
    vector<CGAL::Gmpzf>         y;

    tie(x,y) = parse_data(in);
    d = x.front().size();
    N = y.size();


    /* Quadratic Programming
     */

    // program and solution types
    using Program  = CGAL::Quadratic_program<CGAL::Gmpzf>;
    using Solution = CGAL::Quadratic_program_solution<ET>;

    Program qp(CGAL::LARGER, false, 0, false, 0);

    // A
    for(int n=0; n<N; ++n) {
        for(int i=0; i<d; ++i) {
            qp.set_a(i,n,x[n][i]);
        }
        qp.set_a(d,n,1);
        if(y[n] < 0) {
            qp.set_r(n, CGAL::SMALLER);
        }
        qp.set_b(n,y[n]);
    }

    // D
    for(int i=0; i<d; ++i) {
        qp.set_d(i,i,CGAL::Gmpzf(0.5));
    }
    qp.set_d(d,d,0);

	Solution s = CGAL::solve_quadratic_program(qp, ET());

    vector<CGAL::Quotient<CGAL::Gmpzf>> w(d+1);
    int n = 0;
    for(auto it=s.variable_values_begin(); it!=s.variable_values_end(); ++it, ++n) {
        w[n] = *it;
    }

    cout << "(" << to_double(w[0]) << ", " << to_double(w[1]) << ")" << " " << to_double(w[2]) << endl;
}
