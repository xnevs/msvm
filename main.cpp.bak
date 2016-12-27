#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <numeric>
#include <functional>

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

template <typename NT,
          typename ET,
          typename InstanceIterator,
          typename LabelIterator,
          typename Kernel>
void learn(InstanceIterator instance_begin,
           InstanceIterator instance_end,
           LabelIterator label_begin,
           Kernel K=std::bind(std::inner_product(std::placeholders::_1, std::placeholders::_2, NT(0)))) {
    InstanceIterator instance_it1;
    InstanceIterator instance_it2;
    LabelIterator    label_it1;
    LabelIterator    label_it2;

    auto N = std::distance(instance_begin, instance_end);

    // define program and solution types
    using Program  = CGAL::Quadratic_program<NT>;
    using Solution = CGAL::Quadratic_program_solution<ET>;

    /* Quadratic program
     *   Minimize: alpha' . D . alpha + c' . alpha
     *   subj. to:     <y,alpha> = 0
     *             0 <= alpha_n <= inf ; for n=1...N
     */
    
    // Nonnegative Quadratic program with constraint Ax == b
    Program qp(CGAL::EQUAL, true, 0, false, 0);

    // set A
    label_it1 = label_begin;
    for(int n=0; n<N; ++n) {
        qp.set_a(n,0,*label_it1);
        ++label_it1;
    }
    // set b
    qp.set_b(0.0,0.0);
    
    // set D
    instance_it1 = instance_begin;
    label_it1 = label_begin;
    for(int n=0; n<N; ++n) {
        instance_it2 = instance_begin;
        label_it2 = label_begin;
        for(int m=0; m<=n; ++m) {
            auto val = (*label_it1) * (*label_it2) * K(*instance_it1, *instance_it2);
            qp.set_d(n, m, val);
            ++instance_it2;
        }
        ++instance_it1;
    }

    //set c
    for(int n=0; n<N; ++n) {
        qp.set_c(n, -2.0);
    }

	Solution s = CGAL::solve_nonnegative_quadratic_program(qp, ET());
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

    Program qp(CGAL::EQUAL, true, 0, false, 0);

    // A
    for(int n=0; n<N; ++n) {
        qp.set_a(n,0,y[n]);
    }
    qp.set_b(0.0,0.0);

    // D
    for(int n=0; n<N; ++n) {
        for(int m=0; m<=n; ++m) {
            auto val = y[n]*y[m]*inner_product(begin(x[n]),end(x[n]),begin(x[m]),CGAL::Gmpzf(0.0));
            qp.set_d(n, m, val);
        }
    }

    //c
    for(int n=0; n<N; ++n) {
        qp.set_c(n, -2.0);
    }

	Solution s = CGAL::solve_nonnegative_quadratic_program(qp, ET());

    vector<CGAL::Quotient<CGAL::Gmpzf>> w(d);
    int n = 0;
    for(auto it=s.variable_values_begin(); it!=s.variable_values_end(); ++it, ++n) {
        auto alpha_n = *it;
        for(int i=0; i<d; ++i) {
            w[i] += alpha_n * y[n] * x[n][i];
        }
    }

    cout << "(" << to_double(w[0]) << ", " << to_double(w[1]) << ")" << endl;
}
