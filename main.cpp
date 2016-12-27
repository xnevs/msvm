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

#include <CGAL/Gmpzf.h>

template <typename NT,
          typename ET,
          typename InstanceIterator,
          typename LabelIterator,
          typename Kernel>
auto learn(InstanceIterator instance_begin,
           InstanceIterator instance_end,
           LabelIterator    label_begin,
           Kernel const &   K) {

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
            ++label_it2;
        }
        ++instance_it1;
        ++label_it1;
    }

    //set c
    for(int n=0; n<N; ++n) {
        qp.set_c(n, -2.0);
    }

	Solution s = CGAL::solve_nonnegative_quadratic_program(qp, ET());

    std::vector<CGAL::Quotient<ET>>                                          alpha;
    std::vector<typename std::iterator_traits<InstanceIterator>::value_type> support_vectors;
    std::vector<typename std::iterator_traits<LabelIterator>::value_type>    support_labels;
    instance_it1 = instance_begin;
    label_it1    = label_begin;
    for(auto it=s.variable_values_begin(); it!=s.variable_values_end(); ++it) {
        auto alpha_n = *it;
        if(alpha_n > 0) {
            alpha.push_back(alpha_n);
            support_vectors.push_back(*instance_it1);
            support_labels.push_back(*label_it1);
        }
        ++instance_it1;
        ++label_it1;
    }

    CGAL::Quotient<ET> b = support_labels.front();
    for(int n=0; n<alpha.size(); ++n) {
        b -= alpha[n] * support_labels[n] * K(support_vectors[n],support_vectors.front());
    }

    return [alpha,support_vectors,support_labels,b,K](auto instance) {
        CGAL::Quotient<ET> result;
        for(int n=0; n<alpha.size(); ++n) {
            result += alpha[n] * support_labels[n] * K(support_vectors[n],instance);
        }
        result += b;
        return (result > 0 ? 1 : -1);
    };
}


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
    using NT = CGAL::Gmpzf;
    using ET = CGAL::Gmpzf;

    string filename{argv[1]};

    ifstream in{filename};

    int                         d;
    int                         N;
    vector<vector<CGAL::Gmpzf>> x;
    vector<CGAL::Gmpzf>         y;

    tie(x,y) = parse_data(in);
    d = x.front().size();
    N = y.size();

    struct K {
        NT operator()(vector<NT> const & a, vector<NT> const & b) const {
            return std::inner_product(begin(a), end(a), begin(b), NT(0));
        }
    };

	auto model = learn<NT,ET>(begin(x), end(x), begin(y), K());

    int count = 0;
    for(int n=0; n<N; ++n) {
        auto predicted = model(x[n]);
        if(predicted == y[n]) {
            ++count;
        }
    }
    cout << count << endl;
}
