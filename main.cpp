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


template <typename Kernel,
          typename ET,
          typename Instance,
          typename Label>
struct svm_model {
    Kernel                          kernel;
    std::vector<CGAL::Quotient<ET>> alpha;
    CGAL::Quotient<ET>              b;

    std::vector<Instance> support_vectors;
    std::vector<Label>    support_labels;

    CGAL::Quotient<ET> objective_value;

    svm_model(Kernel const & kernel,
              std::vector<CGAL::Quotient<ET>> const & alpha,
              CGAL::Quotient<ET> const & b,
              std::vector<Instance> const & support_vectors,
              std::vector<Label> const & support_labels,
              CGAL::Quotient<ET> const & objective_value)
      : kernel(kernel), alpha(alpha), b(b),
        support_vectors(support_vectors),
        support_labels(support_labels),
        objective_value(objective_value) {}

    bool operator()(Instance const & instance) const {
        CGAL::Quotient<ET> result(0);
        for(int n=0; n<alpha.size(); ++n) {
            result += alpha[n] * support_labels[n] * kernel(support_vectors[n],instance);
        }
        result += b;
        return result > 0;
    }
};
template <typename Kernel,
          typename ET,
          typename Instance,
          typename Label>
svm_model<Kernel,ET,Instance,Label> make_svm_model(
        Kernel const & kernel,
        std::vector<CGAL::Quotient<ET>> const & alpha,
        CGAL::Quotient<ET> const & b,
        std::vector<Instance> const & support_vectors,
        std::vector<Label> const & support_labels,
        CGAL::Quotient<ET> const & objective_value) {
    return svm_model<Kernel,ET,Instance,Label>(kernel,alpha,b,support_vectors,support_labels, objective_value);
}

template <typename NT,
          typename ET,
          typename Kernel,
          typename InstanceIterator,
          typename LabelIterator>
auto learn(NT C,
           InstanceIterator instance_begin,
           InstanceIterator instance_end,
           LabelIterator    label_begin,
           Kernel const &   kernel=Kernel()) {

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
     *             0 <= alpha_n <= C ; for n=1...N
     */
    
    /* Nonnegative Quadratic program with constraint Ax == b
     * lower bound 0, upper bound C
     */
    Program qp(CGAL::EQUAL, true, NT(0), true, C);

    // set A
    label_it1 = label_begin;
    for(int n=0; n<N; ++n) {
        qp.set_a(n,0,*label_it1);
        ++label_it1;
    }
    // set b
    qp.set_b(0,NT(0));

    
    // set D
    instance_it1 = instance_begin;
    label_it1 = label_begin;
    for(int n=0; n<N; ++n) {
        instance_it2 = instance_begin;
        label_it2 = label_begin;
        for(int m=0; m<=n; ++m) {
            auto val = (*label_it1) * (*label_it2) * kernel(*instance_it1, *instance_it2);  // specify 2d !!, so there is no NT(0.5) factor
            qp.set_d(n, m, val);
            ++instance_it2;
            ++label_it2;
        }
        ++instance_it1;
        ++label_it1;
    }

    // set c
    for(int n=0; n<N; ++n) {
        qp.set_c(n, NT(-1.0));
    }

    Solution s = CGAL::solve_quadratic_program(qp, ET());

    std::vector<CGAL::Quotient<ET>>                                          alpha;
    std::vector<typename std::iterator_traits<InstanceIterator>::value_type> support_vectors;
    std::vector<typename std::iterator_traits<LabelIterator>::value_type>    support_labels;

    instance_it1 = instance_begin;
    label_it1    = label_begin;
    for(auto it=s.variable_values_begin(); it!=s.variable_values_end(); ++it) {
        auto alpha_n = *it;
        if(alpha_n > NT(0)) {
            alpha.push_back(alpha_n);
            support_vectors.push_back(*instance_it1);
            support_labels.push_back(*label_it1);
        }
        ++instance_it1;
        ++label_it1;
    }

    CGAL::Quotient<ET> b = support_labels.front();
    for(int n=0; n<alpha.size(); ++n) {
        b -= alpha[n] * support_labels[n] * kernel(support_vectors[n],support_vectors.front());
    }

    return make_svm_model(kernel,alpha,b,support_vectors,support_labels,s.objective_value());
}

template<typename NT,
         typename ET>
std::tuple<std::vector<NT>,NT> optimize(std::vector<std::vector<NT>> const & constraints) {
    auto K = constraints.front().size();

    using Program  = CGAL::Quadratic_program<NT>;
    using Solution = CGAL::Quadratic_program_solution<ET>;

    Program lp(CGAL::LARGER, true, 0, false, 0);

    // theta unconstrained
    lp.set_l(K,false,0);

    // set A
    int A_row = 0;
    for(int k=0; k<K; ++k) {
        lp.set_a(k,A_row,NT(1));
    }
    lp.set_r(A_row,CGAL::EQUAL);
    lp.set_b(A_row,NT(1));
    ++A_row;
    for(auto & constraint : constraints) {
        int k = 0;
        for(auto & S_k : constraint) {
            lp.set_a(k,A_row,S_k);
            ++k;
        }
        lp.set_a(k,A_row,NT(-1));
        ++A_row;
    }

    // set c
    lp.set_c(K,NT(-1));

    Solution s = CGAL::solve_linear_program(lp, ET());

    std::vector<NT> beta;
    NT              theta;

    for(auto it=s.variable_values_begin(); it!=s.variable_values_end(); ++it) {
        beta.emplace_back(CGAL::to_double(*it));
    }
    theta = beta.back();
    beta.pop_back();

    return {beta,theta};
}

template <typename NT,
          typename ET,
          typename KernelIterator,
          typename InstanceIterator,
          typename LabelIterator>
auto multi_learn(NT C,
                 InstanceIterator instance_begin,
                 InstanceIterator instance_end,
                 LabelIterator    label_begin,
                 KernelIterator   kernel_begin,
                 KernelIterator   kernel_end) {

    using instance_type = typename std::iterator_traits<InstanceIterator>::value_type;

    struct CombinedKernel {
        KernelIterator  kernel_begin;
        KernelIterator  kernel_end;
        std::vector<NT> beta;

        CombinedKernel(KernelIterator kernel_begin,
                       KernelIterator kernel_end)
          : kernel_begin(kernel_begin),
            kernel_end(kernel_end),
            beta(std::distance(kernel_begin,kernel_end), NT(1.0/std::distance(kernel_begin,kernel_end))) {}

        NT operator()(instance_type const & a, instance_type const & b) const {
            NT result(0);
            auto b_it = begin(beta);
            for(auto k_it=kernel_begin; k_it!=kernel_end; ++k_it) {
                result += (*b_it) * (*k_it)(a,b);
                ++b_it;
            }
            return result;
        }
    };

    auto N = std::distance(instance_begin, instance_end);
    auto K = std::distance(kernel_begin, kernel_end);

    CombinedKernel kernel(kernel_begin, kernel_end);

    auto model = learn<NT,ET>(C, instance_begin, instance_end, label_begin, kernel);

    CGAL::Quotient<ET>     S(model.objective_value);
    NT                 theta;

    std::vector<std::vector<NT>> constraints;

    CGAL::Quotient<ET> eps;
    do {
        constraints.emplace_back();
        auto & last_constraint = constraints.back();
        for(int k=0; k<K; ++k) {
        /*
            CGAL::Quotient<ET> S_k(0);
            for(int n=0; n<model.alpha.size(); ++n) {
                for(int m=0; m<model.alpha.size(); ++m) {
                    S_k += NT(0.5) * model.alpha[n] * model.alpha[m] * model.support_labels[n] * model.support_labels[m] * kernel.kernel_begin[k](model.support_vectors[n], model.support_vectors[m]);
                }
                S_k -= model.alpha[n];
            }
        */
        /*
            CGAL::Quotient<ET>     S_k(0);
            for(int n=0; n<model.alpha.size(); ++n) {
                auto alpha_n = model.alpha[n];
                auto label_n = model.support_labels[n];
                auto instance_n = model.support_vectors[n];
                for(int m=0; m<model.alpha.size(); ++m) {
                    S_k += NT(0.5) * alpha_n * model.alpha[m] * label_n * model.support_labels[m] * kernel.kernel_begin[k](instance_n,model.support_vectors[m]);
                }
                S_k -= alpha_n;
            }
        */
        /*
            CGAL::Quotient<ET> S_k(0);
            for(int n=0; n<model.alpha.size(); ++n) {
                for(int m=0; m<n; ++m) {
                    S_k += model.alpha[n] * model.alpha[m] * model.support_labels[n] * model.support_labels[m] * kernel.kernel_begin[k](model.support_vectors[n], model.support_vectors[m]);
                }
                S_k += NT(0.5) * model.alpha[n] * model.alpha[n] * model.support_labels[n] * model.support_labels[n] * kernel.kernel_begin[k](model.support_vectors[n], model.support_vectors[n]);
                S_k -= model.alpha[n];
            }
        */
            CGAL::Quotient<ET>     S_k(0);
            for(int n=0; n<model.alpha.size(); ++n) {
                auto & alpha_n = model.alpha[n];
                auto & label_n = model.support_labels[n];
                auto & instance_n = model.support_vectors[n];
                for(int m=0; m<n; ++m) {
                    S_k += alpha_n * model.alpha[m] * label_n * model.support_labels[m] * kernel.kernel_begin[k](instance_n,model.support_vectors[m]);
                }
                S_k += NT(0.5) * alpha_n * alpha_n * label_n * label_n * kernel.kernel_begin[k](instance_n,instance_n);
                S_k -= alpha_n;
            }
            last_constraint.emplace_back(CGAL::to_double(S_k));
        }

        std::tie(kernel.beta,theta) = optimize<NT,ET>(constraints);

        model = learn<NT,ET>(C, instance_begin, instance_end, label_begin, kernel);
        S = model.objective_value;

        eps = NT(1) - (S/theta);
        if(eps < 0) eps = -eps;
        std::cout << CGAL::to_double(eps) << std::endl;
    } while(eps > 0.1);
    
    return model;
}


using namespace std;

tuple<vector<vector<CGAL::Gmpzf>>,vector<CGAL::Gmpzf>> parse_data(istream &in) {
    vector<vector<CGAL::Gmpzf>> xs;
    vector<CGAL::Gmpzf>         ys;

    string line;
    while(getline(in,line)) {
        xs.emplace_back();
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

template<typename NT>
NT InnerProduct(vector<NT> const & a, vector<NT> const & b){
    return std::inner_product(begin(a), end(a), begin(b), NT(0));
}

template<typename NT>
NT PolynomialKernel5(vector<NT> const & a, vector<NT> const & b){
    auto val = std::inner_product(begin(a), end(a), begin(b), NT(0)) + 500;
    auto val5 = val*val;
    val5 *= val5;
    val5 *= val;
    return val5;
}

template<typename NT>
NT RBFKernel(vector<NT> const & a, vector<NT> const & b){
    int d = a.size();

    NT val = 0;
    for(int i=0; i<d; ++i) {
        auto temp = a[i]-b[i];
        val += temp*temp;
    }
    val *= -5;

    return NT(exp(to_double(val)));
}

int main(int argc, char *argv[]) {
    using NT = CGAL::Gmpzf;
    using ET = CGAL::Gmpzf;

    string filename{argv[1]};

    ifstream in{filename};

    int                d;
    int                N;
    vector<vector<NT>> x;
    vector<NT>         y;

    tie(x,y) = parse_data(in);
    d = x.front().size();
    N = y.size();

    using kernel_type = NT(*)(vector<NT> const &, vector<NT> const &);
    vector<kernel_type> kernels{PolynomialKernel5<NT>, InnerProduct<NT>, RBFKernel<NT>};

    auto model = multi_learn<NT,ET>(1000, begin(x), end(x), begin(y), begin(kernels),end(kernels));
    //auto model = learn<NT,ET>(1000, begin(x), end(x), begin(y), *begin(kernels));

    std::cout << "beta: ";
    for(auto b : model.kernel.beta) {std::cout << CGAL::to_double(b) << " ";} std::cout << std::endl;

    int count = 0;
    for(int n=0; n<N; ++n) {
        auto predicted = (model(x[n]) ? 1 : -1);
        if(predicted == y[n]) {
            ++count;
        }
    }
    cout << count << endl;
}
