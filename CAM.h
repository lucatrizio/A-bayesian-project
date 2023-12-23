#include <math.h>
#include "vector"
#include "armadillo.h"

struct dimensions{
    int J; //n of persons
    int K; //n of DC
    int L; //n of OC
    int V; //dimensions of one data
}; //non  credo nj serva


class chain{
private:
    field<mat> data
    dimensions dim;
    vec pi;
    vec alpha;
    mat w;
    mat beta;
    Col<int> S;
    Mat<int> M;
public:
};
