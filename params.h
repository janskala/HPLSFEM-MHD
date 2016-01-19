#ifndef _PARAMS_
#define _PARAMS_

#include <deal.II/base/parameter_handler.h>
#include <list>
#include <iostream>

namespace mhd
{
  
  using namespace dealii;
  
  class Params
  {
  public:
    Params(const int, char *const *);
    ~Params();
    
    ParameterHandler   prm;
    void setGravity(double*);
    void setWeights(double*);
    void setBC(int*);
    int getDim();

  private:
    void print_usage_message();
    void declare_parameters();
  };
  
}

#endif //_PARAMS_