#include "initcon.h"


namespace mhd
{
  // Initial conditions
  template <int dim>
  void InitialValues<dim>::mhdBlast(const Point<dim> &p,
                                        Vector<double> &v) const
  {
    const double GAMMA=5.0/3.0;

    if (p.norm()<0.1){ // 0.1
        v(7)=10.0/(GAMMA-1.0)+1.0; // U
    }else{
        v(7)=0.1/(GAMMA-1.0)+1.0; // U
    } 
    v(0)=1.0; // rho
    v(1)=0.0; // v
    v(2)=0.0;
    v(3)=0.0;
    v(4)=0.0/sqrt(2); // B
    v(5)=0.0/sqrt(2);
    v(6)=0.0;
    
    v(8)=0.0;  // J
    v(9)=0.0;
    v(10)=0.0;
  }
  
  template <int dim>
  void InitialValues<dim>::harris(const Point<dim> &p,
                                        Vector<double> &v) const
  {
    const double GAMMA=5.0/3.0;
    double yy;
    double pressure;
    v(0)=1.0; // rho
    v(1)=0.0;//-.08*p[0]*p[0]*p[0]*exp(-p[0]*p[0]/12.0)*exp(-(p[1]-20)*(p[1]-20)/8.0); // v
    v(2)=0.0;
    v(3)=0.0;
    yy=p[1]-20.0;
    v(4)=4e-2*yy*exp(-(p[0]*p[0]+yy*yy)/8.0); // B
    v(5)=std::tanh(p[0])-4e-2*p[0]*exp(-(p[0]*p[0]+yy*yy)/8.0);
    v(6)=0.0;
    pressure=0.05+1.0-v(5)*v(5);
    v(7)=pressure/(GAMMA-1.0)+v(5)*v(5)+v(1)*v(1); // U
    v(8)=0.0;  // J
    v(9)=0.0;
    v(10)=1.0/(std::cosh(p[0])*std::cosh(p[0]));
    
  }
  
  template <int dim>
  void InitialValues<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                              std::vector<Vector<double> >   &value_list) const
  {
    const unsigned int n_points = points.size();
    for(unsigned int p=0; p<n_points; ++p)
      (this->*vectorValue)(points[p], value_list[p]);
  }
  
  template <int dim>
  InitialValues<dim>::InitialValues() : Function<dim>(Nv)
  {
    vectorValue=&InitialValues<dim>::mhdBlast;
  }
  
  template <int dim>
  void InitialValues<dim>::setInitialCondition(int cond)
  {
    switch(cond){
      case 0:
        vectorValue=&InitialValues<dim>::mhdBlast;
        break;
      case 1:
        vectorValue=&InitialValues<dim>::harris;
        break;
    }
  }
  
  template <int dim>
  void InitialValues<dim>::vector_value(const Point<dim> &point, Vector<double>&value) const
  {
    (this->*vectorValue)(point, value);
  }

} // end of namespace mhd