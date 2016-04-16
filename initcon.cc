#include "initcon.h"


namespace mhd
{
  // Initial conditions
  template <int dim>
  void InitialValues<dim>::mhdBlast(const Point<dim> &p,
                                        Vector<double> &v) const
  {
    if (p.norm()<0.1){ // 0.1
        v(7)=10.0/(GAMMA-1.0)+1.0; // U
    }else{
        v(7)=0.1/(GAMMA-1.0)+1.0; // U
    } 
    v(0)=1.0; // rho
    v(1)=0.0; // v
    v(2)=0.0;
    v(3)=0.0;
    v(4)=1.0/sqrt(2); // B
    v(5)=1.0/sqrt(2);
    v(6)=0.0;
    
    v(8)=0.0;  // J
    v(9)=0.0;
    v(10)=0.0;
    
    v(11)=0.0; // eta
  }
  
  template <int dim>
  void InitialValues<dim>::harris(const Point<dim> &p,
                                        Vector<double> &v) const
  {
    double yy,xx;
    double pressure;
    
    xx=p[0]-(box[1][0]+box[0][0])*0.5;
    yy=p[1]-(box[1][1]+box[0][1])*0.5;
    
    v(0)=1.0; // rho
    v(1)=0.0;//-.08*p[0]*p[0]*p[0]*exp(-p[0]*p[0]/12.0)*exp(-(p[1]-20)*(p[1]-20)/8.0); // v
    v(2)=0.0;
    v(3)=0.0;
    v(4)=4e-2*yy*exp(-(xx*xx+yy*yy)/8.0); // B
    v(5)=std::tanh(xx)-4e-2*xx*exp(-(xx*xx+yy*yy)/8.0);
    v(6)=0.0;
    pressure=0.05+1.0-v(5)*v(5);
    v(7)=pressure/(GAMMA-1.0)+v(5)*v(5)+v(1)*v(1); // U
    v(8)=0.0;  // J
    v(9)=0.0;
    v(10)=1.0/(std::cosh(xx)*std::cosh(xx));
    
    v(11)=0.0; // eta
  }
  
  template <int dim>
  void InitialValues<dim>::debug(const Point<dim> &/*p*/,
                                        Vector<double> &v) const
  {
    v(0)=1.0;
    v(1)=2.0;
    v(2)=3.0;
    v(3)=4.0;
    v(4)=5.0;
    v(5)=6.0;
    v(6)=7.0;
    v(7)=8.0/(GAMMA-1.0)+v(1)*v(1)+v(2)*v(2)+v(3)*v(3)
                        +v(4)*v(4)+v(5)*v(5)+v(6)*v(6);
    v(8)=9.0;
    v(9)=10.0;
    v(10)=11.0;
    
    v(11)=0.0; // eta
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
  void InitialValues<dim>::setParameters(Params &pars)
  {
    int initCond;
    
    pars.prm.enter_subsection("Simulation");
    {
      initCond=pars.prm.get_integer("Initial condition");
      GAMMA=pars.prm.get_double("gamma");
      ETApar1=pars.prm.get_double("eta param 1");
      ETApar2=pars.prm.get_double("eta param 2");
      
      pars.prm.enter_subsection("Box");
      {
        box[0][0]=pars.prm.get_double("x_min");
        box[0][1]=pars.prm.get_double("y_min");
        if (dim==3) box[0][2]=pars.prm.get_double("z_min");
        box[1][0]=pars.prm.get_double("x_max");
        box[1][1]=pars.prm.get_double("y_max");
        if (dim==3) box[1][2]=pars.prm.get_double("z_max");
      }
      pars.prm.leave_subsection();
    }
    pars.prm.leave_subsection();
    
    switch(initCond){
      case 0:
        vectorValue=&InitialValues<dim>::mhdBlast;
        break;
      case 1:
        vectorValue=&InitialValues<dim>::harris;
        break;
      case 2:
        vectorValue=&InitialValues<dim>::debug;
        break;
    }
  }
  
  template <int dim>
  void InitialValues<dim>::vector_value(const Point<dim> &point, Vector<double>&value) const
  {
    (this->*vectorValue)(point, value);
  }

} // end of namespace mhd