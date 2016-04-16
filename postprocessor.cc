#include "postprocessor.h"
#include "equations.h"

#include<iostream>

namespace mhd
{
  
  template <int dim>
  Postprocessor<dim>::Postprocessor()
  {
    
  }

  template <int dim>
  std::vector<std::string>
  Postprocessor<dim>::get_names() const
  {
    std::vector<std::string> solution_names;
    const char names[3][Nv][4]={ {"rho","u","u_y","u_z","B","B_y","B_z","p","J","J_y","J_z","eta"},
                          {"rho","u","u","u_z","B","B","B_z","p","J","J","J_z","eta"},
                          {"rho","u","u","u","B","B","B","p","J","J","J","eta"}};
    
    for(unsigned int i=0;i<Nv;i++)
        solution_names.push_back(names[dim-1][i]);

    return solution_names;
  }

  template <int dim>
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    Postprocessor<dim>::get_data_component_interpretation() const
  {
    DataComponentInterpretation::DataComponentInterpretation cmpIntpr[Nv];
    std::fill_n(cmpIntpr, Nv, DataComponentInterpretation::component_is_scalar);
    for(unsigned int i=0;i<dim;i++)
      cmpIntpr[1+i]=cmpIntpr[4+i]=cmpIntpr[8+i]=
            DataComponentInterpretation::component_is_part_of_vector;

    std::vector<DataComponentInterpretation::DataComponentInterpretation> 
                              data_component_interpretation;

    for(unsigned int i=0;i<Nv;i++)
      data_component_interpretation.push_back(cmpIntpr[i]);

    return data_component_interpretation;
  }

  template <int dim>
  UpdateFlags Postprocessor<dim>::get_needed_update_flags() const
  {
    return update_values | update_q_points; // | update_gradients
  }
  
  template <int dim>
  void Postprocessor<dim>::
  compute_derived_quantities_vector(const std::vector<Vector<double> >              &uh,
                                    const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                    const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
                                    const std::vector<Point<dim> >                  &/*normals*/,
                                    const std::vector<Point<dim> >                  &/*evaluation_points*/,
                                    std::vector<Vector<double> >                    &comp_quan) const
  {
    const unsigned int n_quadrature_points = uh.size();
    Assert (duh.size() == n_quadrature_points,                  ExcInternalError());
    Assert (comp_quan.size() == n_quadrature_points,  ExcInternalError());

    for(unsigned int q=0; q<n_quadrature_points; ++q){
      double iRh=1.0/uh[q][0];
      comp_quan[q][0]=uh[q][0];     // rho
      comp_quan[q][1]=uh[q][1]*iRh; // v
      comp_quan[q][2]=uh[q][2]*iRh;
      comp_quan[q][3]=uh[q][3]*iRh;
      comp_quan[q][4]=uh[q][4];     // B
      comp_quan[q][5]=uh[q][5];
      comp_quan[q][6]=uh[q][6];
      comp_quan[q][7]=((5./3.)-1.)*(uh[q][7]  // p
              -(uh[q][1]*uh[q][1]+uh[q][2]*uh[q][2]+uh[q][3]*uh[q][3])*iRh
              -(uh[q][4]*uh[q][4]+uh[q][5]*uh[q][5]+uh[q][6]*uh[q][6]));
      comp_quan[q][8]=uh[q][8];     // J
      comp_quan[q][9]=uh[q][9];
      comp_quan[q][10]=uh[q][10];
      comp_quan[q][11]=uh[q][11];   // eta
    }
  }

} // end of namespace mhd