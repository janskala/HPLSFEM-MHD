#include <fstream>

#include "problem.h"

namespace mhd
{
  
  template <int dim>
  void MHDProblem<dim>::output_results(const unsigned int fileNum)
  {
    const char names[3][Nv][4]={ {"Rho","u","u_y","u_z","B","B_y","B_z","p","J","J_y","J_z"},
                           {"Rho","u","u","u_z","B","B","B_z","p","J","J","J_z"},
                           {"Rho","u","u","u","B","B","B","p","J","J","J"}};
    DataComponentInterpretation::DataComponentInterpretation cmpIntpr[Nv];
    std::fill_n(cmpIntpr, Nv, DataComponentInterpretation::component_is_scalar);
    for(unsigned int i=0;i<dim;i++)
      cmpIntpr[1+i]=cmpIntpr[4+i]=cmpIntpr[8+i]=
            DataComponentInterpretation::component_is_part_of_vector;
    
#ifdef USE_TIMER
    TimerOutput::Scope t(computing_timer, "output");
#endif
    const std::string filename = ("results/o" +
                                Utilities::int_to_string (fileNum, 5) +
                                "." +
                                Utilities::int_to_string
                                (triangulation.locally_owned_subdomain(), 4) +
                                ".vtu");
    std::ofstream output(filename);

    DataOut<dim> data_out;
    data_out.attach_triangulation(triangulation);

    std::vector<std::string> solution_names;
    std::vector<DataComponentInterpretation::DataComponentInterpretation> 
                             data_component_interpretation;

    for(unsigned int i=0;i<Nv;i++){
      solution_names.push_back(names[dim-1][i]);
      data_component_interpretation.push_back(cmpIntpr[i]);
    }
    
    // TODO: implement DataPostprocessor for derived quantities
    LA::MPI::Vector &v=solution;
    LA::MPI::Vector &cds=distributed_solution;
    std::pair<types::global_dof_index, types::global_dof_index> range=v.local_range();
    for(unsigned int i=range.first;i<range.second;i+=Nv){
      double iRh=1.0/v[i+0];
      cds[i+0]=v[i+0];  // Rho
      cds[i+1]=v[i+1]*iRh;  // v
      cds[i+2]=v[i+2]*iRh;
      cds[i+3]=v[i+3]*iRh;
      cds[i+4]=v[i+4];  // B
      cds[i+5]=v[i+5];
      cds[i+6]=v[i+6];
      cds[i+7]=((5./3.)-1.)*(v[i+7]-(v[i+1]*v[i+1]+v[i+2]*v[i+2]+v[i+3]*v[i+3])*iRh  // p
                    -(v[i+4]*v[i+4]+v[i+5]*v[i+5]+v[i+6]*v[i+6]));
      cds[i+8]=v[i+8];
      cds[i+9]=v[i+9];
      cds[i+10]=v[i+10];
    }
    cds.compress(VectorOperation::insert);  // write changes in parallel vector
    residue=cds;

    data_out.add_data_vector(dof_handler,residue, solution_names,
                         //   DataOut<dim>::type_dof_data,
                            data_component_interpretation);
    
    Vector<float> subdomain(triangulation.n_active_cells());
    for(unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.add_data_vector(shockIndicator, "refine");
    
    data_out.add_data_vector(dof_handler_s,eta, "eta");
    
    data_out.build_patches(FEO+1);
    data_out.write_vtu(output);
    
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0){
      std::vector<std::string> filenames;
      for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
        filenames.push_back (std::string("o") +
                             Utilities::int_to_string (fileNum, 5) +
                             "." +
                             Utilities::int_to_string(i, 4) +
                             ".vtu");
      const std::string
      pvtu_master_filename = ("results/f" +
                              Utilities::int_to_string (fileNum, 5) +
                              ".pvtu");
      std::ofstream pvtu_master (pvtu_master_filename.c_str());
      data_out.write_pvtu_record (pvtu_master, filenames);
      const std::string
      visit_master_filename = ("results/f" +
                               Utilities::int_to_string (fileNum, 5) +
                               ".visit");
      std::ofstream visit_master (visit_master_filename.c_str());
      data_out.write_visit_record (visit_master, filenames);
    }
  }
  
} // end of namespace mhd