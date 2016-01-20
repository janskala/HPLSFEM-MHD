#include <fstream>

#include "problem.h"

namespace mhd
{
  
  template <int dim>
  void MHDProblem<dim>::output_results(const unsigned int fileNum)
  {
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

    DataOut<dim,hp::DoFHandler<dim> > data_out;
    data_out.attach_triangulation(triangulation);

    std::vector<std::string> solution_names;

    solution_names.push_back("Rho");
    solution_names.push_back("u_x"); // x
    solution_names.push_back("u_y"); // y
    solution_names.push_back("u_z"); // z
    solution_names.push_back("B_x"); // x
    solution_names.push_back("B_y"); // y
    solution_names.push_back("B_z"); // z
    solution_names.push_back("p");
    solution_names.push_back("J_x"); // x
    solution_names.push_back("J_y"); // y
    solution_names.push_back("J_z"); // z

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(Nv,DataComponentInterpretation::component_is_scalar);
    // works only for 3D
    /*data_component_interpretation  // density
      .push_back(DataComponentInterpretation::component_is_scalar);
    data_component_interpretation  // velocity x
      .push_back(DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation  // velocity y
      .push_back(DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation  // velocity z
      .push_back(DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation  // magnetic field x
      .push_back(DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation  // magnetic field y
      .push_back(DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation  // magnetic field z
      .push_back(DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation  // pressure
      .push_back(DataComponentInterpretation::component_is_scalar);
    data_component_interpretation  // current density x
      .push_back(DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation  // current density y
      .push_back(DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation  // current density z
      .push_back(DataComponentInterpretation::component_is_part_of_vector);*/
    
    
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
    
    data_out.build_patches();
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