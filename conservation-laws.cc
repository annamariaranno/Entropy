
/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */


#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/sparse_ilu.h>
#include <fstream>
#include <iostream>



  using namespace dealii;
  template <int dim>
  class InitialValue : public Function<dim>
  {
  public:
	 InitialValue(const unsigned int n_components=1,
			const double time=0.) : Function<dim>(n_components,time){}
    virtual double value (const Point<dim>  &p,
                          const unsigned int component = 0) const;
  };


  template <int dim>
  double InitialValue<dim>::value (const Point<dim> &p,
                                     const unsigned int component) const
  {
	  double t=this->get_time();
	  Assert(component == 0, ExcInternalError());

	  if(dim==1)
	  {

		      if(std::abs(2*p[0] - 0.3) - 0.25 < std::pow(10,-8))
		        	return std::exp(-300.0*(std::pow(2*p[0] - 0.3,2)));
		      else if(std::abs(2*p[0] - 0.9) - 0.2 < std::pow(10,-8))
		      	return 1;
		      else if(std::abs(2*p[0] - 1.6) - 0.2 < std::pow(10,-8))
		      	return std::pow(1-(std::pow((2*p[0] - 1.6)/0.2,2)),0.5);
		      else
		      	return 0;
	  }
	  else
	  {
		  double a = 0.3;
		  		double r0 = 0.4;
		  		double r = std::sqrt(std::pow(p[0],2.)+std::pow(p[1],2.));
		  		double angle = std::atan2(p[1],p[0]);
		  		//Point<dim> pp(r*std::cos(angle-2*numbers::PI*t),r*std::sin(angle-2*numbers::PI*t));
		  		double x =r*std::cos(angle-2*numbers::PI*t);// pp[0];
		  		double y = r*std::sin(angle-2*numbers::PI*t);//pp[1];
		  		double res = 0.5*(1-std::tanh((std::pow(x-r0,2.)+std::pow(y,2.))/std::pow(a,2.)-1));
		  		//std::cout << "Return value"<< res << std::endl;
		  		//return res;
		  		return (x*(1-x)+y*(1-y)); //working for small time_step
		  		//return 1-(pp[0]*pp[0] + pp[1]*pp[1]);
		}

}

template <int dim>
  class BoundaryValues:  public Function<dim>
  {
  public:
    BoundaryValues () {};
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double> &values,
                             const unsigned int component=0) const;
  };

  template <int dim>
  void BoundaryValues<dim>::value_list(const std::vector<Point<dim> > &points,
                                       std::vector<double> &values,
                                       const unsigned int) const
  {
    Assert(values.size()==points.size(),
    ExcDimensionMismatch(values.size(),points.size()));


    for (unsigned int i=0; i<values.size(); ++i)
      {
        //if (points[i](0)<0.5)
          //values[i]=1.;
        //else
          values[i]=0.;
      }
  }



 template<int dim>
  class HyperbolicEquation
  {
  public:
    HyperbolicEquation();
    ~HyperbolicEquation();
    void run();
    //void cycle_refine();

  private:
    void setup_system();
    void assemble_transport_matrix ();
    void solve_time_step();
    void output_results(const unsigned int cycle) const;
    void refine_mesh (const unsigned int min_grid_level,
                      const unsigned int max_grid_level);
    //void refine_grid ();
    void process_solution (const unsigned int cycle);
    void compute_total_error();
    void write_table();



    SphericalManifold<dim> manifold;
    Triangulation<dim>     triangulation;
    FE_Q<dim>              fe;

    const MappingQGeneric<dim>   mapping;
    DoFHandler<dim>              dof_handler;

    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> transport_matrix;
    SparseMatrix<double> system_matrix;

    //Vector<double>       exact_solution;
    Vector<double>       solution;
    Vector<double>       old_solution;
    Vector<double>       right_hand_side; //transport_rhs
    Vector<double>       system_rhs;

    typedef MeshWorker::DoFInfo<dim> DoFInfo;
    typedef MeshWorker::IntegrationInfo<dim> CellInfo;

    static void integrate_cell_term (DoFInfo &dinfo,
                                         CellInfo &info);
    static void integrate_boundary_term (DoFInfo &dinfo,
                                             CellInfo &info);
    static void integrate_face_term (DoFInfo &dinfo1,
                                         DoFInfo &dinfo2,
                                         CellInfo &info1,
                                         CellInfo &info2);

    double               time;
    double               time_step;
    double				 t_end;
    unsigned int         timestep_number;
    unsigned int		 cycle;
    unsigned int		 cycles_number;

    const unsigned int initial_global_refinement;
    std::vector<double> L2_errors, L1_errors;
    const double         theta;
    ConvergenceTable     convergence_table;
  };




  /*template<int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide ()
      :
      Function<dim>(),
      period (0.2)
    {}
    virtual double value (const Point<dim> &p,
                          const unsigned int component = 0) const;
  private:
    const double period;
  };
  template<int dim>
  double RightHandSide<dim>::value (const Point<dim> &p,
                                    const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());
    Assert (dim == 2, ExcNotImplemented());
    const double time = this->get_time();
    const double point_within_period = (time/period - std::floor(time/period));
    if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
      {
        if ((p[0] > 0.5) && (p[1] > -0.5))
          return 1;
        else
          return 0;
      }
    else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
      {
        if ((p[0] > -0.5) && (p[1] > 0.5))
          return 1;
        else
          return 0;
      }
    else
      return 0;
  }*/



 // template<int dim>
  //class BoundaryValues : public Function<dim>
  //{
  //public:
    //virtual double value (const Point<dim>  &p,
      //                    const unsigned int component = 0) const;
  //};



  //template<int dim>
  //double BoundaryValues<dim>::value (const Point<dim> &/*p*/,
                                     //const unsigned int component) const
  //{
    //Assert(component == 0, ExcInternalError());
    //return 0;
  //}



  template<int dim>
  HyperbolicEquation<dim>::HyperbolicEquation ()
    :
    fe(1),
	mapping (fe.degree),
	dof_handler(triangulation),
	time(0.),
	time_step(1. / 10000),
	t_end(10000.*time_step),
	timestep_number(0),
	cycle(0),
	cycles_number(2),
	initial_global_refinement(3),
	theta(.5)
  {}

  template<int dim>
  HyperbolicEquation<dim>::~HyperbolicEquation ()
  {
	  dof_handler.clear();
  }



  template<int dim>
  void HyperbolicEquation<dim>::setup_system()
  {
	  dof_handler.distribute_dofs(fe);

	  std::cout << std::endl
	                << "==========================================="
	                << std::endl
	                << "Cycle:    " << cycle << std::endl
	                << "Number of active cells: " << triangulation.n_active_cells()
	                << std::endl
	                << "Number of degrees of freedom: " << dof_handler.n_dofs()
	                << std::endl
	                << std::endl;

	  constraints.clear ();
	  DoFTools::make_hanging_node_constraints (dof_handler,
	                                                  constraints);
	  constraints.close();

	  DynamicSparsityPattern dsp(dof_handler.n_dofs());
	  DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    transport_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(mapping,
    								  dof_handler,
                                      QGauss<dim>(fe.degree+1),
                                      mass_matrix);
    //MatrixCreator::create_laplace_matrix(dof_handler,
                                    //     QGauss<dim>(fe.degree+1),
                                      //   transport_matrix);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  template <int dim>
    void HyperbolicEquation<dim>::assemble_transport_matrix ()
    {
      MeshWorker::IntegrationInfoBox<dim> info_box;

      const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
      info_box.initialize_gauss_quadrature(n_gauss_points,
                                           n_gauss_points,
                                           n_gauss_points);

      info_box.initialize_update_flags();
      UpdateFlags update_flags = update_quadrature_points |
                                 update_values            |
                                 update_gradients;
      info_box.add_update_flags(update_flags, true, true, true, true);

      info_box.initialize(fe, mapping);

      MeshWorker::DoFInfo<dim> dof_info(dof_handler);

      MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double> >
      assembler;
      assembler.initialize(transport_matrix, right_hand_side);

      MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>,
	  MeshWorker::IntegrationInfoBox<dim> >
      (dof_handler.begin_active(), dof_handler.end(),
       dof_info, info_box,
       &HyperbolicEquation<dim>::integrate_cell_term,
       &HyperbolicEquation<dim>::integrate_boundary_term,
       &HyperbolicEquation<dim>::integrate_face_term,
       assembler);
    }



    template <int dim>
    void HyperbolicEquation<dim>::integrate_cell_term (DoFInfo &dinfo,
                                                     CellInfo &info)
    {
      const FEValuesBase<dim> &fe_v = info.fe_values();
      FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
      const std::vector<double> &JxW = fe_v.get_JxW_values ();

      for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
        {
          Point<dim> beta;
          beta(0) = -fe_v.quadrature_point(point)(1);
          beta(1) = fe_v.quadrature_point(point)(0);
          beta *= 2*numbers::PI;

          for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
              local_matrix(i,j) += beta*fe_v.shape_grad(j,point)*
                                   fe_v.shape_value(i,point) *
                                   JxW[point];
        }
    }

    template <int dim>
    void HyperbolicEquation<dim>::integrate_boundary_term (DoFInfo &dinfo,
                                                         CellInfo &info)
    {
      const FEValuesBase<dim> &fe_v = info.fe_values();
      FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
      Vector<double> &local_vector = dinfo.vector(0).block(0);

      const std::vector<double> &JxW = fe_v.get_JxW_values ();
      const std::vector<Tensor<1,dim> > &normals = fe_v.get_all_normal_vectors ();

      std::vector<double> g(fe_v.n_quadrature_points);

      ZeroFunction<dim> boundary_function;
      boundary_function.value_list (fe_v.get_quadrature_points(), g);

      for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
        {
          Point<dim> beta;
          beta(0) = -fe_v.quadrature_point(point)(1);
          beta(1) = fe_v.quadrature_point(point)(0);
          beta *= 2*numbers::PI;

          const double beta_n=beta * normals[point];
          if (beta_n<0)
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            {
            	local_vector(i) -= beta_n *
            	                   g[point] *
            	                   fe_v.shape_value(i,point) *
            	                   JxW[point];

              for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                local_matrix(i,j) -= beta_n *
                                     fe_v.shape_value(j,point) *
                                     fe_v.shape_value(i,point) *
                                     JxW[point];
            }
        }
    }

    template <int dim>
    void HyperbolicEquation<dim>::integrate_face_term (DoFInfo &dinfo1,
                                                     DoFInfo &dinfo2,
                                                     CellInfo &info1,
                                                     CellInfo &info2)
    {
      const FEValuesBase<dim> &fe_v = info1.fe_values();

      const FEValuesBase<dim> &fe_v_neighbor = info2.fe_values();

      FullMatrix<double> &u1_v1_matrix = dinfo1.matrix(0,false).matrix;
      FullMatrix<double> &u2_v1_matrix = dinfo1.matrix(0,true).matrix;
      FullMatrix<double> &u1_v2_matrix = dinfo2.matrix(0,true).matrix;
      FullMatrix<double> &u2_v2_matrix = dinfo2.matrix(0,false).matrix;


      const std::vector<double> &JxW = fe_v.get_JxW_values ();
      const std::vector<Tensor<1,dim> > &normals = fe_v.get_all_normal_vectors ();

      for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
        {
          Point<dim> beta;
          beta(0) = -fe_v.quadrature_point(point)(1);
          beta(1) = fe_v.quadrature_point(point)(0);
          beta *= 2*numbers::PI;

          const double beta_n=beta * normals[point];
          if (beta_n>0)
            {
              for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                  u1_v1_matrix(i,j) += 0;//beta_n *
                                       //fe_v.shape_value(j,point) *
                                       //fe_v.shape_value(i,point) *
                                       //JxW[point];

              for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
                for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                  u1_v2_matrix(k,j) -= 0;//beta_n *
                                       //fe_v.shape_value(j,point) *
                                       //fe_v_neighbor.shape_value(k,point) *
                                       //JxW[point];
            }
          else
            {
              for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
                  u2_v1_matrix(i,l) += 0;//beta_n *
                                       //fe_v_neighbor.shape_value(l,point) *
                                       //fe_v.shape_value(i,point) *
                                       //JxW[point];

              for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
                for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
                  u2_v2_matrix(k,l) -= 0;//beta_n *
                                       //fe_v_neighbor.shape_value(l,point) *
                                       //fe_v_neighbor.shape_value(k,point) *
                                       //JxW[point];
            }
        }
    }



  template<int dim>
  void HyperbolicEquation<dim>::solve_time_step()
  {

	  //SolverControl           solver_control (1000, 1e-12);
	    //  SolverRichardson<>      solver (solver_control);

	      //PreconditionBlockSSOR<SparseMatrix<double> > preconditioner;

	      //preconditioner.initialize(system_matrix, fe.dofs_per_cell);

	      //solver.solve (system_matrix, solution, system_rhs,
	                    //preconditioner);
	  SparseILU<double> preconditioner;
	  	preconditioner.initialize(system_matrix,
	  			SparseILU<double>::AdditionalData());
    SolverControl solver_control(1000, 1e-8);// * system_rhs.l2_norm());
    SolverGMRES<> gmres(solver_control);

/*    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);*/

    gmres.solve(system_matrix, solution, system_rhs,
             preconditioner);
    		//PreconditionIdentity());

    constraints.distribute(solution);
    if(timestep_number%1000==0)
        std::cout << "     " << solver_control.last_step()
            << " GMRES iterations." << std::endl;
  }



  template<int dim>
  void HyperbolicEquation<dim>::output_results(const unsigned int cycle) const
  {
	  if(timestep_number%50==0)
	    {
		  DataOut<dim> data_out;

		      data_out.attach_dof_handler(dof_handler);
		      data_out.add_data_vector(solution, "U");

		      data_out.build_patches();

		      const std::string filename = "test_solution-"  + Utilities::int_to_string(cycle, 1) + "-"
		                                   + Utilities::int_to_string(timestep_number, 3)
		                                   + ".vtk";
		      std::ofstream output(filename.c_str());
		      data_out.write_vtk(output);
	  }

  }


  template <int dim>
  void HyperbolicEquation<dim>::refine_mesh (const unsigned int min_grid_level,
                                       const unsigned int max_grid_level)
  {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(fe.degree+2),
                                        typename FunctionMap<dim>::type(),
                                        solution,
                                        estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
                                                       estimated_error_per_cell,
                                                       0.6, 0.4);

    if (triangulation.n_levels() > max_grid_level)
      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active(max_grid_level);
           cell != triangulation.end(); ++cell)
        cell->clear_refine_flag ();
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active(min_grid_level);
         cell != triangulation.end_active(min_grid_level); ++cell)
      cell->clear_coarsen_flag ();

    SolutionTransfer<dim> solution_trans(dof_handler);

    Vector<double> previous_solution;
    previous_solution = solution;
    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    triangulation.execute_coarsening_and_refinement ();
    setup_system ();
    assemble_transport_matrix();

    solution_trans.interpolate(previous_solution, solution);
    constraints.distribute (solution);
  }



 template <int dim>
void HyperbolicEquation<dim>::process_solution (const unsigned int cycle)
{
Vector<float> difference_per_cell (triangulation.n_active_cells());
VectorTools::integrate_difference (mapping,
                                  dof_handler,
                                 solution,
                                 InitialValue<dim>(1, time),
                                 difference_per_cell,
                                 QGauss<dim>(fe.degree+2),
                                 VectorTools::L1_norm);
//const double L1_error = difference_per_cell.l1_norm();
L1_errors.push_back(difference_per_cell.l1_norm());

difference_per_cell.reinit(triangulation.n_active_cells());
VectorTools::integrate_difference (mapping,
                                  dof_handler,
                                 solution,
                                 InitialValue<dim>(1, time),
                                 difference_per_cell,
                                 QGauss<dim>(fe.degree+2),
                                 VectorTools::L2_norm);
//const double L2_error = difference_per_cell.l2_norm();
L2_errors.push_back(difference_per_cell.l2_norm());
if(timestep_number%1000==0)
	{
		std::cout << "L2 error:   "<< difference_per_cell.l2_norm()<<std::endl;
	}
//const unsigned int n_active_cells=triangulation.n_active_cells();
//const unsigned int n_dofs=dof_handler.n_dofs();
//std::cout << "Cycle " << cycle << ':'
//        << std::endl
//        << "   Number of active cells:       "
//        << n_active_cells
//        << std::endl
//        << "   Number of degrees of freedom: "
//        << n_dofs
//        << std::endl;
}

 template <int dim>
 void HyperbolicEquation<dim>::compute_total_error()
 {
 	double L1_sum, L2_sum;
 	L1_sum=0;
 	L2_sum=0;
 	for(unsigned int j=0; j<L1_errors.size()-1; j++)
 	{
 		L1_sum+=(L1_errors[j]+L1_errors[j+1]);
 		L2_sum+=(L2_errors[j]+L2_errors[j+1]);
 	}
 	L1_sum*=(time_step/2.);
 	L2_sum*=(time_step/2.);

 	//total_L1_error.push_back(L1_sum);
 	//total_L2_error.push_back(L2_sum);
 	//std::cout<< "L1, L2 errors vector size:   "<<L1_errors.size()<< ",   "
 	//	<<L2_errors.size()<<std::endl;
 	L1_errors.clear();
 	L2_errors.clear();

 	convergence_table.add_value("cycle", cycle);
 	convergence_table.add_value("cells", triangulation.n_active_cells());
 	convergence_table.add_value("dofs", dof_handler.n_dofs());
 	convergence_table.add_value("L1", L1_sum);
 	convergence_table.add_value("L2", L2_sum);
 }

template <int dim>
void HyperbolicEquation<dim>::write_table()
{
    convergence_table.set_precision("L1", 3);
    convergence_table.set_precision("L2", 3);
    convergence_table.set_scientific("L1", true);
    convergence_table.set_scientific("L2", true);
    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("L1", "@f$L^1@f$-error");
    convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");
    std::cout << std::endl;
    convergence_table.write_text(std::cout);

   std::string error_filename = "test_error.tex";
   std::ofstream error_table_file(error_filename.c_str());
   convergence_table.write_tex(error_table_file);

	convergence_table
	.evaluate_convergence_rates("L1", ConvergenceTable::reduction_rate);
	convergence_table
	.evaluate_convergence_rates("L1", ConvergenceTable::reduction_rate_log2);
	convergence_table
	.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
	convergence_table
	.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
	std::cout << std::endl;
	convergence_table.write_text(std::cout);
	std::string conv_filename = "test_convergence.tex";
	std::ofstream table_file(conv_filename.c_str());
	convergence_table.write_tex(table_file);
}


template<int dim>
  void HyperbolicEquation<dim>::run()
  {
	  for(; cycle<=cycles_number; cycle++)
	  {
		  if(cycle==0)
	      {
			  if(dim==1)
				  GridGenerator::hyper_cube (triangulation);
			  else
	      	  {
				  GridGenerator::hyper_ball (triangulation);
	          	  triangulation.set_all_manifold_ids_on_boundary(0);
	          	  triangulation.set_manifold (0, manifold);
	          	  //GridGenerator::hyper_cube (triangulation);
	           }

	      triangulation.refine_global (initial_global_refinement);
	      }

    setup_system();
    assemble_transport_matrix();

    //unsigned int pre_refinement_step = 0;

    Vector<double> tmp;
    //Vector<double> forcing_terms;

    //start_time_iteration:


    //forcing_terms.reinit (solution.size());

    timestep_number = 0;
    time            = 0;

    VectorTools::project(mapping,
            dof_handler,
            constraints,
			//ConstraintMatrix(),
            QGauss<dim>(fe.degree+1),
            InitialValue<dim>(1, time),
            old_solution);

    solution = old_solution;

    process_solution(cycle);
    output_results(cycle);

    system_matrix.copy_from(mass_matrix);
    system_matrix.add(theta * time_step, transport_matrix);

    while (time < t_end)
      {
        time += time_step;
        timestep_number++;

        // std::cout << "Time step " << timestep_number << " at t=" << time
          //        << std::endl;

        mass_matrix.vmult(system_rhs, old_solution);
		tmp.reinit (solution.size());
        transport_matrix.vmult(tmp, old_solution);
        system_rhs.add(-(1 - theta) * time_step, tmp);

        //RIGHT HAND SIDE NEEDS TO BE ADDED WHEN g!=0 (Inflow BC)

        //RightHandSide<dim> rhs_function;
        //rhs_function.set_time(time);
        //VectorTools::create_right_hand_side(dof_handler,
          //                                  QGauss<dim>(fe.degree+1),
            //                                rhs_function,
              //                              tmp);
        //forcing_terms = tmp;
        //forcing_terms *= time_step * theta;

        //rhs_function.set_time(time - time_step);
        //VectorTools::create_right_hand_side(dof_handler,
          //                                  QGauss<dim>(fe.degree+1),
            //                                rhs_function,
              //                              tmp);

        //forcing_terms.add(time_step * (1 - theta), tmp);

        system_rhs += right_hand_side; //transport_rhs is time independent

        constraints.condense (system_matrix, system_rhs);

        /*{
          //?
          BoundaryValues<dim> boundary_values_function;
          boundary_values_function.set_time(time);
          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_function,
                                                   boundary_values);
          MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             solution,
                                             system_rhs);
        }*/

        solve_time_step();

        if(timestep_number%1000==0)
        {
        	std::cout<<"Timestep  "<<timestep_number<< "at time:   "<<time<<std::endl;
        	output_results(cycle);

        }

        /*if ((timestep_number == 1) &&
            (pre_refinement_step < n_adaptive_pre_refinement_steps))
          {
            refine_mesh (initial_global_refinement,
                         initial_global_refinement + n_adaptive_pre_refinement_steps);
            ++pre_refinement_step;
            tmp.reinit (solution.size());
            //forcing_terms.reinit (solution.size());
            std::cout << std::endl;
          goto start_time_iteration;
          }*/
/*        else if ((timestep_number > 0) && (timestep_number % 10 == 0))
          {
            refine_mesh (initial_global_refinement,
                         initial_global_refinement + n_adaptive_pre_refinement_steps);
            tmp.reinit (solution.size());
            //
            //forcing_terms.reinit (solution.size());
          }*/
        process_solution(cycle);
        old_solution = solution;
      }


      triangulation.refine_global();
      compute_total_error();

   }
	  write_table();
  }


/*template <int dim>
void HyperbolicEquation<dim>::cycle_refine()
{
	triangulation.refine_global();
}*/

int main()
{
  try
    {
      using namespace dealii;

      HyperbolicEquation<2> hyperbolic_equation;
      //hyperbolic_equation.set_fe_degree(1);
      hyperbolic_equation.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
