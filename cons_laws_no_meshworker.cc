/*
 * space_age_struct_model.cc
 *
 *  Created on: Feb 23, 2018
 *      Author: annamaria
 */
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
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
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <deal.II/base/tensor_function.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/base/convergence_table.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>


 using namespace dealii;
 
 template <int dim>
class AdvectionField : public TensorFunction<1,dim>
{
public:
  AdvectionField () : TensorFunction<1,dim> () {}
  virtual Tensor<1,dim> value (const Point<dim> &p) const;
  //virtual void value_list (const std::vector<Point<dim> > &points,
    //                       std::vector<Tensor<1,dim> >    &values) const;
                           
  //DeclException2 (ExcDimensionMismatch,
    //            unsigned int, unsigned int,
      //          << "The vector has size " << arg1 << " but should have "
        //        << arg2 << " elements.");
};

template <int dim>
Tensor<1,dim> AdvectionField<dim>::value (const Point<dim> &p) const
{
  Point<dim> value;
  value[0] = -2*numbers::PI*p[1];
  value[1] = 2*numbers::PI*p[0];
  return value;
}
/*template <int dim>
void
AdvectionField<dim>::value_list (const std::vector<Point<dim> > &points,
                                 std::vector<Tensor<1,dim> >    &values) const
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  for (unsigned int i=0; i<points.size(); ++i)
    values[i] = AdvectionField<dim>::value (points[i]);
}*/

  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
	ExactSolution(const unsigned int n_components=1,
			const double time=0.):Function<dim>(n_components,time){}

    virtual double value (const Point<dim>  &p,
                          const unsigned int component = 0) const;
  };


  template <int dim>
  double ExactSolution<dim>::value (const Point<dim> &p,
                                     const unsigned int component) const
  {
	  Assert(component == 0, ExcInternalError());

	  double t=this->get_time();

	  if(dim==1)
	  {
		  	  double xx= p[0]-t;
		      if(std::abs(2*xx - 0.3) - 0.25 < std::pow(10,-8))
		        	return std::exp(-300.0*(std::pow(2*xx - 0.3,2)));
		      else if(std::abs(2*xx - 0.9) - 0.2 < std::pow(10,-8))
		      	return 1;
		      else if(std::abs(2*xx - 1.6) - 0.2 < std::pow(10,-8))
		      	return std::pow(1-(std::pow((2*xx - 1.6)/0.2,2)),0.5);
		      else
		      	return 0;
	  }
	  else
	  {
		  double a = 0.3;
		  double r0 = 0.4;
		  double r = std::sqrt(std::pow(p[0],2)+std::pow(p[1],2));
		  double angle = std::atan2(p[1],p[0]);
		  double xx = r*std::cos(angle-2*numbers::PI*t);
		  double yy = r*std::sin(angle-2*numbers::PI*t);
		  double res = 0.5*(1-std::tanh((std::pow(xx-r0,2)+std::pow(yy,2))/std::pow(a,2)-1));
		  //std::cout << "Return value"<< res << std::endl;
		  return 0.5*(1-std::tanh((std::pow(xx-r0,2.)+std::pow(yy,2.))/std::pow(a,2.)-1));
          //return (p[0]*(1-p[0])+p[1]*(1-p[1])); //working for small time_step
          //return 1-(p[0]*p[0] + p[1]*p[1]);
	  }

}

template <int dim>
  class InitialValue: public Function<dim>
{
public:
	InitialValue(const unsigned int n_components=1,
			const double time=0.)
    :
    	Function<dim>(n_components,time)
	{}

	virtual double value (const Point<dim>  &p,
	                          const unsigned int component = 0) const;
};

template<int dim>
double InitialValue<dim>::value(const Point<dim> &p,
		const unsigned int component) const
		{
			   return ExactSolution<dim>(1,this->get_time()).value(p,component);
		}


template <int dim>
  class BoundaryValues:  public Function<dim>
  {
  public:
    BoundaryValues (const unsigned int n_components=1,
			const double time=0.):Function<dim>(n_components,time){}
			
    virtual double value (const Point<dim> &p,
                             const unsigned int component=0) const;
  };

  template<int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
		const unsigned int component) const
		{
			   return ExactSolution<dim>(1,this->get_time()).value(p,component);
		}

template <int dim>
  class ForcingTerm:  public Function<dim>
  {
  public:
    ForcingTerm (const unsigned int n_components=1,
			const double time=0.):Function<dim>(n_components,time){}
			
    virtual double value (const Point<dim> &p,
                             const unsigned int component=0) const;
  };

  template<int dim>
double ForcingTerm<dim>::value(const Point<dim> &p,
		const unsigned int component) const
		{
			   return 0.;
		}



  template<int dim>
  class HyperbolicEquation
  {
  public:
    HyperbolicEquation();
    ~HyperbolicEquation();
    void run();
    //void set_fe_degree(int degree);


  private:
    void setup_system();
    void local_assemble_matrices();
    void local_assemble_rhs();
    void assemble_system(); //NB: possible only if beta is time-independent
    void assemble_rhs();
    void solve_time_step();
    void output_results(const unsigned int cycle) const;   
    void process_solution (const unsigned int cycle);
    void write_table();

    SphericalManifold<dim> manifold;
    Triangulation<dim>     triangulation;
    //const int			   fe_degree;
    FE_Q<dim>              fe;
    
    const MappingQGeneric<dim>   mapping;
    DoFHandler<dim>              dof_handler;
    
    ConstraintMatrix     constraints;

    FEValues<dim>     fe_values;
    FEFaceValues<dim> fe_face_values;
    
    const unsigned int dofs_per_cell;  // = fe.dofs_per_cell;
	const unsigned int n_q_points;     // = fe_values.get_quadrature().size();
	const unsigned int n_face_q_points; //= fe_face_values.get_quadrature().size();
	
	typename DoFHandler<dim>::active_cell_iterator cell;
	typename DoFHandler<dim>::active_cell_iterator endc;

	FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> transport_matrix;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> system_rhs_matrix;

    //ExactSolution<dim>       exact_solution;
    const AdvectionField<dim> advection_field;
	ForcingTerm<dim>  forcing_term;
	BoundaryValues<dim> boundary_values;
    Vector<double>       solution;
    Vector<double>       old_solution;
    Vector<double>		 time_dependent_rhs;
    Vector<double>       system_rhs;

    double               time;
    double               time_step;
    double				 t_end;
    unsigned int         timestep_number;
    unsigned int		 cycle;
    unsigned int		 cycles_number;

    const unsigned int initial_global_refinement;

    const double         theta;
    ConvergenceTable     convergence_table;
  };
  
template<int dim>
  HyperbolicEquation<dim>::HyperbolicEquation ()
    :
    fe(1),
    mapping (fe.degree),
    dof_handler(triangulation),
    fe_values (mapping, fe,
             QGauss<dim>(fe.degree+2),
             update_values   | update_gradients |
             update_quadrature_points | update_JxW_values),
  	fe_face_values (mapping, fe,
                  QGauss<dim-1>(fe.degree+2),
                  update_values     | update_quadrature_points   |
                  update_JxW_values | update_normal_vectors),
	dofs_per_cell (fe.dofs_per_cell),
	n_q_points (fe_values.get_quadrature().size()),
	n_face_q_points (fe_face_values.get_quadrature().size()),
	//cell (dof_handler.begin_active()),
	//endc (dof_handler.end()),
    time(0.),
    time_step(1. / 10000),
	t_end(10.*time_step),
    timestep_number(0),
    cycle(0),
	cycles_number(0),
	initial_global_refinement(1),
    theta(.5)
  {
	cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
	cell_rhs.reinit (dofs_per_cell);
	local_dof_indices.resize(dofs_per_cell);
  }

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
    system_rhs_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(mapping,
    								dof_handler,
                                      QGauss<dim>(fe.degree+1),
                                      mass_matrix);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
	time_dependent_rhs.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

template <int dim>
    void HyperbolicEquation<dim>::local_assemble_matrices()
    {
    	//assembling B=(beta*grad(phi_j),phi_i)
    	for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      		for (unsigned int i=0; i<dofs_per_cell; ++i)
        	{
          		for (unsigned int j=0; j<dofs_per_cell; ++j)
            		cell_matrix(i,j) += advection_field.value(fe_values.quadrature_point(q_point)) *
                                          fe_values.shape_grad(j,q_point)   *
                                          fe_values.shape_value(i,q_point) *
                                          fe_values.JxW(q_point);
        	}
        	
        //assembling M_in= -(beta*n)(phi_j,phi_i) on inflow boundary
    	for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      		if (cell->face(face)->at_boundary())
        	{
          		fe_face_values.reinit (cell, face);
    
          		for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
            		if (fe_face_values.normal_vector(q_point) *
                			advection_field.value(fe_values.quadrature_point(q_point)) < 0)
              			for (unsigned int i=0; i<dofs_per_cell; ++i)
                		{
                  			for (unsigned int j=0; j<dofs_per_cell; ++j)
                    		cell_matrix(i,j) -= (advection_field.value(fe_values.quadrature_point(q_point)) *
                                                   fe_face_values.normal_vector(q_point) *
                                                   fe_face_values.shape_value(i,q_point) *
                                                   fe_face_values.shape_value(j,q_point) *
                                                   fe_face_values.JxW(q_point));
                		}
                		
                		
        	}
    }
    
template <int dim>
    void HyperbolicEquation<dim>::local_assemble_rhs()
    {	    	
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        	{
        		for (unsigned int i=0; i<dofs_per_cell; ++i)
        		{
        			cell_rhs(i) += fe_values.shape_value(i,q_point)*
                                forcing_term.value(fe_values.quadrature_point(q_point)) *
                                fe_values.JxW (q_point);
        			/*std::cout<< "Forcing term:  "<<
        					forcing_term.value(fe_values.quadrature_point(q_point))
        					<<std::endl;*/
        		}
        	}

        	
        	for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      		if (cell->face(face)->at_boundary())
        	{
          		fe_face_values.reinit (cell, face);
          		
          		for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
            		if (fe_face_values.normal_vector(q_point) *
                			advection_field.value(fe_values.quadrature_point(q_point)) < 0)
            		{
              			for (unsigned int i=0; i<dofs_per_cell; ++i)
                		{
                  			cell_rhs(i) -= (advection_field.value(fe_values.quadrature_point(q_point)) *
                                            fe_face_values.normal_vector(q_point) *
                                            boundary_values.value(fe_values.quadrature_point(q_point))         *
                                            fe_face_values.shape_value(i,q_point) *
                                            fe_face_values.JxW(q_point));
                		}

              			std::cout<< "normal vector:  "<<
              			           fe_face_values.normal_vector(q_point)[0]
              			           <<std::endl;
              			std::cout<< "normal vector:  "<<
              					fe_face_values.normal_vector(q_point)[1]
																	  <<std::endl;

            		}
            }
        	
        //cell_rhs.print(); this is 0 because the contributes of the BV vanish
        						//due to the normal

    }

    
template <int dim>
    void HyperbolicEquation<dim>::assemble_system()
    {
		//typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active();
		endc = dof_handler.end();
		for (; cell!=endc; ++cell)
  		{
    		fe_values.reinit (cell);
    		cell_matrix = 0;
    		
    		local_assemble_matrices();
    		cell->get_dof_indices (local_dof_indices);
    		for (unsigned int i=0; i<local_dof_indices.size(); ++i)
    		{
      			for (unsigned int j=0; j<local_dof_indices.size(); ++j)
        			transport_matrix.add (local_dof_indices[i],
                           local_dof_indices[j],
                           cell_matrix(i,j));
			}
    	}

    	system_matrix.copy_from(mass_matrix);
    	system_matrix.add(theta*time_step, transport_matrix);
    	
    	system_rhs_matrix.copy_from(mass_matrix);
    	system_rhs_matrix.add(-(1-theta)*time_step, transport_matrix);
    	//Make sure old_solution is already initialized
    	system_rhs_matrix.vmult(system_rhs, old_solution);
    }
    
template <int dim>
    void HyperbolicEquation<dim>::assemble_rhs()
    {
		cell = dof_handler.begin_active();
		endc = dof_handler.end();
		for (; cell!=endc; ++cell)
  		{
    		fe_values.reinit (cell);
    		cell_rhs = 0;
			//assemble rhs for F and G at time n
			forcing_term.set_time(time);
			boundary_values.set_time(time);
			//std::cout<< "Time set:  "<<boundary_values.get_time()<<std::endl;
    		local_assemble_rhs();
    		cell->get_dof_indices (local_dof_indices);
    		for (unsigned int i=0; i<local_dof_indices.size(); ++i)
    		{
    			time_dependent_rhs(local_dof_indices[i]) += time_step*theta*cell_rhs(i);
    		}
    		
    		cell_rhs = 0;
    		//assemble rhs for F and G at time n-1
    		forcing_term.set_time(time-time_step);
			boundary_values.set_time(time-time_step);
			//std::cout<< "Time set:  "<<boundary_values.get_time()<<std::endl;
    		local_assemble_rhs();
    		for (unsigned int i=0; i<local_dof_indices.size(); ++i)
    		{
    			time_dependent_rhs(local_dof_indices[i]) += time_step*(1-theta)*cell_rhs(i);
    		}
    	}
    	
    }
    
template<int dim>
  void HyperbolicEquation<dim>::solve_time_step()
  {

    SolverControl solver_control(1000, 1e-10 * system_rhs.l2_norm());
    SolverGMRES<> gmres(solver_control);

    gmres.solve(system_matrix, solution, system_rhs,
             PreconditionIdentity());

    //constraints.distribute(solution);
    if(timestep_number%10==0)
        std::cout << "     " << solver_control.last_step()
            << " GMRES iterations." << std::endl;
  }
  
template<int dim>
  void HyperbolicEquation<dim>::output_results(const unsigned int cycle) const
  {
	  //if(timestep_number%10==0)
	    //{
		  DataOut<dim> data_out;

		      data_out.attach_dof_handler(dof_handler);
		      data_out.add_data_vector(solution, "U");

		      data_out.build_patches();

		      const std::string filename = "solution-"  + Utilities::int_to_string(cycle, 1) + "-"
		                                   + Utilities::int_to_string(timestep_number, 3)
		                                   + ".vtk";
		      std::ofstream output(filename.c_str());
		      data_out.write_vtk(output);
	  //}

  }
  
  template <int dim>
void HyperbolicEquation<dim>::process_solution (const unsigned int cycle)
{
Vector<float> difference_per_cell (triangulation.n_active_cells());
VectorTools::integrate_difference (mapping,
                                  dof_handler,
                                 solution,
                                 ExactSolution<dim>(1, time),
                                 difference_per_cell,
                                 QGauss<dim>(fe.degree+2),
                                 VectorTools::L1_norm);
const double L1_error = difference_per_cell.l1_norm();
VectorTools::integrate_difference (mapping,
                                  dof_handler,
                                 solution,
                                 ExactSolution<dim>(1,time),
                                 difference_per_cell,
                                 QGauss<dim>(fe.degree+2),
                                 VectorTools::L2_norm);
const double L2_error = difference_per_cell.l2_norm();

const unsigned int n_active_cells=triangulation.n_active_cells();
const unsigned int n_dofs=dof_handler.n_dofs();
/*std::cout << "Cycle " << cycle << ':'
        << std::endl
        << "   Number of active cells:       "
        << n_active_cells
        << std::endl
        << "   Number of degrees of freedom: "
        << n_dofs
        << std::endl;*/
convergence_table.add_value("cycle", cycle);
convergence_table.add_value("cells", n_active_cells);
convergence_table.add_value("dofs", n_dofs);
convergence_table.add_value("L1", L1_error);
convergence_table.add_value("L2", L2_error);

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

     std::string error_filename = "error.tex";
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
  	std::string conv_filename = "convergence.tex";
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
    			/*GridGenerator::hyper_ball (triangulation);
        		triangulation.set_all_manifold_ids_on_boundary(0);
        		triangulation.set_manifold (0, manifold);*/
    			GridGenerator::hyper_cube (triangulation);
    		}

    		triangulation.refine_global (initial_global_refinement);
    	}

    	setup_system();

      	VectorTools::project(mapping,
            dof_handler,
            constraints,
			//ConstraintMatrix(),
            QGauss<dim>(fe.degree+1),
            InitialValue<dim>(),
            old_solution);
    	solution = old_solution;
    	//old_solution.print();
    	output_results(cycle);
            
    	assemble_system();
    	//transport_matrix.print(std::cout);

    	timestep_number = 0;
    	time            = 0;

    	while (time < t_end)
      	{
        	time += time_step;
        	++timestep_number;
        	//if(timestep_number%10==0)
            	std::cout << "Time step " << timestep_number << " at t=" << time
                  	<< std::endl;

         	assemble_rhs();
         	system_rhs.add(1.,time_dependent_rhs);
         	//time_dependent_rhs.print();
         	time_dependent_rhs.reinit(dof_handler.n_dofs());

        	solve_time_step();
        	//solution.print();

        	output_results(cycle);

        	old_solution = solution;
      	}

      process_solution(cycle);
      
      triangulation.refine_global();
    }

    write_table();
  }

 /* template<int dim>
   	void HyperbolicEquation<dim>::set_fe_degree(int degree)
   {
	  fe_degree=degree;
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
