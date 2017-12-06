/*
 * conservation-laws.cc
 *
 *  Created on: Dec 5, 2017
 *      Author: annamaria
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <map>

#include <deal.II/base/time_stepping.h>
#include <deal.II/lac/identity_matrix.h>


using namespace dealii;

class InitialValue : public Function<1>
  {
  public:
    virtual double value (const Point<1>  &p,
                          const unsigned int component = 0) const;
  };



  double InitialValue::value (const Point<1> &p,
                                     const unsigned int component) const
  {
    Assert(component == 0, ExcInternalError());
    if(std::abs(2*p[0] - 0.3) - 0.25 < std::pow(10,-8))
      	return std::exp(-300.0*(std::pow(2*p[0] - 0.3,2)));
    else if(std::abs(2*p[0] - 0.9) - 0.2 < std::pow(10,-8))
    	return 1;
    else if(std::abs(2*p[0] - 1.6) - 0.2 < std::pow(10,-8))
    	return std::pow(1-(std::pow((2*p[0] - 1.6)/0.2,2)),0.5);
    else
    	return 0;
  }

class CL
{
public:
  CL();
  void run();
private:
  void setup_system();
  void assemble_system();
  double get_source (const double time,
                     const Point<1> &point) const;
  Vector<double> evaluate_CL (const double time,
                                     const Vector<double> &y) const;

  void output_results (const unsigned int time_step,
                      TimeStepping::runge_kutta_method method) const;

  void set_initial_value();
  void explicit_method (const TimeStepping::runge_kutta_method method,
                        const unsigned int                     n_time_steps,
                        const double                           initial_time,
                        const double                           final_time);

  double                       absorption_cross_section;
  Triangulation<1>             triangulation;
  FE_Q<1>                      fe;
  DoFHandler<1>                dof_handler;
  ConstraintMatrix             constraint_matrix;
  SparsityPattern              sparsity_pattern;
  SparseMatrix<double>         system_matrix;
  SparseMatrix<double>         mass_matrix;
  SparseDirectUMFPACK          inverse_mass_matrix;
  Vector<double>               solution;
  Vector<double>               old_solution;
  double 					   b[4];
  double 					   c[4];
};

  CL::CL()
    :
    absorption_cross_section(1.),
    fe(2),
    dof_handler(triangulation),
	b({1/6,1/3,1/3,1/6}),
	c({0,1/2,1/2,1})
  {}


  void CL::setup_system ()
  {
    dof_handler.distribute_dofs(fe);
    VectorTools::interpolate_boundary_values(dof_handler,0,ZeroFunction<1>(),constraint_matrix);
    constraint_matrix.close();
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,dsp,constraint_matrix);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
  }


  void CL::assemble_system ()
  {
    system_matrix = 0.;
    mass_matrix = 0.;
    const QGauss<1> quadrature_formula(3);
    FEValues<1> fe_values(fe, quadrature_formula,
                          update_values | update_gradients | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_matrix (dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    DoFHandler<1>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_matrix = 0.;
        cell_mass_matrix = 0.;
        fe_values.reinit (cell);
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              {
                cell_matrix(i,j) += ((- absorption_cross_section *
                                      fe_values.shape_value(i,q_point) *
                                      fe_values.shape_value(j,q_point)) *
                                     fe_values.JxW(q_point));
                cell_mass_matrix(i,j) += fe_values.shape_value(i,q_point) *
                                         fe_values.shape_value(j,q_point) *
                                         fe_values.JxW(q_point);
              }
        cell->get_dof_indices(local_dof_indices);
        constraint_matrix.distribute_local_to_global(cell_matrix,local_dof_indices,system_matrix);
        constraint_matrix.distribute_local_to_global(cell_mass_matrix,local_dof_indices,mass_matrix);
      }
    inverse_mass_matrix.initialize(mass_matrix);
  }

  double CL::get_source (const double time,
                                const Point<1> &point) const
  {
    return 0.0;
  }

  Vector<double> CL::evaluate_CL (const double time,
                                                const Vector<double> &y) const
  {
    Vector<double> tmp(dof_handler.n_dofs());
    tmp = 0.;
    system_matrix.vmult(tmp,y);
    const QGauss<1> quadrature_formula(3);
    FEValues<1> fe_values(fe,
                          quadrature_formula,
                          update_values | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
    Vector<double>  cell_source(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    DoFHandler<1>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_source = 0.;
        fe_values.reinit (cell);
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
            const double source = get_source(time,
                                             fe_values.quadrature_point(q_point));
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              cell_source(i) += source *
                                fe_values.shape_value(i,q_point) *
                                fe_values.JxW(q_point);
          }
        cell->get_dof_indices(local_dof_indices);
        constraint_matrix.distribute_local_to_global(cell_source,
                                                     local_dof_indices,
                                                     tmp);
      }
    Vector<double> value(dof_handler.n_dofs());
    inverse_mass_matrix.vmult(value,tmp);
    return value;
  }

  void CL::output_results (const unsigned int time_step,
                                  TimeStepping::runge_kutta_method method) const
  {
    std::string method_name;

    method_name = "rk4";

    DataOut<1> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    const std::string filename = "solution-" + method_name + "-" +
                                 Utilities::int_to_string (time_step, 3) +
                                 ".vtu";
    std::ofstream output(filename.c_str());
    data_out.write_vtu(output);
  }

  void CL::set_initial_value()
  {
	  VectorTools::interpolate(dof_handler,
	                               InitialValue(),
	                               old_solution);

  }

  void CL::explicit_method (  const TimeStepping::runge_kutta_method method,
                                   const unsigned int                     n_time_steps,
                                   const double                           initial_time,
                                   const double                           final_time)
  {
    const double time_step = (final_time-initial_time)/static_cast<double> (n_time_steps);
    double time = initial_time;
    solution = 0.;
    Vector<double> kj;
    output_results(0,method);
    for (unsigned int i=0; i<n_time_steps; ++i)
      {
    	solution+=old_solution;
    	for (unsigned int j=1; j<=4; j++)
    		{
    			kj=this->evaluate_CL(time+i*time_step + c[j]*time_step,old_solution);
    			kj*=time_step*b[j];
    			solution+=kj;//.equ(time_step*b[j],kj);//time_step*b[j]*kj;
    		}
        if ((i+1)%10==0)
          output_results(i+1,method);
        old_solution = solution;
      }
  }

  void CL::run ()
  {
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(4);
    //Triangulation<1>::active_cell_iterator
    //cell = triangulation.begin_active(),
    //endc = triangulation.end();
    //for (; cell!=endc; ++cell)
      //for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
        //if (cell->face(f)->at_boundary())
          //{
            //if ((cell->face(f)->center()[0]==0.) || (cell->face(f)->center()[0]==5.))
              //cell->face(f)->set_boundary_id(1);
            //else
              //cell->face(f)->set_boundary_id(0);
          //}

    setup_system();
    assemble_system();
    set_initial_value();
    unsigned int       n_steps      = 0;
    const unsigned int n_time_steps = 100;
    const double       initial_time = 0.;
    const double       final_time   = 100;

    explicit_method (TimeStepping::RK_CLASSIC_FOURTH_ORDER,
                     n_time_steps,
                     initial_time,
                     final_time);
    std::cout << "Fourth order Runge-Kutta: error=" << solution.l2_norm() << std::endl;
    std::cout << std::endl;
  }


int main ()
{
  try
    {
      CL cons_law;
      cons_law.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };
  return 0;
}

