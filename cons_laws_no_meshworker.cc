/*
 * entropy_viscosity.cc
 *
 *  Created on: May 3, 2018
 *      Author: annamaria
 */
#include <deal.II/base/quadrature.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
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
#include<iomanip>

using namespace dealii;


template <int dim>
class AdvectionField : public TensorFunction<1,dim>
{
	public:
	AdvectionField () : TensorFunction<1,dim> () {}
	virtual Tensor<1,dim> value (const Point<dim> &p) const;
};

template <int dim>
Tensor<1,dim> AdvectionField<dim>::value (const Point<dim> &p) const
{
  Point<dim> value;
  value[0] = 1.;//-2*numbers::PI*p[1];
  value[1] = 1.;//2*numbers::PI*p[0];
  return value;
}

/*template <int dim>
class InitialValue: public Function<dim>
{
	public:
	InitialValue(const unsigned int n_components=1,
			const double time=0.) : Function<dim>(n_components,time){}
	virtual double value (const Point<dim>  &p,
			const unsigned int component = 0) const;
};

template<int dim>
double InitialValue<dim>::value(const Point<dim> &p,
		const unsigned int component) const
{
	Assert(component == 0, ExcInternalError());
	//double t=this->get_time();

	if(dim==1)
	{
		double x= p[0];
		if(std::fabs(2*x - 0.3) - 0.25 < 1e-8)
			return std::exp(-300.0*(std::pow(2*x - 0.3,2.)));
		else if(std::fabs(2*x - 0.9) - 0.2 < 1e-8)
			return 1;
		else if(std::fabs(2*x - 1.6) - 0.2 < 1e-8)
			return std::pow(1-(std::pow((2*x - 1.6)/0.2,2.)),0.5);
		else
			return 0;
	}
	else
	{
		double a = 0.3;
		double r0 = 0.4;
		double x = p[0];
		double y = p[1];
		double res = 0.5*(1-std::tanh((std::pow(x-r0,2.)+std::pow(y,2.))/std::pow(a,2.)-1));
		//std::cout << "Return value"<< res << std::endl;
		//return res;
		//return (p[0]*(1-p[0])+p[1]*(1-p[1])); //working for small time_step
		return 1-(p[0]*p[0] + p[1]*p[1]);
	}
}*/


template <int dim>
class ExactSolution : public Function<dim>
{
	public:
	ExactSolution(const unsigned int n_components=1,
			const double time=0.) : Function<dim>(n_components,time){}
    double value (const Point<dim>  &p,
    		const unsigned int component = 0) const;
};


template <int dim>
double ExactSolution<dim>::value (const Point<dim> &p,
		const unsigned int component) const
{
	double t=this->get_time();

	if(dim==1)
	{
		double x= p[0];
		if(std::fabs(2*x - 0.3) - 0.25 < 1e-8)
			return std::exp(-300.0*(std::pow(2*x - 0.3,2.)));
		else if(std::fabs(2*x - 0.9) - 0.2 < 1e-8)
			return 1;
		else if(std::fabs(2*x - 1.6) - 0.2 < 1e-8)
			return std::pow(1-(std::pow((2*x - 1.6)/0.2,2.)),0.5);
		else
			return 0;
	}

	else
	{
		//NON SMOOTH ROTATING
		/*double res=0;
		double a = 0.3;
		double r0 = 0.4;
		double r = std::sqrt(std::pow(p[0],2.)+std::pow(p[1],2.));
		double angle = std::atan2(p[1],p[0]);
		double xx=r*std::cos(angle-2*numbers::PI*t),
				yy=r*std::sin(angle-2*numbers::PI*t);
		if(std::sqrt(xx*xx+yy*yy)<a)
		{
			res=std::sqrt((xx-r0)*(xx-r0)+yy*yy);
		}
		return res;*/

		//SMOOTH ROTATING
		/*double a = 0.3;
		double r0 = 0.4;
		double r = std::sqrt(std::pow(p[0],2.)+std::pow(p[1],2.));
		double angle = std::atan2(p[1],p[0]);
		//Point<dim> pp(r*std::cos(angle-2*numbers::PI*t),r*std::sin(angle-2*numbers::PI*t));
		double x =r*std::cos(angle-2*numbers::PI*t);// pp[0];
		double y = r*std::sin(angle-2*numbers::PI*t);//pp[1];
		double res = 0.5*(1-std::tanh((std::pow(x-r0,2.)+std::pow(y,2.))/std::pow(a,2.)-1));
		//std::cout << "Return value"<< res << std::endl;
		return res;*/
		//return (x*(1-x)+y*(1-y));
		//return 1-(pp[0]*pp[0] + pp[1]*pp[1]);

		//BURGERS
		double res=0;
		double x=p[0], y=p[1];

		if(t<1e-12)
		{
			if(x<0.5 && y>0.5)
				res=-0.2;
			else if(x>0.5 && y>0.5)
				res=-1;
			else if(x<0.5 && y<0.5)
				res=0.5;
			else if(x>0.5 && y<0.5)
				res=0.8;
		}

		else
		{
			if(x<0.5-(3.*t/5.))
			{
				if(y>0.5+(3.*t/20.))
					res=-0.2;
				else
					res=0.5;
			}
			else if(x>0.5-(3.*t/5.)&& x<0.5-t/4.)
			{
				if(y>(-8.*x/7.+15./14.-15.*t/28.))
					res=-1.;
				else
					res=0.5;
			}
			else if(x>0.5-t/4. && x<0.5-t/2.)
			{
				if(y>(x/6.+5./12.-5.*t/24.))
					res=-1.;
				else
					res=0.5;
			}
			else if(x>0.5-t/2. && x<0.5+(4.*t/5.))
			{
				if(y>x-(5./(18.*t))*(x+t-0.5)*(x+t-0.5))
					res=-1.;
				else
					res=(2.*x-1.)/(2.*t);
			}
			else if(x>0.5+(4.*t/5.))
			{
				if(y>0.5-(t/10.))
					res=-1.;
				else
					res=0.8;
			}
		}

		return res;
	}
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
	return 0.;
}

template <int dim>
class ForcingTerm:  public Function<dim>
{
	public:
    ForcingTerm (const unsigned int n_components=1,
    		const double time=0.) : Function<dim>(n_components,time){}
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

	private:
	void generate_grid();
	void setup_system();
	void set_initial_distribution();
	void local_assemble_matrices();
	void local_assemble_rhs();
	void assemble_system(); //NB: possible only if beta is time-independent
	void assemble_rhs();
	void assemble_nl_rhs(Vector<double> current_solution);
	void solve_time_step();
	void assemble_viscosity_matrix();
	void output_results(Vector<double> solution) const;
	void process_solution ();
	void compute_residual();
	void compute_total_error();
	void write_table();

	SphericalManifold<dim> manifold;
	//SphericalManifold<dim> exact_sol_manifold;
	Triangulation<dim>     triangulation;
	//Triangulation<dim>     exact_sol_triangulation;
	FE_Q<dim>              fe;

	const MappingQGeneric<dim>   mapping;
	DoFHandler<dim>              dof_handler;
	//DoFHandler<dim>              exact_sol_dof_handler;

	ConstraintMatrix     constraints;

	/*FEValues<dim>     fe_values(mapping, fe,
   		 QGauss<dim>(fe.degree+2),
			 update_values   | update_gradients |
			 update_quadrature_points | update_JxW_values);*/
	/*FEFaceValues<dim> fe_face_values(mapping, fe,
			 QGauss<dim-1>(fe.degree+2),
			 update_values     | update_quadrature_points   |
			 update_JxW_values | update_normal_vectors);*/

	const unsigned int dofs_per_cell,dofs_per_face;  // = fe.dofs_per_cell, = fe.dofs_per_face;
 	//const unsigned int n_q_points;     // = fe_values.get_quadrature().size();
 	//const unsigned int n_face_q_points; //= fe_face_values.get_quadrature().size();

 	//typename DoFHandler<dim>::active_cell_iterator cell;
 	//typename DoFHandler<dim>::active_cell_iterator endc;

 	FullMatrix<double>                   cell_matrix;
 	FullMatrix<double>                   face_matrix;
 	Vector<double>                       cell_rhs;
 	Vector<double>                       face_rhs;
 	std::vector<types::global_dof_index> local_dof_indices;
 	std::vector<types::global_dof_index> local_face_dof_indices;
 	SparsityPattern      sparsity_pattern;
 	SparseMatrix<double> mass_matrix;
 	SparseMatrix<double> transport_matrix;
 	SparseMatrix<double> stab_matrix;
 	SparseMatrix<double> viscosity_matrix;
 	SparseMatrix<double> system_matrix;
 	SparseMatrix<double> system_rhs_matrix;


 	const AdvectionField<dim> advection_field;
 	ForcingTerm<dim>  forcing_term;
 	ExactSolution<dim> /*BoundaryValues<dim>*/ boundary_values;
 	//Vector<double>       exact_solution;
 	std::map<types::global_dof_index,double> inflow_boundary;
 	std::vector<double> nu_vector;

 	Vector<double>       initial_solution, second_solution, third_solution, solution;
 	Vector<double>       old_solution, older_solution, oldest_solution;
 	Vector<double>		 ent_solution;
 	Vector<double>		 time_dependent_rhs;
 	Vector<double>       system_rhs;

 	//Vector<double>		 ent_old_solution, ent_solution, fp_solution;

 	double               time, t_end;
 	unsigned int		 output_times, n_times;
 	double               time_step;


 	unsigned int         timestep_number;
 	unsigned int		 cycle;
 	unsigned int		 cycles_number;

 	const unsigned int initial_global_refinement;
 	double				h_x;
 	const double         theta;
 	const double		 solver_tol;
 	const double		 lambda;
 	std::vector<double> L2_errors, L1_errors;
 	//std::vector<double> total_L2_error, total_L1_error;
 	std::vector<double> residual_vector;
 	ConvergenceTable     convergence_table;
   };

 template<int dim>
 HyperbolicEquation<dim>::HyperbolicEquation ()
 	 :
 	 fe(2),
     mapping (fe.degree),
     dof_handler(triangulation),
	 //exact_sol_dof_handler(exact_sol_triangulation),
     //fe_values (mapping, fe,
    //		 QGauss<dim>(fe.degree+2),
	//		 update_values   | update_gradients |
		//	 update_quadrature_points | update_JxW_values),
	 //fe_face_values (mapping, fe,
		//	 QGauss<dim-1>(fe.degree+2),
		//	 update_values     | update_quadrature_points   |
		//	 update_JxW_values | update_normal_vectors),
 	 dofs_per_cell (fe.dofs_per_cell),
	 dofs_per_face (fe.dofs_per_face),
	 //n_q_points (fe_values.get_quadrature().size()),
	 //n_face_q_points (fe_face_values.get_quadrature().size()),
     time(0.),
	 t_end(0.5),//1.0),
	 output_times(t_end*100000),
	 n_times(10*output_times),
	 time_step(t_end/n_times),
     timestep_number(0),
     cycle(0),
	 cycles_number(3),
	 initial_global_refinement(3),
	 h_x(1./std::pow(2.,initial_global_refinement)),
     theta(1.),
	 solver_tol(1e-8),
	 lambda(1.)
	 {
	 	 cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
	 	 face_matrix.reinit (dofs_per_face, dofs_per_face);
	 	 cell_rhs.reinit (dofs_per_cell);
	 	 face_rhs.reinit (dofs_per_face);
	 	 local_dof_indices.resize(dofs_per_cell);
	 	 local_face_dof_indices.resize(dofs_per_face);
	 	 std::cout<< "time_step:   "<<time_step<<std::endl;
         L2_errors.resize(n_times);
         L1_errors.resize(n_times);
         std::ofstream test_file("parameters.txt");
        test_file<<"final time:     "<< t_end<<std::endl;
     	test_file<<"time step:     "<<time_step<<std::endl;
     	test_file<<"theta:   "<< theta<<std::endl;
     	test_file<<": initial_global_refinement:  "<< initial_global_refinement<<std::endl;
     	test_file<<"solver_tol:   "<< solver_tol<<std::endl;
	 }

template<int dim>
HyperbolicEquation<dim>::~HyperbolicEquation ()
{
	dof_handler.clear();
}

template<int dim>
void HyperbolicEquation<dim>::generate_grid()
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

			/*GridGenerator::hyper_ball (exact_sol_triangulation);
			exact_sol_triangulation.set_all_manifold_ids_on_boundary(0);
			exact_sol_triangulation.set_manifold (0, exact_sol_manifold);*/
			GridGenerator::hyper_cube (triangulation);

			triangulation.refine_global (initial_global_refinement);

			std::cout<< "Mesh size at cycle 0:   "
					//<<GridTools::minimal_cell_diameter(triangulation)/std::sqrt(2.)
					<<h_x
					<<std::endl;
			std::cout<< "Mesh size at cycle "<< cycles_number<<":   "
								//<<GridTools::minimal_cell_diameter(triangulation)/std::pow(2.,cycles_number)/std::sqrt(2.)
					<<h_x/std::pow(2.,cycles_number)
					<<std::endl;
			//exit(0);


			//typename Triangulation<dim>::active_cell_iterator
			//cell = triangulation.begin_active(),
			//endc = triangulation.end();

			//unsigned int face_count=0;
			/*for (; cell!=endc; ++cell)
			{
				if(cell->at_boundary())
				{
					for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
					{
						if(cell->face(face_number)->at_boundary())
						{
							//fe_face_values.reinit (cell, face_number);
							//Point<dim> face_center(cell->face(face)->center());
							//for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
							//{
							double x=cell->face(face_number)->center()[0];
							double y=cell->face(face_number)->center()[1];
								if (((std::fabs(x-10)<1e-12) && (y>0))
									|| ((std::fabs(y-10)<1e-12) && (x<0))
									|| ((std::fabs(x+ 10)<1e-12) &&( y<0))
									|| ((std::fabs(y+10)<1e-12) &&( x>0)))
									//check with computed inflow boundary
									//fe_face_values.normal_vector(q_point) *
										//advection_field.value(cell->face(face)->center()) < 0)
								{
									face_count++;
									cell->face(face_number)->set_boundary_id(1);
									//std::cout<<"Inflow point:   ("<<x<<" , " <<y<<")"<<std::endl;
									//exit(0);
								}
							//}
						}


					}

				}

			}*/
			//exit(0);
		}


		//exact_sol_triangulation.refine_global (initial_global_refinement + cycles_number);
	}

	else
	{
		triangulation.refine_global();
		h_x/=2.;
	}
	//std::cout<< "Number of faces on inflow boundary is: "<<face_count<<std::endl;
}

template<int dim>
void HyperbolicEquation<dim>::setup_system()
{
	dof_handler.distribute_dofs(fe);
	//exact_sol_dof_handler.distribute_dofs(fe);

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

	constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler,
			dsp
			//constraints,
			/*keep_constrained_dofs =  true*/);
	sparsity_pattern.copy_from(dsp);

	mass_matrix.reinit(sparsity_pattern);
 	transport_matrix.reinit(sparsity_pattern);
 	stab_matrix.reinit(sparsity_pattern);
 	viscosity_matrix.reinit(sparsity_pattern);

 	system_matrix.reinit(sparsity_pattern);
 	system_rhs_matrix.reinit(sparsity_pattern);

 	MatrixCreator::create_mass_matrix(mapping,
 			dof_handler,
			QGauss<dim>(fe.degree+2),
			mass_matrix);

 	MatrixCreator::create_laplace_matrix(mapping,
		 			dof_handler,
					QGauss<dim>(fe.degree+2),
					stab_matrix);

 	//exact_solution.reinit(exact_sol_dof_handler.n_dofs());
    initial_solution.reinit(dof_handler.n_dofs());
    second_solution.reinit(dof_handler.n_dofs());
    third_solution.reinit(dof_handler.n_dofs());
 	solution.reinit(dof_handler.n_dofs());
 	old_solution.reinit(dof_handler.n_dofs());
 	older_solution.reinit(dof_handler.n_dofs());
 	oldest_solution.reinit(dof_handler.n_dofs());
 	time_dependent_rhs.reinit(dof_handler.n_dofs());
 	system_rhs.reinit(dof_handler.n_dofs());

 	ent_solution.reinit(dof_handler.n_dofs());
 	//ent_old_solution.reinit(dof_handler.n_dofs());
    //fp_solution.reinit(dof_handler.n_dofs());

 	VectorTools::interpolate_boundary_values (mapping,
 			dof_handler,
            1,
			//ConstantFunction<dim>(5),
            ZeroFunction<dim>(),
			//ExactSolution<dim>(1,time),
            inflow_boundary);
}

template <int dim>
void HyperbolicEquation<dim>::set_initial_distribution()
{
	VectorTools::project(mapping,
			dof_handler,
			constraints,
			QGauss<dim>(fe.degree+2),
			ExactSolution<dim>(1,time),
			initial_solution);

	oldest_solution = initial_solution;//for entropy viscosity computation
    //older_solution = initial_solution;
	//old_solution=initial_solution;
	//ent_old_solution=initial_solution;

	Vector<float> difference_per_cell (triangulation.n_active_cells());

	//use integrate difference

	VectorTools::integrate_difference (mapping,
			dof_handler,
			initial_solution,
			ExactSolution<dim>(1,time),
			difference_per_cell,
			QGauss<dim>(fe.degree+2), //to have less integration error
			VectorTools::L2_norm);

	//it's more an interpolation error as it is for time=0
	//L2_errors.push_back(difference_per_cell.l2_norm());
	std::cout<< "L2 initial error:"<< difference_per_cell.l2_norm()<<std::endl;
	L2_errors[0]=difference_per_cell.l2_norm();
    //std::cout<< "L2 initial error:"<< L2_errors[0]<<std::endl;

	VectorTools::integrate_difference (mapping,
			dof_handler,
			initial_solution,
			ExactSolution<dim>(1,time),
			difference_per_cell,
			QGauss<dim>(fe.degree+2), //to have less integration error
			VectorTools::L1_norm);

	L1_errors[0]=difference_per_cell.l1_norm();

	VectorTools::project(mapping,
			dof_handler,
			constraints,
			QGauss<dim>(fe.degree+2),
			ExactSolution<dim>(1,time+time_step),
			second_solution);

	//ent_solution=old_solution;
	older_solution=second_solution;
	//solution = old_solution; //for output_results();

	Vector<float> new_difference_per_cell (triangulation.n_active_cells());

	//use integrate difference

	VectorTools::integrate_difference (mapping,
			dof_handler,
			second_solution,
			ExactSolution<dim>(1,time+time_step),
			new_difference_per_cell,
			QGauss<dim>(fe.degree+2), //to have less integration error
			VectorTools::L2_norm);

	//it's more an interpolation error as it is for time=0
	//L2_errors.push_back(difference_per_cell.l2_norm());
	std::cout<< "L2 second error:"<< new_difference_per_cell.l2_norm()<<std::endl;
	L2_errors[1]=new_difference_per_cell.l2_norm();

	VectorTools::integrate_difference (mapping,
			dof_handler,
			second_solution,
			ExactSolution<dim>(1,time+time_step),
			new_difference_per_cell,
			QGauss<dim>(fe.degree+2), //to have less integration error
			VectorTools::L1_norm);

	L1_errors[1]=new_difference_per_cell.l1_norm();

	VectorTools::project(mapping,
			dof_handler,
			constraints,
			QGauss<dim>(fe.degree+2),
			ExactSolution<dim>(1,time+2.*time_step),
			third_solution);

	old_solution=third_solution;

	Vector<float> third_difference_per_cell (triangulation.n_active_cells());

	VectorTools::integrate_difference (mapping,
			dof_handler,
			third_solution,
			ExactSolution<dim>(1,time+time_step),
			third_difference_per_cell,
			QGauss<dim>(fe.degree+2), //to have less integration error
			VectorTools::L2_norm);

	std::cout<< "L2 third error:"<< third_difference_per_cell.l2_norm()<<std::endl;
	L2_errors[1]=third_difference_per_cell.l2_norm();

	VectorTools::integrate_difference (mapping,
			dof_handler,
			third_solution,
			ExactSolution<dim>(1,time+time_step),
			third_difference_per_cell,
			QGauss<dim>(fe.degree+2), //to have less integration error
			VectorTools::L1_norm);

	L1_errors[1]=third_difference_per_cell.l1_norm();

}

template <int dim>
void HyperbolicEquation<dim>::assemble_system()
{
	//int sent=0;
	typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	FEValues<dim>     fe_values(mapping, fe,
	   		 QGauss<dim>(fe.degree+2),
				 update_values   | update_gradients |
				 update_quadrature_points | update_JxW_values);
	FEFaceValues<dim> fe_face_values(mapping, fe,
						 QGauss<dim-1>(fe.degree+2),
						 update_values     | update_quadrature_points   |
						 update_JxW_values | update_normal_vectors);
	double face_count=0;
	for (; cell!=endc; ++cell)
	{
		fe_values.reinit (cell);
		cell_matrix = 0;

		//local_assemble_matrices();
		for (unsigned int q_point=0; q_point<fe_values.get_quadrature().size(); ++q_point)
		{
			for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
				for (unsigned int j=0; j<dofs_per_cell; ++j)
				{
					cell_matrix(i,j) += advection_field.value(fe_values.quadrature_point(q_point)) *
										fe_values.shape_grad(j,q_point)   *
										fe_values.shape_value(i,q_point) *
										fe_values.JxW(q_point);
				}
			}
		}

		//assembling M_in= -(beta*n)(phi_j,phi_i) on inflow boundary
		if (cell->at_boundary())
		{
			for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
			{
				if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id()==1)
				{
					fe_face_values.reinit (cell, face);
					face_matrix = 0;
					//sent=-1;
					face_count++;

					for (unsigned int q_point=0; q_point<fe_face_values.get_quadrature().size(); ++q_point)
					{
						//if (fe_face_values.normal_vector(q_point) *
						//advection_field.value(fe_face_values.quadrature_point(q_point)) < 0)
						//{

						for (unsigned int i=0; i<dofs_per_face; ++i)
						{
							for (unsigned int j=0; j<dofs_per_face; ++j)
							{
								face_matrix(i,j) += (advection_field.value(fe_face_values.quadrature_point(q_point)) *
										fe_face_values.normal_vector(q_point) *
										fe_face_values.shape_value(i,q_point) *
										fe_face_values.shape_value(j,q_point) *
										fe_face_values.JxW(q_point));
							}
						}
						//}
					}

					cell->face(face)->get_dof_indices (local_face_dof_indices);
					for (unsigned int i=0; i<local_face_dof_indices.size(); ++i)
					{
						for (unsigned int j=0; j<local_face_dof_indices.size(); ++j)
							transport_matrix.add (local_face_dof_indices[i],
									local_face_dof_indices[j],
									-face_matrix(i,j));
					}
				}
			}
		}

		cell->get_dof_indices (local_dof_indices);
		for (unsigned int i=0; i<local_dof_indices.size(); ++i)
		{
			for (unsigned int j=0; j<local_dof_indices.size(); ++j)
				transport_matrix.add (local_dof_indices[i],
						local_dof_indices[j],
						cell_matrix(i,j));
		}
	}

	std::cout<< "Number of faces on inflow boundary is: "<<face_count<<std::endl;

	system_matrix.copy_from(mass_matrix);
	system_matrix.add(theta*time_step, transport_matrix);

	system_rhs_matrix.copy_from(mass_matrix);
	system_rhs_matrix.add(-(1-theta)*time_step, transport_matrix);
}

template <int dim>
void HyperbolicEquation<dim>::assemble_rhs()
{
	double face_count=0;
	typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
	FEValues<dim>     fe_values(mapping, fe,
		   		 QGauss<dim>(fe.degree+2),
					 update_values   | update_gradients |
					 update_quadrature_points | update_JxW_values);
	FEFaceValues<dim> fe_face_values(mapping, fe,
							 QGauss<dim-1>(fe.degree+2),
							 update_values     | update_quadrature_points   |
							 update_JxW_values | update_normal_vectors);
	for (; cell!=endc; ++cell)
	{
		fe_values.reinit (cell);
		cell_rhs = 0;

		//assemble rhs for F and G at time n
		forcing_term.set_time(time);
		boundary_values.set_time(time);
		//local_assemble_rhs();
		for (unsigned int q_point=0; q_point<fe_values.get_quadrature().size(); ++q_point)
		{
			for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
				cell_rhs(i) += fe_values.shape_value(i,q_point)*
								forcing_term.value(fe_values.quadrature_point(q_point)) *
								fe_values.JxW (q_point);
			}
		}

		if (cell->at_boundary())
		{
			for (unsigned int facenumber=0; facenumber<GeometryInfo<dim>::faces_per_cell; ++facenumber)
			{
				if (cell->face(facenumber)->at_boundary() && cell->face(facenumber)->boundary_id()==1)
				{
					fe_face_values.reinit (cell, facenumber);
					face_rhs = 0;
					face_count++;

					for (unsigned int q_point=0; q_point<fe_face_values.get_quadrature().size(); ++q_point)
					{
						//if (fe_face_values.normal_vector(q_point) *
							//advection_field.value(fe_face_values.quadrature_point(q_point)) < 0)
						//{
						for (unsigned int i=0; i<dofs_per_face; ++i)
						{
							face_rhs(i) += (advection_field.value(fe_face_values.quadrature_point(q_point)) *
									fe_face_values.normal_vector(q_point) *
									boundary_values.value(fe_face_values.quadrature_point(q_point))*
									fe_face_values.shape_value(i,q_point) *
									fe_face_values.JxW(q_point));
						}
						//}
					}

					cell->face(facenumber)->get_dof_indices (local_face_dof_indices);
					for (unsigned int i=0; i<local_face_dof_indices.size(); ++i)
					{
						time_dependent_rhs(local_face_dof_indices[i]) -= time_step*theta*face_rhs(i);
					}
				}
			}
		}

		cell->get_dof_indices (local_dof_indices);
		for (unsigned int i=0; i<local_dof_indices.size(); ++i)
		{
			time_dependent_rhs(local_dof_indices[i]) += time_step*theta*cell_rhs(i);
		}

		cell_rhs = 0;
		//assemble rhs for F and G at time n-1
		forcing_term.set_time(time-time_step);
		boundary_values.set_time(time-time_step);

		//local_assemble_rhs();
		for (unsigned int q_point=0; q_point<fe_values.get_quadrature().size(); ++q_point)
		{
			for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
				cell_rhs(i) += fe_values.shape_value(i,q_point)*
								forcing_term.value(fe_values.quadrature_point(q_point)) *
								fe_values.JxW (q_point);
			}
		}

		if (cell->at_boundary())
		{
			for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
			{
				if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id()==1)
				{
					fe_face_values.reinit (cell, face);
					face_rhs = 0;
					for (unsigned int q_point=0; q_point<fe_face_values.get_quadrature().size(); ++q_point)
					{
						//if (fe_face_values.normal_vector(q_point) *
							//	advection_field.value(fe_face_values.quadrature_point(q_point)) < 0)
						//{
							for (unsigned int i=0; i<dofs_per_face; ++i)
							{
								face_rhs(i) += (advection_field.value(fe_face_values.quadrature_point(q_point)) *
												fe_face_values.normal_vector(q_point) *
												boundary_values.value(fe_face_values.quadrature_point(q_point))*
												fe_face_values.shape_value(i,q_point) *
												fe_face_values.JxW(q_point));
							}
						//}
					}

					cell->face(face)->get_dof_indices (local_face_dof_indices);
					for (unsigned int i=0; i<local_face_dof_indices.size(); ++i)
					{
						time_dependent_rhs(local_face_dof_indices[i]) -= time_step*(1-theta)*face_rhs(i);
					}
				}
			}
		}

		for (unsigned int i=0; i<local_dof_indices.size(); ++i)
		{
			time_dependent_rhs(local_dof_indices[i]) += time_step*(1-theta)*cell_rhs(i);
		}
	}

	/*if(timestep_number%output_times==0)
	{
		std::cout<< "Number of faces on inflow boundary is: "<<face_count<<std::endl;
	}*/

}

template <int dim>
void HyperbolicEquation<dim>::assemble_nl_rhs(Vector<double> current_solution)
{
	Vector<double> non_linear_term(current_solution);
	non_linear_term.scale(current_solution);
	non_linear_term/=2.;

	typename DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active(),
			endc = dof_handler.end();

	FEValues<dim>     fe_values(mapping, fe,
			   		 QGauss<dim>(fe.degree+2),
						 update_values   | update_gradients |
						 update_quadrature_points | update_JxW_values);
	//Vector<double> cell_rhs(dofs_per_cell);

	for (; cell!=endc; ++cell)
	{
		fe_values.reinit (cell);
		cell_rhs = 0;
		std::vector<Tensor<1,dim> > nl_gradient(fe_values.get_quadrature().size());
		fe_values.get_function_gradients(non_linear_term,nl_gradient);
		for (unsigned int q_point=0; q_point<fe_values.get_quadrature().size(); ++q_point)
		{
			for (unsigned int i=0; i<dofs_per_cell; i++)
			{
				cell_rhs(i) += fe_values.shape_value(i,q_point)*
						advection_field.value(fe_values.quadrature_point(q_point))*
						nl_gradient[q_point]*
						fe_values.JxW (q_point);
			}
		}

		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i=0; i<local_dof_indices.size(); i++)
		{
			time_dependent_rhs(local_dof_indices[i]) += time_step*cell_rhs(i);
		}

	}
}

template<int dim>
void HyperbolicEquation<dim>::solve_time_step()
{
	SparseILU<double> preconditioner;
	preconditioner.initialize(system_matrix,
			SparseILU<double>::AdditionalData());
    SolverControl solver_control(1000, solver_tol /** system_rhs.l2_norm()*/);
    //SolverGMRES<> gmres(solver_control);
    SolverCG<> cg(solver_control);
    cg.solve(system_matrix, solution, system_rhs,
    		preconditioner);
    		//PreconditionIdentity());

    //constraints.distribute (solution);

    if(timestep_number%output_times==0)
    std::cout << "CG iterations:     " << solver_control.last_step()
			<< std::endl;
}

template<int dim>
void HyperbolicEquation<dim>::assemble_viscosity_matrix()
{
	typename DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active(),
			endc = dof_handler.end();

	Quadrature<dim> q_dummy(fe.get_unit_support_points());
	FEValues<dim>     dummy_fe_values(mapping, fe, q_dummy,
				update_values   | update_gradients |
				update_quadrature_points);
	FEValues<dim>     fe_values(mapping, fe,
				QGauss<dim>(fe.degree+2),
				update_values   | update_gradients |
				update_quadrature_points | update_JxW_values);
	double c_e=1.;//0.5;
	double c_max=0.4/fe.degree;//0.1/fe.degree;

	//compute denominator: ||E-E_bar||

	Vector<double> entropy(ent_solution);//old_solution);
	entropy.scale(ent_solution);//old_solution);
	entropy*=(3./2.);///=2.;
	Vector<double> entropy_old(older_solution);
	entropy_old.scale(older_solution);
	entropy_old*=(4./2.);///=2.;
	Vector<double> entropy_oldest(oldest_solution);
	entropy_oldest.scale(oldest_solution);
	entropy_oldest/=2.;

	Vector<double> fp_entropy(ent_solution);
	fp_entropy.scale(ent_solution);
	fp_entropy.scale(ent_solution);
	fp_entropy/=3.;///2.;

	//compute averaged E
	const double area(numbers::PI);
	const double average_entropy(fp_entropy.l1_norm()/area);
	Vector<double> entropy_difference(fp_entropy);
	entropy_difference.add(-average_entropy);
	const double E_difference=entropy_difference.linfty_norm();
	double nu_max, nu_e, nu_h;
	double h_min, h_k, f_prime_max; //8*std::pow(numbers::PI,2.);

	for (; cell!=endc; ++cell)
	{
		dummy_fe_values.reinit(cell);
		fe_values.reinit(cell);
		h_min=2.;

		for(unsigned int i=0; i<3; i++)
		{
			h_k=cell->vertex(i).distance(cell->vertex(i+1));
			if(h_k<h_min)
				h_min=h_k;
		}

		f_prime_max=0.;
		//double div_F=0.;

		std::vector<double> entr_values(dummy_fe_values.get_quadrature().size()),
				old_entr_values(dummy_fe_values.get_quadrature().size()),//;
				oldest_entr_values(dummy_fe_values.get_quadrature().size());
		dummy_fe_values.get_function_values (entropy, entr_values);
		dummy_fe_values.get_function_values (entropy_old, old_entr_values);
		dummy_fe_values.get_function_values (entropy_oldest, oldest_entr_values);

		std::vector<double> entr_res(dummy_fe_values.get_quadrature().size());

		//std::vector<Tensor<1,dim> > entr_gradient(dummy_fe_values.get_quadrature().size());//, old_entr_gradient(fe_values.get_quadrature().size());
		//dummy_fe_values.get_function_gradients (fp_entropy, entr_gradient);
		std::vector<Tensor<1,dim> > F_gradient(dummy_fe_values.get_quadrature().size());//, old_entr_gradient(fe_values.get_quadrature().size());
		dummy_fe_values.get_function_gradients (fp_entropy, F_gradient);
		//fe_values.get_function_gradients (entropy_old, old_entr_gradient );
		//std::cout<<"Quadrature points:   "<<fe_values.n_quadrature_points<<std::endl;

		for (unsigned int q_point=0; q_point<dummy_fe_values.n_quadrature_points; ++q_point)
		{
			//std::cout<<"  "<< q_point<<std::endl;
			//h_k=fe_values.quadrature_point(q_point).distance(fe_values.quadrature_point(q_point+1));

			if(advection_field.value(dummy_fe_values.quadrature_point(q_point)).norm() > std::fabs(f_prime_max))
				f_prime_max =advection_field.value(dummy_fe_values.quadrature_point(q_point)).norm();

			entr_res[q_point]=(entr_values[q_point]- old_entr_values[q_point]//)
					 	 	 	 	 +oldest_entr_values[q_point])
									/(time_step*2.)
									+
						/*div_F = */ F_gradient[q_point]*advection_field.value(dummy_fe_values.quadrature_point(q_point));
									//entr_gradient[q_point]*advection_field.value(dummy_fe_values.quadrature_point(q_point));

		}

		std::vector<double>::iterator max_it=std::max_element(entr_res.begin(),entr_res.end());
		nu_max = c_max*h_min*f_prime_max;
		nu_e = c_e*h_min*h_min*(*max_it)/E_difference;

		nu_h=std::min(nu_max,nu_e);
		nu_vector.push_back(nu_h);

		//entropy viscosity contribution to matrices
		cell_matrix = 0;

		for (unsigned int q_point=0; q_point<fe_values.get_quadrature().size(); ++q_point)
		{
			for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
				for (unsigned int j=0; j<dofs_per_cell; ++j)
				{
					cell_matrix(i,j) += nu_h *
							fe_values.shape_grad(j,q_point)   *
							fe_values.shape_grad(i,q_point) *
							fe_values.JxW(q_point);
				}
			}
		}

		cell->get_dof_indices (local_dof_indices);
		for (unsigned int i=0; i<local_dof_indices.size(); ++i)
		{
			for (unsigned int j=0; j<local_dof_indices.size(); ++j)
				viscosity_matrix.add (local_dof_indices[i],
						local_dof_indices[j],
						cell_matrix(i,j));
		}
	}
	if(nu_vector.size()!=triangulation.n_active_cells())
	{
		std::cout<<"Non enough coefficients saved:  "<< nu_vector.size()
				<<" , "<<triangulation.n_active_cells() <<std::endl;
		exit(0);
	}
}

template<int dim>
void HyperbolicEquation<dim>::output_results(Vector<double> solution) const
{
		DataOut<dim> data_out;

		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(solution, "U");

		data_out.build_patches();

		const std::string filename = "visc_burgers_solution2-"  + Utilities::int_to_string(cycle, 1) + "-"
									+ Utilities::int_to_string(timestep_number, 3)
									+ ".vtk";
		std::ofstream output(filename.c_str());
		data_out.write_vtk(output);
}

template <int dim>
void HyperbolicEquation<dim>::process_solution ()
{
	Vector<float> difference_per_cell (/*exact_sol_*/triangulation.n_active_cells());

	//Vector<double> interpolated_solution(exact_sol_dof_handler.n_dofs());

	//interpolate solution onto exact_solution mesh
	/*VectorTools::interpolate_to_different_mesh(dof_handler, solution,
			exact_sol_dof_handler,constraints,interpolated_solution);*/

	/*interpolated_solution.add(-1, exact_solution);
	use integrate difference
	VectorTools::integrate_difference (mapping,
			exact_sol_dof_handler,
			interpolated_solution,
			ZeroFunction<dim>(),
			//ExactSolution<dim>(1,time),
			difference_per_cell,
			QGauss<dim>(fe.degree+2),
			VectorTools::L2_norm);*/

	VectorTools::integrate_difference (mapping,
				dof_handler,
				solution,
				ExactSolution<dim>(1,time),
				difference_per_cell,
				QGauss<dim>(fe.degree+2),
				VectorTools::L2_norm);

	//const double L2_error = difference_per_cell.l2_norm();
	//L2_errors.push_back(difference_per_cell.l2_norm());
	L2_errors[timestep_number]=difference_per_cell.l2_norm();

	if(timestep_number%output_times==0)
	{
		std::cout << "L2 error:   "<< difference_per_cell.l2_norm()<<std::endl;
	}

	VectorTools::integrate_difference (mapping,
			dof_handler,
			solution,
			ExactSolution<dim>(1, time),
			difference_per_cell,
			QGauss<dim>(fe.degree+2),
			VectorTools::L1_norm);

	//const double L1_error = difference_per_cell.l1_norm();
	L1_errors[timestep_number]=difference_per_cell.l1_norm();

	/*if(timestep_number==n_times)
	{

		const unsigned int n_active_cells=triangulation.n_active_cells();
		const unsigned int n_dofs=dof_handler.n_dofs();

		convergence_table.add_value("cycle", cycle);
		convergence_table.add_value("cells", n_active_cells);
		convergence_table.add_value("dofs", n_dofs);
		//only for test: L1 and L2 norms have to be integrated on time
		convergence_table.add_value("L1", L1_error);
		convergence_table.add_value("L2", L2_error);
	}*/

}

template <int dim>
void HyperbolicEquation<dim>::compute_residual()
{
	//NB: SIMPLIFIED RESIDUAL AS g=f=0 (no inflow boundary contributions)
	std::vector<double> cell_res(dofs_per_cell);
	std::vector<double> face_res(dofs_per_face);

	Vector<double> vector_res;
	vector_res.reinit(dof_handler.n_dofs());
	FEValues<dim>     fe_values(mapping, fe,
				 QGauss<dim>(fe.degree+2),
					 update_values   | update_gradients |
					 update_quadrature_points | update_JxW_values);
	FEFaceValues<dim> fe_face_values(mapping, fe,
							 QGauss<dim-1>(fe.degree+2),
							 update_values     | update_quadrature_points   |
							 update_JxW_values | update_normal_vectors);

	std::vector<double> sol_values(fe_values.get_quadrature().size()), old_sol_values(fe_values.get_quadrature().size());
	std::vector<double> face_sol_values(fe_face_values.get_quadrature().size()), face_old_sol_values(fe_face_values.get_quadrature().size());

	std::vector<Tensor<1,dim> > sol_gradient(fe_values.get_quadrature().size()), old_sol_gradient(fe_values.get_quadrature().size());

	typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(), endc = dof_handler.end();

	unsigned int it=0;

	for (; cell!=endc; ++cell)
	{
		//double nu_coeff=nu_vector[it];
		fe_values.reinit (cell);

		std::fill(cell_res.begin(),cell_res.end(), 0);

	    cell->get_dof_indices (local_dof_indices);

	    fe_values.get_function_values (solution, sol_values);
	    fe_values.get_function_values (old_solution, old_sol_values);

	    fe_values.get_function_gradients (solution, sol_gradient );
	    fe_values.get_function_gradients (old_solution, old_sol_gradient );

	    for (unsigned int q_point=0; q_point<fe_values.get_quadrature().size(); ++q_point)
	    {
	    	const double W = fe_values.JxW(q_point);

	    	for (unsigned int i=0; i<dofs_per_cell; ++i)
	    	{
	    		//rhs
	    		cell_res[i] += (sol_values[q_point]-old_sol_values[q_point])
	    				*fe_values.shape_value(i,q_point)*W;
	    		//lhs: time n
	    		cell_res[i] += time_step*theta*
	    				(advection_field.value(fe_values.quadrature_point(q_point))
	    				*sol_gradient[q_point]
						*fe_values.shape_value(i,q_point)*W
						//+
						//diffusion term
						//nu_coeff
						//*sol_gradient[q_point]
						//*fe_values.shape_grad(i,q_point)*W
						);
	    		//lhs: time n-1
	    		cell_res[i] += time_step*(1-theta)*
	    				(advection_field.value(fe_values.quadrature_point(q_point))
	    				*old_sol_gradient[q_point]
						*fe_values.shape_value(i,q_point)*W
						//+
						//diffusion term
						//nu_coeff
						//*sol_gradient[q_point]
						//*fe_values.shape_grad(i,q_point)*W
						);

	    		if (cell->at_boundary())
	    		{
	    			for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    			{
	    				if (cell->face(face)->at_boundary() && cell->face(face)->boundary_id()==1)
	    				{
	    					fe_face_values.reinit (cell, face);
	    					fe_face_values.get_function_values (solution, face_sol_values);
	    					fe_face_values.get_function_values (old_solution, face_old_sol_values);
	    					std::fill(face_res.begin(),face_res.end(), 0);

	    					for (unsigned int q_point=0; q_point<fe_face_values.get_quadrature().size(); ++q_point)
	    					{
	    						for (unsigned int i=0; i<dofs_per_face; ++i)
	    						{
	    							//face term : time n
	    							face_res[i] -= time_step*theta*
	    									(advection_field.value(fe_face_values.quadrature_point(q_point)) *
	    									fe_face_values.normal_vector(q_point) *
											face_sol_values[q_point]*
											fe_face_values.shape_value(i,q_point) *
											fe_face_values.JxW(q_point));

	    							//face term: time n-1
	    							face_res[i] -= time_step*(1-theta)*
	    									(advection_field.value(fe_face_values.quadrature_point(q_point)) *
	    									fe_face_values.normal_vector(q_point) *
											face_old_sol_values[q_point]*
											fe_face_values.shape_value(i,q_point) *
											fe_face_values.JxW(q_point));
	    						}
	    					}

	    					cell->face(face)->get_dof_indices (local_face_dof_indices);
	    					for (unsigned int i=0; i<dofs_per_face; ++i) //dofs_per_face= local_face_dof_indices.size()
	    					{
	    						vector_res(local_face_dof_indices[i]) += face_res[i];
	    					}
	    				}
	    			}
	    		}
	    	}
	    }

	    for (unsigned int i=0; i<dofs_per_cell; ++i) //dofs_per_face= local_face_dof_indices.size()
	    {
	    	vector_res(local_dof_indices[i]) += cell_res[i];
	    }

	    it++;
	}

	//NB: SIMPLIFIED RESIDUAL AS g=f=0 (no inflow boundary contributions)
	//apply inflow boundary values
	/*for (std::map<unsigned int,double>::iterator it=inflow_boundary.begin(); it!=inflow_boundary.end(); ++it)
	{
		vector_res(it->first)=0.;
	}*/

	for (unsigned int j=0; j!= vector_res.size(); j++)
	{
		if(std::fabs(vector_res(j)) > solver_tol)
		{
			std::cout<< "High residual:   "<< vector_res(j) <<std::endl;
		}
	}

	residual_vector.push_back(vector_res.l2_norm());
	//std::cout<< "Residual at time:  "<< time<< "   is :   "<<vector_res.l2_norm()<<"\n"<<
		//"with tolerance set to:    "<<solver_tol <<std::endl;
}

template <int dim>
void HyperbolicEquation<dim>::compute_total_error()
{
	double L1_sum, L2_sum;
	L1_sum=0;
	L2_sum=0;
	std::vector<double> L2_errors_squared;

	L2_errors_squared.push_back(L2_errors[0]*L2_errors[0]);
	/*if(cycle==1)
	{
	std::cout<< "L1, L2 errors vector size:   "<<L1_errors.size()<< ",   "
			<<L2_errors.size()<<std::endl;
	exit(0);
	}*/
	for(unsigned int j=0; j<L1_errors.size()-1; j++)
	{
		//std::cout<<"Error not here"<<std::endl;
		L1_sum+=(L1_errors[j]+L1_errors[j+1]);
		L2_errors_squared.push_back(L2_errors[j+1]*L2_errors[j+1]);
		L2_sum+=(L2_errors_squared[j]+L2_errors_squared[j+1]);
		//std::cout<<"Itearotor j  "<<j<<std::endl;
	}
	//std::cout<<"Error not here"<<std::endl;
	L1_sum*=(time_step/2.);
	L2_sum*=(time_step/2.);
	//std::cout<<"Error not here"<<std::endl;
	//total_L1_error.push_back(L1_sum);
	//total_L2_error.push_back(L2_sum);
	std::cout<< "L1, L2 errors vector size:   "<<L1_errors.size()<< ",   "
		<<L2_errors.size()<<std::endl;
	L1_errors.clear(); L1_errors.resize(n_times);
	L2_errors.clear(); L2_errors.resize(n_times);
	convergence_table.add_value("cycle", cycle);
	convergence_table.add_value("cells", triangulation.n_active_cells());
	convergence_table.add_value("dofs", dof_handler.n_dofs());
	convergence_table.add_value("L1", L1_sum);
	convergence_table.add_value("L2", std::sqrt(L2_sum));
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

	//std::string error_filename = "fp_visc_error.tex";
	//std::ofstream error_table_file(error_filename.c_str());
	//convergence_table.write_tex(error_table_file);

  	convergence_table.evaluate_convergence_rates("L1", ConvergenceTable::reduction_rate);
  	convergence_table.evaluate_convergence_rates("L1", ConvergenceTable::reduction_rate_log2);
  	convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
  	convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
  	std::cout << std::endl;
  	convergence_table.write_text(std::cout);
  	std::string conv_filename = "visc_burgers_convergence2.tex";
  	std::ofstream table_file(conv_filename.c_str());
  	convergence_table.write_tex(table_file);
}

template<int dim>
void HyperbolicEquation<dim>::run()
{
	//EXACT_SOLUTION ON MORE REFINED GRID
	//AND THEN INTERPOLATE SOLUTION ON THAT MORE REFINED GRID
	for(; cycle<=cycles_number; cycle++)
    {
		unsigned int exit_times=0, fp_times=0;
		//double exit_tol=2.*L2_errors[0]-tol;
		timestep_number = 0;
    	time            = 0;

		generate_grid();

    	setup_system();

    	set_initial_distribution();

    	output_results(initial_solution);
    	//assert(0);

    	//assemble_system();
    	system_matrix.copy_from(mass_matrix);
    	system_matrix.add(0.1*h_x*time_step*theta,stab_matrix);
    	system_rhs_matrix.copy_from(mass_matrix);
    	system_rhs_matrix.add(-0.1*h_x*time_step*(1-theta),stab_matrix);
    	//std::cout<< "L1, L2 errors vector size:   "<<L1_errors.size()<< ",   "
    		//	<<L2_errors.size()<<std::endl;

    	time += 2*time_step; //start from third timestep
    	timestep_number+=2;
    	//ent_solution=old_solution;

    	while (timestep_number < n_times)
      	{
            //std::vector<Vector<double>> fp_solutions;
        	time += time_step;
        	timestep_number++;

        	//std::cout <<std::setprecision(20)<< "Time step " << timestep_number << " at t=" << time
        	   //               	<< "Final time:   "<<t_end<< std::endl;
            //system_rhs_matrix.vmult(system_rhs, old_solution);
         	//assemble_rhs();

         	//system_rhs.add(1.,time_dependent_rhs);

            double current_fp_L2_error;

        	//VISCOSITY TERM
            ent_solution=old_solution;
            nu_vector.clear();
            viscosity_matrix.reinit(sparsity_pattern);
            assemble_viscosity_matrix();

         	system_matrix.add(theta*time_step, viscosity_matrix);
            system_rhs_matrix.add(-(1-theta)*time_step, viscosity_matrix);

         	system_rhs_matrix.vmult(system_rhs, old_solution);
         	assemble_nl_rhs(ent_solution);
         	//assemble_rhs();

         	system_rhs.add(-1.,time_dependent_rhs);
         	//system_rhs.add(1.,time_dependent_rhs);

         	//constraints.condense (system_matrix, system_rhs);

         	solve_time_step();


         	process_solution();
            current_fp_L2_error=L2_errors[timestep_number];

            time_dependent_rhs.reinit(dof_handler.n_dofs());
            //VISCOSITY TERM
            system_matrix.add(-theta*time_step, viscosity_matrix);
            system_rhs_matrix.add((1-theta)*time_step, viscosity_matrix);

         	//if(timestep_number>=2)
            if(L2_errors[timestep_number]>(4.+cycle)*L2_errors[0])
            {
                //std::cout<<"Difference : "<< L2_errors[timestep_number]-L2_errors[0]<<std::endl;
                unsigned int fp_steps=0;
                oldest_solution=older_solution;
                older_solution=old_solution;
				//old_solution=solution;
				//ent_solution=solution;

				while(fp_steps<1)
				{
					current_fp_L2_error=L2_errors[timestep_number];

					//old_solution=solution;
					ent_solution=solution;
					//fp_solutions.push_back(old_solution);
					nu_vector.clear();
					viscosity_matrix.reinit(sparsity_pattern);
					assemble_viscosity_matrix();

					system_matrix.add(theta*time_step, viscosity_matrix);
					system_rhs_matrix.add(-(1-theta)*time_step, viscosity_matrix);

					system_rhs_matrix.vmult(system_rhs, old_solution);
					assemble_nl_rhs(old_solution);
					//assemble_rhs();

					system_rhs.add(-1.,time_dependent_rhs);
					//system_rhs.add(1.,time_dependent_rhs);

					//constraints.condense (system_matrix, system_rhs);

					solve_time_step();

					process_solution();
					//std::cout<<"No problem here"<<std::endl;

					fp_steps++;

					time_dependent_rhs.reinit(dof_handler.n_dofs());
					system_matrix.add(-theta*time_step, viscosity_matrix);
					system_rhs_matrix.add((1-theta)*time_step, viscosity_matrix);

					if(L2_errors[timestep_number]>current_fp_L2_error)
					{
						exit_times++;
						//std::cout << "Time step " << timestep_number << " at t=" << time
        				//<< std::endl;
						//std::cout<< "Exit, L2_Errors:   "<<current_fp_L2_error<<"    "<<L2_errors[timestep_number]<<std::endl;
						L2_errors[timestep_number]=current_fp_L2_error;
						break;
						//exit(0);
					}

					else
					{
						if(L2_errors[timestep_number]<4*L2_errors[0])//-2*L2_errors[0]<(tol))
						{
							fp_times++;
							std::cout<<"fp iterations:   "<<fp_steps<<std::endl;
							std::cout<<"L2 errors before and after:   "<<current_fp_L2_error<<"    "<< L2_errors[timestep_number]<<std::endl;
							break;
						}


						if(fp_steps==10)
						{
							std::cout<< "Maximum number of fix point iteration reached!"<<std::endl;
							std::cout<< "L2_errors:   "<<current_fp_L2_error<<"    "<<L2_errors[timestep_number]<<std::endl;
						}
					}
				}
				//std::cout<<"fp iterations:   "<<fp_steps<<std::endl;
            }
            else
            {
            	oldest_solution=older_solution;
            	older_solution=old_solution;
            	//old_solution = solution;
            	//ent_solution=old_solution;
            	time_dependent_rhs.reinit(dof_handler.n_dofs());
            }


			//system_rhs_matrix.vmult(system_rhs, old_solution);
			//assemble_rhs();

			//system_rhs.add(1.,time_dependent_rhs);

         	//constraints.condense (system_matrix, system_rhs);

         	//solve_time_step();


        	if(timestep_number<=10)
        	{
        		std::cout << "Time step " << timestep_number << " at t=" << time
        				<< std::endl;
        		//std::cout<< "L2 norm of time dependent rhs should be 0: "<<time_dependent_rhs.l2_norm()<<std::endl;
        		//compute_residual();
        		output_results(solution);
        	}

        	if(timestep_number%(output_times)==0)
        	{
        		std::cout << "Time step " << timestep_number << " at t=" << time
                  	<< std::endl;
        		//std::cout<<"Fixed point steps"<<fp_steps<<std::endl;
        		output_results(solution);
        	}

        	old_solution=solution;

        	//process_solution();
        	//exact_solution.set_time(time);

        	//clear time-dependent objects before next iteration:
        	/*if(timestep_number>=2)
         	{
         		system_matrix.add(-theta*time_step, viscosity_matrix);
         		system_rhs_matrix.add((1-theta)*time_step, viscosity_matrix);
         	}*/

      	}
    	//std::cout <<"Last time step:   "<<timestep_number<<std::endl;
    	//triangulation.refine_global() in generate_grid()
    	compute_total_error();
    	inflow_boundary.clear();
    	//dof_handler.clear();
        //tol/=10;
        //std::cout<<"Fp times:  "<<fp_times<<std::endl;
       // std::cout<<"Exit times:  "<<exit_times<<std::endl;
    }

    write_table();
}

int main()
{
	try
	{
		//using namespace dealii;

		HyperbolicEquation<2> hyperbolic_equation;
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


