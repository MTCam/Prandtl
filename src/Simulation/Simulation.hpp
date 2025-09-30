#pragma once

#include "mfem.hpp"
#include "Prandtl.hpp"

namespace Prandtl
{

using namespace mfem;

class Simulation
{
private:
    int numProcs, myRank;
    int order;
    int dim;
    int num_equations;
    int ref_levels;
    int vis_steps;
    int nancheck_steps;
    int precision;
    int num_dofs_scalar;
    int num_dofs_system;
    int print_interval;
    int ti;
    int checkpoint_cycle;

    bool done = false;
    bool variable_dt = false;
    bool clock_simulation = true;
    bool nancheck = false;
    bool visualize = true;
    bool visit = false;
    bool paraview = true;
    bool checkpoint_load = false;
    bool checkpoint_save = false;

    std::shared_ptr<PhysicsConstants> physicsConstants;

    std::string output_file_path;
    std::string paraview_folder;
    std::string checkpoints_folder;

    real_t t, t_final, dt, dt_real;
    real_t cfl;
    real_t hmin;
    real_t Re, Ma;
    real_t next_save_t;
    real_t save_dt1;
    real_t save_dt2;
    real_t trigger_t;
    real_t save_dt;
    real_t next_checkpoint_t;
    real_t checkpoint_dt;

    real_t V_sq;

    real_t alpha_max;

    Array<int> mesh_ordering;
    std::shared_ptr<ParMesh> pmesh;

    int btype = BasisType::GaussLobatto;
    int ordering = Ordering::byNODES;
    std::shared_ptr<DG_FECollection> fec;
    std::shared_ptr<DG_FECollection> fec0;
    std::shared_ptr<ParFiniteElementSpace> vfes;
    std::shared_ptr<ParFiniteElementSpace> fes0;
    std::unique_ptr<ParFiniteElementSpace> fes;
    std::unique_ptr<ParFiniteElementSpace> dfes;

    std::unique_ptr<VectorFunctionCoefficient> u0;

    std::shared_ptr<ParGridFunction> sol;
    std::shared_ptr<ParGridFunction> dudx;
    std::shared_ptr<ParGridFunction> dudy;
    std::shared_ptr<ParGridFunction> dudz;

    std::shared_ptr<ParGridFunction> eta;
    std::shared_ptr<ParGridFunction> alpha;
    std::shared_ptr<ParGridFunction> r_gf;
    
    std::shared_ptr<NavierStokesFlux> flux;
    std::shared_ptr<NumericalFlux> numericalFlux;

    std::shared_ptr<LiftingScheme> liftingScheme;
    std::vector<std::shared_ptr<VectorFunctionCoefficient>> BC_coeff;

    FunctionCoefficient r_coef, z_coef;
    
    ParGridFunction rho, mom, energy;
    
    std::unique_ptr<ParGridFunction> u, v, w;
    std::unique_ptr<ParGridFunction> p, rho_axi;


    std::unique_ptr<ParaViewDataCollection> pd;
    std::unique_ptr<VisItDataCollection> vd;

    std::shared_ptr<ODESolver> ode_solver;
    std::unique_ptr<DGSEMOperator> NS;

    int signature;

    std::shared_ptr<BdrFaceIntegrator> bdr_face_integrator;
    std::vector<Array<int>> bdr_marker_vector;
    Array<int> set_marker;
    int max_bdr_attr;

#ifdef AXISYMMETRIC
    void ConservativeToPrimitive(const Vector &U_cons,
                                ParGridFunction &rho_out,
                                ParGridFunction &uz_out,
                                ParGridFunction &ur_out,
                                ParGridFunction &p_out) const;  
#endif

    Simulation();

    // change some shared_ptrs to unique_ptrs

public:    
    static Simulation& SimulationCreate();
    void LoadConfig(const std::string &config_file_path);

    ~Simulation();

    void Run();

    Simulation(const Simulation&) = delete;
    Simulation& operator = (const Simulation&) = delete;
};

}