#include "Simulation.hpp"
#include "ConditionFactory.hpp"

#include "LidDrivenCavity.hpp"
#include "TaylorGreenVortex.hpp"

#include "SodShockTube.hpp"
#include "ModifiedSodShockTube.hpp"
#include "LaxShockTube.hpp"
#include "LeBlancShockTube.hpp"
#include "Problem123.hpp"
#include "WoodwardColellaBlastWave.hpp"
#include "WoodwardColellaBlastWaveLeft.hpp"
#include "WoodwardColellaBlastWaveRight.hpp"
#include "WoodwardColellaBlastWaveCollision.hpp"
#include "ShuOsherShock.hpp"
#include "AcousticWave.hpp"

#include "DoubleMachReflection.hpp"
#include "ForwardFacingStep.hpp"
#include "BackwardFacingStep.hpp"
#include "TriplePointShockInteraction.hpp"
#include "KelvinHelmholtzInstability.hpp"
#include "IsentropicVortex.hpp"
#include "Ramp.hpp"
#include "Nagashima_Ramjet.hpp"

#include "json.hpp"
#include <filesystem>
#include <mpi.h>

namespace Prandtl
{

Simulation& Simulation::SimulationCreate()
{
    static Simulation sim;
    return sim;
}

Simulation::Simulation()
    : r_coef([](const Vector &X){ return X[1];}), z_coef([](const Vector &X){ return X[0];})
{
    Mpi::Init();
    numProcs = Mpi::WorldSize();
    myRank = Mpi::WorldRank();
    Hypre::Init();


    if (Mpi::Root())
    {
        std::cout << "================================================" << std::endl;
        std::cout << "DGSEM Simulation Starting with " << numProcs << " Processors!" << std::endl;
        std::cout << "================================================" << std::endl;       
        
#ifdef PARABOLIC
        std::cout << "The Navier-Stokes Equations will be Solved!" << std::endl;
#else
        std::cout << "The Euler Equations will be Solved!" << std::endl;
#endif

#ifdef AXISYMMETRIC
        std::cout << "Axisymmetric Mode will be Activated!" << std::endl;
#endif

#ifdef SUBCELL_FV_BLENDING
        std::cout << "Subcell Finite Volume Blending will be Activated!" << std::endl; 
#endif
    }
}

Simulation::~Simulation()
{
    if (Mpi::Root())
    {
        std::cout << "================================================" << std::endl;
        std::cout << "DGSEM Simulation Destroyed!" << std::endl;
        std::cout << "================================================" << std::endl;
    }
}

bool debug_simulation = false;

void Simulation::LoadConfig(const std::string &config_file_path)
{
    /*
    Load the configuration file and parse the settings
    */
    std::ifstream config_file(config_file_path);
    if (!config_file.is_open())
    {
        std::cerr << "Error Opening Configuration File at " << config_file_path << std::endl;
        return;
    }

    // Parse the JSON configuration file
    nlohmann::json config;
    config_file >> config;
    auto runtime = config["runTime"];

    order = runtime.value("order", 3);
    dim = runtime.value("dim", 2);
    num_equations = runtime.value("num_equations", 4);

    precision = runtime.value("precision", 15);
    std::cout.precision(precision);

    cfl = runtime.value("cfl", 1.0);
    print_interval = runtime.value("print_interval", 10);
    output_file_path = runtime["output_file_path"].get<std::string>();
    paraview_folder = runtime.value("paraview_folder", "ParaView");
    checkpoint_dt = runtime.value("checkpoint_dt", 0.01);
    checkpoints_folder = output_file_path + "/" + runtime.value("checkpoints_folder", "Checkpoints");
    checkpoint_load = runtime.value("checkpoint_load", false);
    checkpoint_save = runtime.value("checkpoint_save", false);
    if (checkpoint_save) 
    {
        std::filesystem::create_directories(checkpoints_folder);
    }
    
    visualize = runtime["visualize"].get<bool>();
    if (visualize)
    {
        save_dt1 = runtime.value("initial_save_dt", 0.01);
        save_dt2 = runtime.value("refined_save_dt", save_dt1);
        trigger_t = runtime.value("refined_trigger", 2.0);
        save_dt = save_dt1;
        vis_steps = runtime.value("vis_steps", 100);
        paraview = runtime["paraview"].get<bool>();
        visit = runtime["visit"].get<bool>();
        if (!paraview && !visit)
        {
            std::cerr << "Error: Both ParaView and VisIt visualization options are disabled. Please choose at least one." << std::endl;
            return;
        }
    }

    nancheck = runtime["nancheck"].get<bool>();
    if (nancheck)
    {
        nancheck_steps = runtime.value("nancheck_steps", 1000);
    }

    clock_simulation = runtime["clock_simulation"].get<bool>();
    variable_dt = runtime["variable_dt"].get<bool>();

    if (runtime.contains("dt") && !variable_dt)
    {
        dt = runtime.value("dt", 1e-4);
    }
    t_final = runtime["final_time"].get<real_t>();

    physicsConstants = std::make_shared<PhysicsConstants>(
        runtime.value("gamma", 1.4),
        runtime.value("Pr", 0.72),
        runtime.value("R_gas", 287.05),
        runtime.value("mu", 0.02));


    if (runtime.contains("lifting_scheme"))
    {
        if (runtime["lifting_scheme"].get<std::string>() == "BR1")
        {
            liftingScheme = std::make_shared<LiftingBR1>();
        }
        else
        {
            std::cerr << "Error: Invalid lifting scheme specified." << std::endl;
            return;
        }
    }
    else
    {
        liftingScheme = nullptr;
    }

    std::string ode_solver_string = runtime["ode_solver"].get<std::string>();
    if (ode_solver_string == "ForwardEuler")
    {
        ode_solver = std::make_shared<ForwardEulerSolver>();
    }
    else if (ode_solver_string == "RK2")
    {
        ode_solver = std::make_shared<RK2Solver>();
    }
    else if (ode_solver_string == "RK3SSP")
    {
        ode_solver = std::make_shared<RK3SSPSolver>();
    }
    else if (ode_solver_string == "RK4")
    {
        ode_solver = std::make_shared<RK4Solver>();
    }
    else if (ode_solver_string == "RK6")
    {
        ode_solver = std::make_shared<RK6Solver>();
    }
    else if (ode_solver_string == "RK8")
    {
        ode_solver = std::make_shared<RK8Solver>();
    }
    else
    {
        std::cerr << "Error: Invalid ODE solver specified." << std::endl;
        return;
    }

    flux = std::make_shared<NavierStokesFlux>(dim,
        physicsConstants->gamma, physicsConstants->Pr, physicsConstants->mu, physicsConstants->mu0,
        physicsConstants->mu_bulk, physicsConstants->R_gas, physicsConstants->Ts, physicsConstants->T0);

    if (runtime["numerical_flux"].get<std::string>() == "Chandrashekar")
    {
        numericalFlux = std::make_shared<ChandrashekarFlux>(*flux, physicsConstants->gamma);
    }
    else
    {
        std::cerr << "Error: Invalid numerical flux specified." << std::endl;
        return;
    }


    signature = runtime["conditions"]["initial_conditions"].value("signature", 0);
    std::string IC_key = runtime["conditions"]["initial_conditions"].value("function", "LidDrivenCavityIC");

    if (signature == 0)
    {
        u0 = std::make_unique<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetInitialCondition0(IC_key)());
    }
    else if (signature == 1)
    {
        real_t x1 = runtime["conditions"]["initial_conditions"]["params"].value("x1", 0.0);
        u0 = std::make_unique<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetInitialCondition1(IC_key)(x1));
    }
    else if (signature == 2)
    {
        real_t x1 = runtime["conditions"]["initial_conditions"]["params"].value("x1", 0.0);
        real_t x2 = runtime["conditions"]["initial_conditions"]["params"].value("x2", 0.0);
        u0 = std::make_unique<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetInitialCondition2(IC_key)(x1, x2));
    }
    else if (signature == 3)
    {
        real_t x1 = runtime["conditions"]["initial_conditions"]["params"].value("x1", 0.0);
        real_t x2 = runtime["conditions"]["initial_conditions"]["params"].value("x2", 0.0);
        real_t x3 = runtime["conditions"]["initial_conditions"]["params"].value("x3", 0.0);
        u0 = std::make_unique<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetInitialCondition3(IC_key)(x1, x2, x3));
    }
    else if (signature == 4)
    {
        real_t x1 = runtime["conditions"]["initial_conditions"]["params"].value("x1", 0.0);
        real_t x2 = runtime["conditions"]["initial_conditions"]["params"].value("x2", 0.0);
        real_t x3 = runtime["conditions"]["initial_conditions"]["params"].value("x3", 0.0);
        real_t x4 = runtime["conditions"]["initial_conditions"]["params"].value("x4", 0.0);
        u0 = std::make_unique<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetInitialCondition4(IC_key)(x1, x2, x3, x4));
    }
    else if (signature == 5)
    {
        real_t x1 = runtime["conditions"]["initial_conditions"]["params"].value("x1", 0.0);
        real_t x2 = runtime["conditions"]["initial_conditions"]["params"].value("x2", 0.0);
        real_t x3 = runtime["conditions"]["initial_conditions"]["params"].value("x3", 0.0);
        real_t x4 = runtime["conditions"]["initial_conditions"]["params"].value("x4", 0.0);
        real_t x5 = runtime["conditions"]["initial_conditions"]["params"].value("x5", 0.0);
        u0 = std::make_unique<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetInitialCondition5(IC_key)(x1, x2, x3, x4, x5));
    }
    else
    {
        std::cerr << "Error: Invalid initial condition signature." << std::endl;
        return;
    }

    Mesh *mesh;
    mesh = new Mesh(runtime["mesh_file"].get<std::string>());
    bool periodic;
    if (runtime.contains("periodic"))
    {
        periodic = runtime["periodic"].get<bool>();
    }
    else
    {
        periodic = false;
    }

    if (dim == 1 && !periodic)
    {
        Array<int> left;
        Array<int> right;
        left.Append(1);
        right.Append(2);
        mesh->bdr_attribute_sets.SetAttributeSet("left", left);
        mesh->bdr_attribute_sets.SetAttributeSet("right", right);
    }

    if (runtime.contains("ser_ref_levels"))
    {
        ref_levels = runtime.value("ser_ref_levels", 0);
        for (int lev = 0; lev < ref_levels; lev++)
        {
            mesh->UniformRefinement();
        }
    }

    if (mesh->GetNE() < Mpi::WorldSize())
    {
        std::cerr << "Error: Number of elements is less than number of processors." << std::endl;
        return;
    }

    if (runtime.contains("mesh_ordering"))
    {
        if (runtime["mesh_ordering"].get<std::string>() == "Hilbert")
        {
            mesh->GetHilbertElementOrdering(mesh_ordering);
        }
        else if (runtime["mesh_ordering"].get<std::string>() == "Gecko")
        {
            mesh->GetGeckoElementOrdering(mesh_ordering);
        }
        mesh->ReorderElements(mesh_ordering);
    }

    if (dim > 1)
    {
        mesh->EnsureNCMesh();
    }

    pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, *mesh);
    mesh->Clear();

    if (runtime.contains("par_ref_levels"))
    {
        ref_levels = runtime.value("par_ref_levels", 0);
        for (int lev = 0; lev < ref_levels; lev++)
        {
            pmesh->UniformRefinement();
        }
    }

    fec = std::make_shared<DG_FECollection>(order, dim, btype);
    fec0 = std::make_shared<DG_FECollection>(0, dim);
    vfes = std::make_shared<ParFiniteElementSpace>(pmesh.get(), fec.get(), num_equations, ordering);
    fes0 = std::make_shared<ParFiniteElementSpace>(pmesh.get(), fec0.get());
    dfes = std::make_unique<ParFiniteElementSpace>(pmesh.get(), fec.get(), dim, ordering);
    fes = std::make_unique<ParFiniteElementSpace>(pmesh.get(), fec.get());

    num_dofs_scalar = fes->GetNDofs();
    num_dofs_system = vfes->GetVSize();
    
    // sol = std::make_shared<ParGridFunction>(vfes.get());

    if (checkpoint_load)
    {
        real_t root_t = 0.0;
        int root_ti = 0;
 
        if (Mpi::Root())
        {
            checkpoint_cycle = runtime.value("checkpoint_cycle", 0);
            MFEM_VERIFY(checkpoint_cycle > 0, "Invalid or missing cycle number in JSON");
                std::string meta_file = checkpoints_folder + "/Cycle" + std::to_string(checkpoint_cycle) + "/checkpoint_cycle_" + std::to_string(checkpoint_cycle) + ".json";
                
                std::ifstream meta(meta_file);
                MFEM_VERIFY(meta, "Failed to open meta file " << meta_file);
                    nlohmann::json J;
                    meta >> J;
                    root_t = J.value("time", 0.0);
                    root_ti = J.value("cycle", 0);
                    MFEM_VERIFY(root_ti == checkpoint_cycle, "Mismatch between provided cycle number and value in meta file");
        }

        MPI_Bcast(&root_t, 1, MPI_DOUBLE, 0, pmesh->GetComm());
        MPI_Bcast(&root_ti, 1, MPI_INT, 0, pmesh->GetComm());
        MPI_Barrier(pmesh->GetComm());

        t = root_t;
        ti = root_ti;

        std::ostringstream fname;
        fname << checkpoints_folder << "/Cycle" << ti << "/checkpoint_cycle_" << ti << "." << std::setw(8) << std::setfill('0') << myRank << ".chk";
        std::cout << fname.str() << "\n";
        std::ifstream checkpoint_load(fname.str(), std::ios::binary);
        MFEM_VERIFY(checkpoint_load, "Failed to open checkpoint file for reading: " << fname.str());

        sol.reset();
        sol = std::make_shared<ParGridFunction>(pmesh.get(), checkpoint_load);

        MPI_Barrier(pmesh->GetComm());

    }
    else 
    {
        sol = std::make_shared<ParGridFunction>(vfes.get());
        sol->ProjectCoefficient(*u0);

        t = 0.0;
        ti = 0;
    }

#ifdef AXISYMMETRIC
    r_coef = FunctionCoefficient([](const Vector &X){
        return X[1]; });
    r_gf = std::make_shared<ParGridFunction>(fes.get());
    r_gf->ProjectCoefficient(r_coef);
    
    real_t *r_data = r_gf->GetData();
    real_t *sol_data = sol->GetData();

    for (int i = 0; i < num_dofs_scalar; ++i)
        {
            for (int j=0; j < num_equations ; ++j)
            {
               sol_data[j*num_dofs_scalar+i] *= r_data[i]; 
            }
        }

    if (debug_simulation)
    {
        real_t *sol_state = sol->GetData();
        std::vector<std::pair<real_t, real_t>> zr(num_dofs_scalar, {0.0, 0.0});

        std::cout << "\n === sol state rU values after weighting by r ===\n";

        for (int e = 0; e < pmesh->GetNE(); e++)
        {
            const FiniteElement &fe = *fes->GetFE(e);
            ElementTransformation &Tr = *fes->GetElementTransformation(e);

            Array<int> ldofs;
            fes->GetElementDofs(e, ldofs);

            const IntegrationRule &fe_nodes = fe.GetNodes();
            if (fe_nodes.GetNPoints() == fe.GetDof())
            {
                for (int ldof = 0; ldof < fe.GetDof(); ldof++)
                {
                    const IntegrationPoint &ip = fe_nodes.IntPoint(ldof);
                    Vector X(dim);
                    Tr.Transform(ip, X);
                    const int gdof = ldofs[ldof];
                    zr[gdof] = {static_cast<real_t>(X(0)), static_cast<real_t>(X(1))};
                }
            }
        }

        for (int i = 0; i < num_dofs_scalar; ++i)
        {
            real_t rho = sol_state[0*num_dofs_scalar+i];
            real_t rhoU = sol_state[1*num_dofs_scalar+i];
            real_t rhoV = sol_state[2*num_dofs_scalar+i];
            real_t E = sol_state[3*num_dofs_scalar+i];
            real_t z = zr[i].first;
            real_t r = zr[i].second;

            std::cout << " DOF#" << std::setw(2) << i 
                    << " (z, r) = ("<< std::fixed << std::setprecision(2) << std::setw(4) << z //std::round(z*100)/100.0
                    << ", " << std::setw(4) << std::round(r*100)/100.0 << "),  state = ["
                    << std::setw(4) << std::round(rho*100)/100.0 << ", "
                    << std::setw(4) << std::round(rhoU*100)/100.0 << ", "
                    << std::setw(4) << std::round(rhoV*100)/100.0 << ", "
                    << std::setw(5) << std::round(E*100)/100.0 << "]\n";
        }
       
    }
#endif

    next_checkpoint_t = t + checkpoint_dt;
    if (visualize) { next_save_t = t + save_dt; }

    eta = std::make_shared<ParGridFunction>(fes0.get());
    alpha = std::make_shared<ParGridFunction>(fes0.get());

    dudx = std::make_shared<ParGridFunction>(vfes.get());
    dudy = std::make_shared<ParGridFunction>(vfes.get());
    dudz = std::make_shared<ParGridFunction>(vfes.get());

    Geometry::Type gtype = vfes->GetFE(0)->GetGeomType();

    if (runtime.contains("alpha_max"))
    {
        alpha_max = runtime["alpha_max"].get<real_t>();
    }
    else
    {
        alpha_max = 0.5;
    }

    NS = std::make_unique<DGSEMOperator>(vfes, fes0, pmesh, eta, alpha, dudx, dudy, dudz,
        std::make_unique<Prandtl::DGSEMIntegrator>(pmesh, fes0, alpha, liftingScheme, *numericalFlux, order + 1, physicsConstants->gamma),
        std::make_unique<Prandtl::PerssonPeraireIndicator>(vfes, fes0, eta, std::make_shared<Prandtl::ModalBasis>(*fec, gtype, order, dim), physicsConstants->gamma), physicsConstants->gamma, r_gf, alpha_max);

    if (runtime["conditions"].contains("boundary_conditions"))
    {
        max_bdr_attr = pmesh->bdr_attributes.Max();  
        bdr_marker_vector.reserve(max_bdr_attr + 1);
        auto boundaries = runtime["conditions"]["boundary_conditions"];
        for (auto& boundary : boundaries.items())
        {
            std::string boundaryName = boundary.key();
            if (!pmesh->bdr_attribute_sets.AttributeSetExists(boundaryName))
            {
                // This rank has no faces with this boundary name; skip
                continue;
            }
            Array<int> marker(max_bdr_attr);
            marker = 0;
            // bdr_marker_vector.push_back(Array<int>(max_bdr_attr));
            set_marker = pmesh->bdr_attribute_sets.GetAttributeSetMarker(boundaryName);
            for (int b = 0; b < max_bdr_attr; b++)
            {
                if (set_marker[b])
                {
                    // bdr_marker_vector.back()[b] = 1;
                    marker[b] = 1;
                }
            }
            bdr_marker_vector.push_back(marker);

            auto bc_props = boundary.value();  // This is a JSON object.
            std::string type = bc_props["type"].get<std::string>();

            if (type == "symmetry" || type == "axis")
            {
                auto symmetry = std::make_unique<SymmetryBdrFaceIntegrator>(liftingScheme, *numericalFlux, order + 1, NS->GetTimeRef(), physicsConstants->gamma, true, false);

                NS->AddBdrFaceIntegrator(symmetry.release(), bdr_marker_vector.back());

#ifdef AXISYMMETRIC
                if (type == "axis")
                {
                   NS->SetAxisBoundaryMarker(bdr_marker_vector.back());
                   NS->SetLowOrderAxis(true);
                }
#endif
            }
            else if (type == "slip")
            {
                auto slip = std::make_unique<SlipWallBdrFaceIntegrator>(liftingScheme, *numericalFlux, order + 1, NS->GetTimeRef(), physicsConstants->gamma, true, false);

                NS->AddBdrFaceIntegrator(slip.release(), bdr_marker_vector.back());

            }
            else if (type == "no-slip-adiabatic")
            {   
                if (bc_props["velocity"].contains("vector"))
                {
                    std::string velBC_key = bc_props["velocity"]["vector"].get<std::string>();
                    std::string heatBC_key = bc_props["heat"]["scalar"].get<std::string>();
                    NS->AddBdrFaceIntegrator(
                        new NoSlipAdiabWallBdrFaceIntegrator(liftingScheme, *numericalFlux, order + 1, NS->GetTimeRef(), physicsConstants->gamma,
                            ConditionFactory::Instance().GetScalarBoundaryCondition(heatBC_key),
                            ConditionFactory::Instance().GetVectorBoundaryCondition(velBC_key)), bdr_marker_vector.back());
                }
                else if (bc_props["velocity"].contains("function"))
                {
                    std::shared_ptr<VectorFunctionCoefficient> velBC;
                    std::shared_ptr<FunctionCoefficient> heatBC;

                    std::string velBC_key = bc_props["velocity"]["function"].get<std::string>();
                    std::string heatBC_key = bc_props["heat"]["function"].get<std::string>();

                    bool td;
                    if (bc_props["velocity"].contains("time_dependent"))
                    {
                        td = bc_props["velocity"]["time_dependent"].get<bool>();
                    }
                    else
                    {
                        td = false;
                    }

                    signature = bc_props["velocity"]["signature"].get<int>();
                    if (signature == 0)
                    {
                        velBC = std::make_shared<VectorFunctionCoefficient>(dim, ConditionFactory::Instance().GetVectorFunctionBoundaryCondition0(velBC_key)());
                    }
                    else if (signature == 1)
                    {
                        real_t x1 = bc_props["velocity"]["params"].value("x1", 0.0);
                        velBC = std::make_shared<VectorFunctionCoefficient>(dim, ConditionFactory::Instance().GetVectorFunctionBoundaryCondition1(velBC_key)(x1));
                    }
                    else if (signature == 2)
                    {
                        real_t x1 = bc_props["velocity"]["params"].value("x1", 0.0);
                        real_t x2 = bc_props["velocity"]["params"].value("x2", 0.0);
                        velBC = std::make_shared<VectorFunctionCoefficient>(dim, ConditionFactory::Instance().GetVectorFunctionBoundaryCondition2(velBC_key)(x1, x2));
                    }
                    else
                    {
                        std::cerr << "Error: Invalid boundary condition signature." << std::endl;
                        return;
                    }

                    signature = bc_props["heat"]["signature"].get<int>();
                    if (signature == 0)
                    {
                        if (td)
                        {
                            heatBC = std::make_shared<FunctionCoefficient>(ConditionFactory::Instance().GetScalarTDFunctionBoundaryCondition0(heatBC_key)());
                        }
                        else
                        {
                            heatBC = std::make_shared<FunctionCoefficient>(ConditionFactory::Instance().GetScalarFunctionBoundaryCondition0(heatBC_key)());
                        }
                    }
                    else if (signature == 1)
                    {
                        real_t x1 = bc_props["heat"]["params"].value("x1", 0.0);
                        if (td)
                            heatBC = std::make_shared<FunctionCoefficient>(ConditionFactory::Instance().GetScalarTDFunctionBoundaryCondition1(heatBC_key)(x1));
                        else
                        {
                            heatBC = std::make_shared<FunctionCoefficient>(ConditionFactory::Instance().GetScalarFunctionBoundaryCondition1(heatBC_key)(x1));
                        }
                    }
                    else if (signature == 2)
                    {
                        real_t x1 = bc_props["heat"]["params"].value("x1", 0.0);
                        real_t x2 = bc_props["heat"]["params"].value("x2", 0.0);
                        if (td)
                            heatBC = std::make_shared<FunctionCoefficient>(ConditionFactory::Instance().GetScalarTDFunctionBoundaryCondition2(heatBC_key)(x1, x2));
                        else
                        {
                            heatBC = std::make_shared<FunctionCoefficient>(ConditionFactory::Instance().GetScalarFunctionBoundaryCondition2(heatBC_key)(x1, x2));
                        }
                    }
                    else
                    {
                        std::cerr << "Error: Invalid boundary condition signature." << std::endl;
                        return;
                    }

                    NS->AddBdrFaceIntegrator(
                        new NoSlipAdiabWallBdrFaceIntegrator(liftingScheme, *numericalFlux, order + 1, NS->GetTimeRef(), physicsConstants->gamma, *heatBC, *velBC, td), bdr_marker_vector.back());
                }
                else
                {
                    std::cerr << "Error: Invalid boundary condition type specified." << std::endl;
                    return;
                }
            }
            else if (type == "no-slip-isothermal")
            {

            }
            else if (type == "supersonic-outflow")
            {
                auto outlet = std::make_unique<SupersonicOutflowBdrFaceIntegrator>(liftingScheme, *numericalFlux, order + 1, NS->GetTimeRef(), physicsConstants->gamma);

                NS->AddBdrFaceIntegrator(outlet.release(), bdr_marker_vector.back());

            }
            else if (type == "supersonic-inflow")
            {
                if (bc_props.contains("vector"))
                {
                std::string state_key = bc_props["vector"].get<std::string>();
                auto inlet = std::make_unique<SupersonicInflowBdrFaceIntegrator>(liftingScheme, *numericalFlux, order + 1, NS->GetTimeRef(), physicsConstants->gamma,
                        ConditionFactory::Instance().GetVectorBoundaryCondition(state_key));

                    NS->AddBdrFaceIntegrator(inlet.release(), bdr_marker_vector.back());

                }
                else
                {
                    std::string state_key = bc_props["function"].get<std::string>();
                    signature = bc_props["signature"].get<int>();
                    std::shared_ptr<VectorFunctionCoefficient> stateBC;
                    bool td;
                    if (bc_props.contains("time_dependent"))
                    {
                        td = bc_props["time_dependent"].get<bool>();
                    }
                    else
                    {
                        td = false;
                    }

                    if (signature == 0)
                    {
                        if (td)
                        {
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition0(state_key)());
                        }
                        else
                        {
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorFunctionBoundaryCondition0(state_key)());
                        }
                        
                    }
                    else if (signature == 1)
                    {
                        real_t x1 = bc_props["params"].value("x1", 0.0);
                        if (td)
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition1(state_key)(x1));
                        else
                        {
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorFunctionBoundaryCondition1(state_key)(x1));
                        }
                    }
                    else if (signature == 2)
                    {
                        real_t x1 = bc_props["params"].value("x1", 0.0);
                        real_t x2 = bc_props["params"].value("x2", 0.0);
                        if (td)
                        {
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition2(state_key)(x1, x2));
                        }
                        else
                        {
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorFunctionBoundaryCondition2(state_key)(x1, x2));
                        }
                    
                    }
                    else
                    {
                        std::cerr << "Error: Invalid boundary condition signature." << std::endl;
                        return;
                    }
                    auto inlet = std::make_unique<SupersonicInflowBdrFaceIntegrator>(liftingScheme, *numericalFlux, order + 1, NS->GetTimeRef(), physicsConstants->gamma,
                        *stateBC, td);

                    NS->AddBdrFaceIntegrator(inlet.release(), bdr_marker_vector.back());

                }
            }
            else if (type == "specified-state")
            {
                if (bc_props.contains("vector"))
                {
                    std::string state_key = bc_props["vector"].get<std::string>();
                    NS->AddBdrFaceIntegrator(new SpecifiedStateBdrFaceIntegrator(liftingScheme, *numericalFlux, order + 1, NS->GetTimeRef(), physicsConstants->gamma,
                        ConditionFactory::Instance().GetVectorBoundaryCondition(state_key)), bdr_marker_vector.back());
                }
                else
                {
                    std::string state_key = bc_props["function"].get<std::string>();
                    signature = bc_props["signature"].get<int>();
                    std::shared_ptr<VectorFunctionCoefficient> stateBC;
                    bool td;
                    if (bc_props.contains("time_dependent"))
                    {
                        td = bc_props["time_dependent"].get<bool>();
                    }
                    else
                    {
                        td = false;
                    }

                    if (signature == 0)
                    {
                        if (td)
                        {
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition0(state_key)());
                        }
                        else
                        {
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorFunctionBoundaryCondition0(state_key)());
                        }
                        
                    }
                    else if (signature == 1)
                    {
                        real_t x1 = bc_props["params"].value("x1", 0.0);
                        if (td)
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition1(state_key)(x1));
                        else
                        {
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorFunctionBoundaryCondition1(state_key)(x1));
                        }
                    }
                    else if (signature == 2)
                    {
                        real_t x1 = bc_props["params"].value("x1", 0.0);
                        real_t x2 = bc_props["params"].value("x2", 0.0);
                        if (td)
                        {
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition2(state_key)(x1, x2));
                        }
                        else
                        {
                            stateBC = std::make_shared<VectorFunctionCoefficient>(num_equations, ConditionFactory::Instance().GetVectorFunctionBoundaryCondition2(state_key)(x1, x2));
                        }
                    
                    }
                    else
                    {
                        std::cerr << "Error: Invalid boundary condition signature." << std::endl;
                        return;
                    }
                    NS->AddBdrFaceIntegrator(new SpecifiedStateBdrFaceIntegrator(liftingScheme, *numericalFlux, order + 1, NS->GetTimeRef(), physicsConstants->gamma, *stateBC, td), bdr_marker_vector.back());
                }
            }
            else
            {
                std::cerr << "Error: Invalid boundary condition type specified." << std::endl;
                return;
            }
        }
    }

    if (Mpi::Root())
    {
        std::cout << "The Number of Degrees of Freedom per Conservative Variable per Rank: " << num_dofs_scalar << std::endl;
        std::cout << "The Number of Degrees of Freedom (System) per Rank: " << num_dofs_system << std::endl;
    }

    NS->SetTime(t);
    ode_solver->Init(*NS);

    rho.MakeRef(fes.get(), *sol, 0);
    mom.MakeRef(dfes.get(), *sol, num_dofs_scalar);
    energy.MakeRef(fes.get(), *sol, (dim + 1) * num_dofs_scalar);

    u = std::make_unique<ParGridFunction>(fes.get());
    if (dim > 1)
    {
        v = std::make_unique<ParGridFunction>(fes.get()); 
        if (dim > 2)
        {
            w = std::make_unique<ParGridFunction>(fes.get());
        }
    }
    p = std::make_unique<ParGridFunction>(fes.get());

#ifdef AXISYMMETRIC
    rho_axi = std::make_unique<ParGridFunction>(fes.get());
#endif
    
    
    if (visualize)
    {
        if (paraview)
        {
            pd = std::make_unique<ParaViewDataCollection>(paraview_folder, pmesh.get());
            pd->SetPrefixPath(output_file_path);
#ifdef AXISYMMETRIC
            pd->RegisterField("Density", rho_axi.get());
#else
            pd->RegisterField("Density", &rho);
#endif
            pd->RegisterField("Horizontal V", u.get());
            if (dim > 1)
            {
                pd->RegisterField("Vertical V", v.get());
                if (dim > 2)
                {
                    pd->RegisterField("Normal V", w.get());
                }
            }
            pd->RegisterField("Pressure", p.get());
            pd->RegisterField("Blending Coeff", alpha.get());
            pd->SetLevelsOfDetail(order);
            pd->SetDataFormat(VTKFormat::BINARY);
            pd->SetHighOrderOutput(true);
        }
        else if (visit)
        {
            vd = std::make_unique<VisItDataCollection>("VisIt", pmesh.get());
            vd->SetPrefixPath(output_file_path);
            vd->SetPrecision(precision);
            vd->SetFormat(DataCollection::PARALLEL_FORMAT);

#ifdef AXISYMMETRIC
            vd->RegisterField("Density", rho_axi.get());
#else
            vd->RegisterField("Density", &rho);
#endif
            vd->RegisterField("Horizontal V", u.get());
            if (dim > 1)
            {
                vd->RegisterField("Vertical V", v.get());
                if (dim > 2)
                {
                    vd->RegisterField("Normal V", w.get());
                }
            }
            vd->RegisterField("Pressure", p.get());
            vd->RegisterField("Blending Coeff", alpha.get());
        }
    }
}

void Simulation::Run()
{
    if (Mpi::Root())
    {
        std::cout << "================================================" << std::endl;
        std::cout << "DGSEM Simulation Running Now!" << std::endl;
        std::cout << "================================================" << std::endl;
    }

    // Get the minimum characteristic element size and compute the initial time step if time step is variable
 
    if (variable_dt && cfl > 0.0)
    {
        
        hmin = infinity();
        for (int i = 0; i < pmesh->GetNE(); i++)
        {
            hmin = std::min(pmesh->GetElementSize(i, 1), hmin);
        }
        MPI_Allreduce(MPI_IN_PLACE, &hmin, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh->GetComm());
        Vector z(sol->Size());
        NS->Mult(*sol, z);
        real_t max_char_speed = NS->GetMaxCharSpeed();
        MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1,  MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh->GetComm());
        dt = cfl * hmin / (max_char_speed * std::pow(order + 1, 2));
    }

    // Clock the simulation?
    if (clock_simulation)
    {
        tic_toc.Clear();
        tic_toc.Start();
    }

    // print the first node:
#ifdef AXISYMMETRIC
        Vector U_cons(sol->Size());
        NS->RecoverStateFromWeighted(*sol, U_cons);
        ConservativeToPrimitive(U_cons, *rho_axi, *u, *v, *p);
#endif
    if (Mpi::Root())
        {
            int i = 0;  
#ifdef AXISYMMETRIC
            real_t rhoi = (*rho_axi)(i);
            real_t ui = (*u)(i);               
            real_t vi = (*v)(i);
            real_t pi = (*p)(i);

            NS->SetAxisFloorsFromFreestream(rhoi, pi);
#else
            real_t rhoi = rho(i);
            real_t ui = mom(i)/rhoi;               
            real_t vi = mom(i + num_dofs_scalar)/rhoi;
            real_t pi = physicsConstants->gammaM1 * (energy(i) - 0.5*rhoi*ui*ui);
#endif
            std::cout << "***  initial at dof #" << i << ":  "
                        << "rho = " << std::round(rhoi*10000)/10000 << ",  u = " << std::round(ui*100)/100 << ",  v = " << std::round(vi*100)/100 << ",  p = " << std::round(pi*100)/100 << "\n";
        }
    // Visualize the initial condition?
    if (visualize)
    {
#ifdef AXISYMMETRIC

        if (debug_simulation)
        {
            for (int i = 0; i < num_dofs_scalar; i++)
            {
                std::cout << " DOF#" << std::setw(2) << i 
                        << "    primitive state =["
                        << std::setw(4) << std::round((*rho_axi)(i)*100)/100.0 << ", "
                        << std::setw(4) << std::round((*u)(i)*100)/100.0 << ", "
                        << std::setw(4) << std::round((*v)(i)*100)/100.0 << ", "
                        << std::setw(5) << std::round((*p)(i)*100)/100.0 << "]\n";
            }
       
        }

        
#else
        for (int i = 0; i < num_dofs_scalar; i++)
        {
            (*u)(i) = mom(i) / rho(i);
            V_sq = (*u)(i) * (*u)(i);
            if (dim > 1)
            {
                (*v)(i) = mom(i + num_dofs_scalar) / rho(i);
                V_sq += (*v)(i) * (*v)(i);
                if (dim > 2)
                {
                    (*w)(i) = mom(i + 2 * num_dofs_scalar) / rho(i);
                    V_sq += (*w)(i) * (*w)(i);
                }
            }
            (*p)(i) = physicsConstants->gammaM1 * (energy(i) - 0.5 * rho(i) * V_sq);
        }
#endif

        if (paraview)
        {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
        }
        else if (visit)
        {
            vd->SetCycle(ti);
            vd->SetTime(t);
            vd->Save();
        }
    }

    
    while (!done)
    {
        
        if (debug_simulation)
        {
            std::cout << "################################################################################" << "\n";
            std::cout << "######################### [TIME STEP = " << ti << ", TIME = " << t << "] #########################" << "\n";
            std::cout << "################################################################################" << "\n";
        }
    

        // Compute the time step size
        dt_real = std::min(dt, t_final - t);

        // Perform the time step
        ode_solver->Step(*sol, t, dt_real);

        // Update the time step size with CFL?
        if (variable_dt && cfl > 0)
        {
            real_t max_char_speed = NS->GetMaxCharSpeed();
            MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh->GetComm());
            dt = cfl * hmin / (max_char_speed * std::pow(order + 1, 2));
        }
        ti++;

        // Check for completion
        done = (t >= t_final - 1e-8 * dt);

        // Check for NaN/Inf values?
        if (nancheck && ti % nancheck_steps == 0)
        {
            for (const real_t &val : rho)
            {
                if (std::isnan(val) || std::isinf(val))
                {
                    MFEM_ABORT("NaN/Inf Detected at Time Step " + std::to_string(ti) + " on Rank " + std::to_string(myRank));
                    break;
                }
            }
        }
        // Visualize the solution?
        // if (visualize && (done || ti % vis_steps == 0))
        if (visualize && (done || t >= next_save_t))
        {
        
#ifdef AXISYMMETRIC
        Vector U_cons(sol->Size());
        NS->RecoverStateFromWeighted(*sol, U_cons);
        ConservativeToPrimitive(U_cons, *rho_axi, *u, *v, *p);
#else
        for (int i = 0; i < num_dofs_scalar; i++)
        {       
                (*u)(i) = mom(i) / rho(i);
                V_sq = (*u)(i) * (*u)(i);
                if (dim > 1)
                {
                    (*v)(i) = mom(i + num_dofs_scalar) / rho(i);
                    V_sq += (*v)(i) * (*v)(i);
                    if (dim > 2)
                    {
                        (*w)(i) = mom(i + 2 * num_dofs_scalar) / rho(i);
                        V_sq += (*w)(i) * (*w)(i);
                    }
                }
                (*p)(i) = physicsConstants->gammaM1 * (energy(i) - 0.5 * rho(i) * V_sq);
        }
#endif

            if (paraview)
            {
                pd->SetCycle(ti);
                pd->SetTime(t);
                pd->Save();
            }
            else if (visit)
            {
                vd->SetCycle(ti);
                vd->SetTime(t);
                vd->Save();
            }


            save_dt = (t < trigger_t) ? save_dt1 : save_dt2;
            next_save_t += save_dt;

        }


        if (checkpoint_save && (done || t >= next_checkpoint_t))
        {
            // writing the solution to a checkpoint file in a subfolder

            std::string cycle_dir = checkpoints_folder + "/Cycle" + std::to_string(ti);
            std::error_code ec;
            std::filesystem::create_directories(cycle_dir, ec);
            MFEM_VERIFY(!ec, "Failed to create a directory " << cycle_dir << " : " << ec.message());

            std::ostringstream checkpoint_file;
            checkpoint_file << cycle_dir << "/checkpoint_cycle_" << ti << "." << std::setw(8) << std::setfill('0') << myRank << ".chk";
            std::ofstream checkpoint_save(checkpoint_file.str(), std::ios::binary);
            MFEM_VERIFY(checkpoint_save, "Failed to open checkpoint file for writing: " << checkpoint_file.str());

            sol->Save(checkpoint_save);
            checkpoint_save.close();

            if (Mpi::Root())
                {
                // writing time and cycle data to a json file
                std::string meta_file = cycle_dir + "/checkpoint_cycle_" + std::to_string(ti)+".json";
                std::ofstream meta(meta_file);
                MFEM_VERIFY(meta, "Failed to open meta file for writing: " << meta_file);

                meta << std::fixed << "{" << "\n" << " \"time\": " << t << "," << "\n"
                                    << " \"cycle\": " << ti << "\n"
                                    << "}" << "\n";
                meta.close();             
                }
            MPI_Barrier(pmesh->GetComm());   
            
            next_checkpoint_t += checkpoint_dt;
        }

        if (ti % print_interval == 0)
        {
            if (Mpi::Root())
            {
                std::cout << "time step: " << ti << ", time: " << t << ", dt: " << dt << "\n";
            }
        }
    }

#ifdef AXISYMMETRIC

    auto stats = NS->GetAxisReconStats(true);

    if (Mpi::Root())
    {
        const double denom = (stats.calls > 0) ? (double)stats.calls : 1.0;
        const double highOrder_shape_percentage = 100.0 * (double)stats.highOrder_shape / denom;
        const double lowOrder_ray2_percentage = 100.0 * (double)stats.lowOrder_ray2 / denom;
        const double lowOrder_ray1_percentage = 100.0 * (double)stats.lowOrder_ray1 / denom;
        const double lowOrder_copy_percentage = 100.0 * (double)stats.lowOrder_copy / denom;
        std::cout << "[Axis Reconstruction]" << "\n" 
                  << "High Order [shape] : " << std::round(highOrder_shape_percentage*1000)/1000.0 << "%" << "\n" 
                  << "Low  Order [ray2]  : " << std::round(lowOrder_ray2_percentage*1000)/1000.0 << "%" << "\n" 
                  << "Low  Order [ray1]  : " << std::round(lowOrder_ray1_percentage*1000)/1000.0 << "%" << "\n" 
                  << "Low  Order [copy]  : " << std::round(lowOrder_copy_percentage*1000)/1000.0 << "%" << std::endl;
    }

#endif

    // VectorFunctionCoefficient u_final(num_equations, ConditionFactory::Instance().GetVectorTDFunctionBoundaryCondition1("AcousticWaveExactSolution")(1.4)); 
    // u_final.SetTime(1.0);
    // ParGridFunction u_final_gf(vfes.get());
    // u_final_gf.ProjectCoefficient(u_final);
    // real_t error_L1 = sol->ComputeLpError(1.0, *u0);
    // real_t error_L2 = sol->ComputeLpError(2.0, *u0);
    // real_t error_Linf = sol->ComputeLpError(infinity(), *u0);
    // if (Mpi::Root())
    // {
    //     std::cout << "L1 Error: " << error_L1 << std::endl;
    //     std::cout << "L2 Error: " << error_L2 << std::endl;
    //     std::cout << "Linf Error: " << error_Linf << std::endl;
    // }

    // Stop the clock if enabled
    if (clock_simulation)
    {
        tic_toc.Stop();
        if (Mpi::Root())
        {
            std::cout << "================================================" << std::endl;
            std::cout << "DGSEM Simulation Completed in " << tic_toc.RealTime() << " seconds!" << std::endl;
            std::cout << "================================================" << std::endl;
        }
    }
    else
    {
        if (Mpi::Root())
        {
            std::cout << "================================================" << std::endl;
            std::cout << "DGSEM Simulation Completed!" << std::endl;
            std::cout << "================================================" << std::endl;
        }
    }
}

#ifdef AXISYMMETRIC
void Simulation::ConservativeToPrimitive(const Vector &U_cons,
                                ParGridFunction &rho_out,
                                ParGridFunction &uz_out,
                                ParGridFunction &ur_out,
                                ParGridFunction &p_out) const
{
    for (int i = 0; i < num_dofs_scalar; i++)
    {
        const real_t rho = U_cons[i];
        const real_t mz  = U_cons[i + num_dofs_scalar];
        const real_t mr  = U_cons[i + 2*num_dofs_scalar];
        const real_t E   = U_cons[i + 3*num_dofs_scalar];

        real_t uz = 0.0, ur = 0.0, p = 0.0;

        uz = mz / rho;
        ur = mr / rho;
        const real_t Vsq = uz*uz + ur*ur;
        p = physicsConstants->gammaM1 * (E - 0.5 * rho * Vsq);
    
        rho_out(i) = rho;
        uz_out(i)  = uz;
        ur_out(i)  = ur;
        p_out(i)   = p;
    }
}
#endif


}