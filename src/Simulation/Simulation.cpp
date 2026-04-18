#include "Simulation.hpp"
#include "ConditionFactory.hpp"

#include "LidDrivenCavity.hpp"
#include "TaylorGreenVortex.hpp"
#include "TaylorGreenVortex2D.hpp"

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

  Simulation& Simulation::SimulationCreate(std::string device_cfg)
{
  static Simulation sim(device_cfg);
  return sim;
}

  void Simulation::InitDevice(std::string device_cfg)
  {
    if(device_cfg.empty()){
      device_cfg = "cpu";
    }
    if(!device_){
      device_ = std::make_unique<mfem::Device>(device_cfg);
      if(myRank == 0){ device_->Print(); }
    }
  }

  Simulation::Simulation(std::string device_cfg)
    : r_coef([](const Vector &X){ return X[1];}), z_coef([](const Vector &X){ return X[0];})
{
    Mpi::Init();
    numProcs = Mpi::WorldSize();
    myRank = Mpi::WorldRank();
    Hypre::Init();
    InitDevice(device_cfg);

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

constexpr bool debug_simulation = true;

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
        nancheck_steps = runtime.value("nancheck_steps", 1);
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
    std::string mesh_file_name(runtime["mesh_file"].get<std::string>());
    {
      ScopedTimer timer("ReadMesh");
      mesh = new Mesh(runtime["mesh_file"].get<std::string>());
    }
 
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
    if (!periodic && mesh->GetNBE() == 0){
      mesh->GenerateBoundaryElements();
    }
    if (runtime.contains("ser_ref_levels"))
      {
        ref_levels = runtime.value("ser_ref_levels", 0);
        for (int lev = 0; lev < ref_levels; lev++)
        {
            mesh->UniformRefinement();
        }
    }
    mesh->FinalizeTopology();

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

    // TODO: Let's gate this for now
    if (dim > 1 && runtime.value("use_nc_mesh", true))
      {
        mesh->EnsureNCMesh();
      }

    // Completely finalize the mesh
    mesh->FinalizeMesh(0, true);
    pmesh = std::make_shared<ParMesh>(MPI_COMM_WORLD, *mesh);
    mesh->Clear();
    if(myRank == 0 && debug_simulation){
      std::cout << "Mesh distributed" << std::endl;
    }
    if (runtime.contains("par_ref_levels"))
    {
        ref_levels = runtime.value("par_ref_levels", 0);
        for (int lev = 0; lev < ref_levels; lev++)
        {
            pmesh->UniformRefinement();
        }
    }

    pmesh->ExchangeFaceNbrData();
    fec0 = std::make_shared<DG_FECollection>(0, dim);
    fes0 = std::make_shared<ParFiniteElementSpace>(pmesh.get(), fec0.get());

    fec = std::make_shared<DG_FECollection>(order, dim, btype);
    vfes = std::make_shared<ParFiniteElementSpace>(pmesh.get(), fec.get(), num_equations, ordering);
    dfes = std::make_unique<ParFiniteElementSpace>(pmesh.get(), fec.get(), dim, ordering);
    fes = std::make_unique<ParFiniteElementSpace>(pmesh.get(), fec.get());

    // Let's do an initial exchange to get the data structures populated
    vfes->ExchangeFaceNbrData();
    fes0->ExchangeFaceNbrData();
    dfes->ExchangeFaceNbrData();
    fes->ExchangeFaceNbrData();

    num_dofs_scalar = fes->GetNDofs();
    num_dofs_system = vfes->GetVSize();

    if(myRank == 0 && debug_simulation){
      std::cout << "Initial exchanges complete." << std::endl;
    }

    stateLayout = std::make_shared<StateLayout>(dim, num_dofs_scalar);
    gasModel = std::make_shared<IdealGasModel>(*physicsConstants, *stateLayout);

    flux = std::make_shared<NavierStokesFlux>(*gasModel);
    if (runtime["numerical_flux"].get<std::string>() == "Chandrashekar"){
      numericalFlux = std::make_shared<ChandrashekarFlux>(*flux, *gasModel);
    } else {
      std::cerr << "Error: Invalid numerical flux specified." << std::endl;
      return;
    }

    if(myRank == 0 && debug_simulation){
      std::cout << "StateLayout and GasModel created." << std::endl;
    }
 
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

        if(myRank == 0 && debug_simulation){
          std::cout << "Checkpoint loaded @ (" << ti << ", " << t << ")"
                    << std::endl;
        }
    }
    else 
    {
        sol = std::make_shared<ParGridFunction>(vfes.get());
        sol->ProjectCoefficient(*u0);

        t = 0.0;
        ti = 0;

        if(myRank == 0 && debug_simulation){
          std::cout << "Run is not a restart: (step=0, t=0)" << std::endl;
        }
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
        Prandtl::FieldStateView fields{sol_state};
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
          real_t rho = fields.mass(*stateLayout, i);
          real_t rhoU = fields.momentum_x(*stateLayout, i);
          real_t rhoV = fields.momentum_y(*stateLayout, i);
          real_t E = fields.energy(*stateLayout, i);
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

    std::vector<std::shared_ptr<ParGridFunction> > grad_u(dim);
    for(int idim = 0;idim < dim;idim++)
      grad_u[idim] = std::make_shared<ParGridFunction>(vfes.get());

    Geometry::Type gtype = vfes->GetFE(0)->GetGeomType();

    if (runtime.contains("alpha_max"))
    {
        alpha_max = runtime["alpha_max"].get<real_t>();
    }
    else
    {
        alpha_max = 0.5;
    }

    auto integrator =
      std::make_unique<Prandtl::DGSEMIntegrator>(pmesh, fes0, alpha, liftingScheme, *numericalFlux, order+1);

    auto indicator =
      std::make_unique<Prandtl::PerssonPeraireIndicator>(vfes, fes0, eta,
                                                         std::make_unique<Prandtl::ModalBasis>(*fec, gtype, order, dim),
                                                         *gasModel);

    NS = std::make_unique<DGSEMOperator>(vfes, fes0, pmesh, eta, alpha, grad_u, std::move(integrator),
                                         std::move(indicator), *gasModel, r_gf, alpha_max);

    if(myRank == 0 && debug_simulation){
      std::cout << "DGSEMOperator created." << std::endl;
    }

    mfem::Vector bc_vector_data;
    mfem::Vector bc_scalar_data;
    mfem::Array<Prandtl::BCDescriptor> bc_descriptors;

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

            Prandtl::BCDescriptor bc_descr{};
            bc_descr.flags = 0;
            bc_descr.data_index = -1;
            bc_descr.bdr_attr = -1;
            bc_descr.data_kind = int(Prandtl::BCDataKind::None);
            bc_descr.type = int(Prandtl::BCType::Invalid);

            auto bc_props = boundary.value();  // This is a JSON object.
            std::string type = bc_props["type"].get<std::string>();
            
            if (type == "symmetry")
              {
                auto symmetry = std::make_unique<SymmetryBdrFaceIntegrator>(liftingScheme, *gasModel, *numericalFlux,
                                                                            order + 1, NS->GetTimeRef(), true, false);
                
                NS->AddBdrFaceIntegrator(symmetry.release(), bdr_marker_vector.back());
                bc_descr.type = int(Prandtl::BCType::Symmetry);
                bc_descr.data_kind = int(Prandtl::BCDataKind::None);
              }
            else if (type == "axis")
              {
#ifndef AXISYMMETRIC
                MFEM_ABORT("AXIS BC requires axisymmetry build.");
#else
                // bc_descr.type = int(Prandtl::BCType::Axis);
                auto symmetry = std::make_unique<SymmetryBdrFaceIntegrator>(liftingScheme, *gasModel, *numericalFlux,
                                                                            order + 1, NS->GetTimeRef(), true, false);
                
                NS->AddBdrFaceIntegrator(symmetry.release(), bdr_marker_vector.back());
                NS->SetAxisBoundaryMarker(bdr_marker_vector.back());
                NS->SetLowOrderAxis(true);
#endif
              }
            else if (type == "slip")
              {
                auto slip = std::make_unique<SlipWallBdrFaceIntegrator>(liftingScheme, *gasModel, *numericalFlux,
                                                                        order + 1, NS->GetTimeRef(), true, false);
                
                NS->AddBdrFaceIntegrator(slip.release(), bdr_marker_vector.back());
                bc_descr.type = int(Prandtl::BCType::SlipWall);
                bc_descr.data_kind = int(Prandtl::BCDataKind::None);
              }
            else if (type == "no-slip-adiabatic")
              {
                // if(debug_simulation && Mpi::Root()){
                //  std::cout << "Detected noslip boundary... configuring." << std::endl;
                // }
                bc_descr.type = int(Prandtl::BCType::NoSlipAdiab);
                bc_descr.data_kind = int(Prandtl::BCDataKind::VectorAndScalarConstant);
                if (bc_props["velocity"].contains("vector"))
                  {
                    std::string velBC_key = bc_props["velocity"]["vector"].get<std::string>();
                    std::string heatBC_key = bc_props["heat"]["scalar"].get<std::string>();
                    // std::string state_key = bc_props["vector"].get<std::string>();
                    auto vel_bc = ConditionFactory::Instance().GetVectorBoundaryCondition(velBC_key);
                    auto heat_bc = ConditionFactory::Instance().GetScalarBoundaryCondition(heatBC_key);

                    mfem::Vector bc_data(vel_bc.Size() + 1);
                    std::ostringstream Ostr;
                    Ostr << "Wall velocity: < ";
                    for(int ivec = 0;ivec < vel_bc.Size();ivec++){
                      bc_data[ivec] = vel_bc[ivec];
                      Ostr << vel_bc[ivec] << " ";
                    }
                    Ostr << ">" << std::endl;
                    Ostr << "Heat: " << heat_bc << std::endl;
                    bc_data[vel_bc.Size()] = heat_bc;
                    bc_descr.data_index = Prandtl::AppendBCVectorPayload(bc_vector_data, bc_data);                    
                    Ostr << "bc_vector_data index: " << bc_descr.data_index << std::endl
                         << "BC Data So Far: [";
                    for(int ivec=0;ivec < bc_vector_data.Size();ivec++){
                      Ostr << bc_vector_data[ivec] << " ";
                    }
                    Ostr << "]" << std::endl;
                    // std::cout << Ostr.str();
                    NS->AddBdrFaceIntegrator(
                                             new NoSlipAdiabWallBdrFaceIntegrator(liftingScheme, *gasModel, *numericalFlux, order + 1, NS->GetTimeRef(),
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
                                             new NoSlipAdiabWallBdrFaceIntegrator(liftingScheme, *gasModel,
                                                                                  *numericalFlux, order + 1, NS->GetTimeRef(),
                                                                                  *heatBC, *velBC, td),
                                             bdr_marker_vector.back());
                    
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
              auto outlet = std::make_unique<SupersonicOutflowBdrFaceIntegrator>(liftingScheme, *gasModel,
                                                                                 *numericalFlux, order + 1, NS->GetTimeRef());

                NS->AddBdrFaceIntegrator(outlet.release(), bdr_marker_vector.back());
                bc_descr.type = int(Prandtl::BCType::SupersonicOutflow);
                bc_descr.data_kind = int(Prandtl::BCDataKind::None);
            }
            else if (type == "supersonic-inflow")
            {
                if (bc_props.contains("vector"))
                {
                std::string state_key = bc_props["vector"].get<std::string>();
                mfem::Vector bc_state = ConditionFactory::Instance().GetVectorBoundaryCondition(state_key);
                auto inlet = std::make_unique<SupersonicInflowBdrFaceIntegrator>(
                                                              liftingScheme, *gasModel, 
                                                              *numericalFlux, order + 1, NS->GetTimeRef(), bc_state);
                NS->AddBdrFaceIntegrator(inlet.release(), bdr_marker_vector.back());
                const int data_offset = Prandtl::AppendBCVectorPayload(bc_vector_data, bc_state);
                bc_descr.type = int(Prandtl::BCType::SupersonicInflow);
                bc_descr.data_kind = int(Prandtl::BCDataKind::VectorConstant);
                bc_descr.data_index = data_offset;
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
                    auto inlet = std::make_unique<SupersonicInflowBdrFaceIntegrator>(liftingScheme, *gasModel, *numericalFlux,
                                                                                     order + 1, NS->GetTimeRef(), *stateBC, td);

                    NS->AddBdrFaceIntegrator(inlet.release(), bdr_marker_vector.back());

                }
            }
            else if (type == "specified-state")
            {
                if (bc_props.contains("vector"))
                {
                    std::string state_key = bc_props["vector"].get<std::string>();
                    NS->AddBdrFaceIntegrator(new SpecifiedStateBdrFaceIntegrator(liftingScheme, *gasModel, *numericalFlux,
                                                                                 order + 1, NS->GetTimeRef(),
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
                    NS->AddBdrFaceIntegrator(new SpecifiedStateBdrFaceIntegrator(liftingScheme, *gasModel, *numericalFlux, order + 1, NS->GetTimeRef(),
                                                                                 *stateBC, td), bdr_marker_vector.back());
                }
            }
            else
            {
                std::cerr << "Error: Invalid boundary condition type specified." << std::endl;
                return;
            }
            bc_descriptors.Append(bc_descr);
        }
    }

    if(myRank == 0 && debug_simulation){
      std::cout << "Boundary conditions configured." << std::endl;
    }

    // Set up the operator cache
    NS->SetBCDescriptorData(bc_descriptors, bc_scalar_data, bc_vector_data);
    //    NS->SetTime(t); // Finalize does this.
    NS->Finalize(t);

    if(myRank == 0 && debug_simulation){
      std::cout << "DGSEMOperator finalized." << std::endl;
    }

    if (Mpi::Root())
    {
        std::cout << "The Number of Degrees of Freedom per Conservative Variable per Rank: " << num_dofs_scalar << std::endl;
        std::cout << "The Number of Degrees of Freedom (System) per Rank: " << num_dofs_system << std::endl;
    }

    ode_solver->Init(*NS);

    rho.MakeRef(fes.get(), *sol, offset_mass(*stateLayout));
    mom.MakeRef(dfes.get(), *sol, offset_momentum(*stateLayout));
    energy.MakeRef(fes.get(), *sol, offset_energy(*stateLayout));

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
#ifdef SUBCELL_FV_BLENDING
            pd->RegisterField("Blending Coeff", alpha.get());
#endif
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
#ifdef SUBCELL_FV_BLENDING
            vd->RegisterField("Blending Coeff", alpha.get());
#endif
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

    IntegralMeasures diag;
    diag.mass = 0.0;
    diag.ke = 0.0;
    diag.en = 0.0;
    diag.max_press = 0.0;
    diag.min_press = 0.0;
    diag.max_dens = 0.0;
    diag.min_dens = 0.0;
    diag.max_temp = 0.0;
    diag.min_temp = 0.0;

    // print the first node:
#ifdef AXISYMMETRIC
    Vector U_cons(sol->Size());
    NS->RecoverStateFromWeighted(*sol, U_cons);
    ConservativeToPrimitive(U_cons, *rho_axi, *u, *v, *p);
    NS->ComputeIntegralMeasures(U_cons, diag);
#else
    NS->ComputeIntegralMeasures(*sol, diag);
#endif

    IntegralMeasures diag0 = NS->GetIntegralMeasuresBaseline();

    // Get the minimum characteristic element size and compute the initial time step if time step is variable
    real_t heff = infinity();
    hmin = infinity();
    for (int i = 0; i < pmesh->GetNE(); i++)
      {
        hmin = std::min(pmesh->GetElementSize(i, 1), hmin);
      }
    MPI_Allreduce(MPI_IN_PLACE, &hmin, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh->GetComm());
    heff = hmin / ((order+1)*(order+1));
    if (debug_simulation && Mpi::Root()){
      std::cout << "Mesh h_min: " << heff << std::endl;
    }
    if (variable_dt && cfl > 0.0)
    {        
        Vector z(sol->Size());
        NS->Mult(*sol, z);
        real_t max_char_speed = NS->GetMaxCharSpeed();
        MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1,  MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh->GetComm());
        real_t dt_adv = heff / max_char_speed;
#ifdef PARABOLIC
        real_t nu_eff = \
          std::max(1.0, physicsConstants->gamma/physicsConstants->Pr) * physicsConstants->mu / diag0.min_dens;
        real_t dt_diff = heff * heff / nu_eff;
        real_t dt_m1 = 1.0 / (1.0/dt_adv + 1.0/dt_diff);
        dt = cfl / dim / (1.0/dt_adv + 1.0/dt_diff);
#else
        dt = cfl / dim * dt_adv;
#endif
        if(Mpi::Root()){
          std::cout << "Fixed CFL: " << cfl << std::endl
                    << "Initial Timestep DT: " << dt << std::endl;
        }
    } else {
      if(Mpi::Root()){
        std::cout << "Fixed Timestep DT: " << dt << std::endl;
      }
    }

    // Clock the simulation?
    if (clock_simulation)
    {
        tic_toc.Clear();
        tic_toc.Start();
    }

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
          real_t *sol_state = sol->GetData();
          Prandtl::DofStateView dofState{sol_state, i};
          real_t rhoi = gasModel->density(dofState);
          real_t ui = gasModel->velocity(dofState, 0);
          real_t vi = dim > 1 ? gasModel->velocity(dofState, 1) : 0.0;
          real_t wi = dim > 2 ? gasModel->velocity(dofState, 2) : 0.0;
          real_t pi = gasModel->pressure(dofState);
          // bug: incorrect calculation of kinetic energy
          //real_t pi = physicsConstants->gammaM1 * (energy(i) - 0.5*rhoi*ui*ui);
#endif
          std::cout << "***  initial at dof #" << i << ":  "
                    << "rho = " << std::round(rhoi*10000)/10000 << ",  velocity = <"
                    << std::round(ui*100)/100;
          if(dim > 1){
            std::cout << ", " << std::round(vi*100)/100;
            if(dim > 2){
              std::cout << ", " << std::round(wi*100)/100;
            }
          }
          std::cout << ">, p = " << std::round(pi*100)/100 << std::endl;
          std::cout << "Total Mass:           " << diag0.mass << std::endl 
                    << "Total Energy:         " << diag0.en << std::endl
                    << "Total Kinetic Energy: " << diag0.ke << std::endl;
        }
    // Visualize the initial condition?
    if (visualize)
    {
      if(Mpi::Root()){
        std::cout << "Writing initial soln..." << std::endl;
      }
      ScopedTimer timer("VisInit");

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
        const real_t *sol_state = sol->HostRead();
        for (int i = 0; i < num_dofs_scalar; i++)
          {
            Prandtl::DofStateView dofState{sol_state, i};
            (*u)(i) = gasModel->velocity(dofState, 0);
            if (dim > 1)
            {
              (*v)(i) = gasModel->velocity(dofState, 1);
              if (dim > 2)
                {
                  (*w)(i) = gasModel->velocity(dofState, 2);
                }
            }
            (*p)(i) = gasModel->pressure(dofState);
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
          MPI_Barrier(pmesh->GetComm());
          if(Mpi::Root()){
            std::cout << "################################################################################" << "\n";
            std::cout << "######################### [TIME STEP = " << ti << ", TIME = " << t << "] #########################" << "\n";
            std::cout << "################################################################################" << "\n";
          }
        }

        // Compute the time step size
        dt_real = std::min(dt, t_final - t);

        // Perform the time step
        ode_solver->Step(*sol, t, dt_real);
        ti++;

        real_t cfl_rep = 0.0;
        if (ti % print_interval == 0 || (variable_dt && cfl > 0) || debug_simulation){
#ifdef AXISYMMETRIC
          Vector U_cons(sol->Size());
          NS->RecoverStateFromWeighted(*sol, U_cons);
          NS->ComputeIntegralMeasures(U_cons, diag);
#else
          NS->ComputeIntegralMeasures(*sol, diag);
#endif
        }
        // Update the time step size with CFL?
        if ((variable_dt && cfl > 0) || (ti%print_interval == 0) || debug_simulation)
        {
          real_t max_char_speed = NS->GetMaxCharSpeed();
          MPI_Allreduce(MPI_IN_PLACE, &max_char_speed, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh->GetComm());
          real_t dt_adv = heff / max_char_speed;
#ifdef PARABOLIC
          real_t nu_eff = \
            std::max(1.0, physicsConstants->gamma/physicsConstants->Pr) * physicsConstants->mu / diag.min_dens;
          real_t dt_diff = heff * heff / nu_eff;
          real_t dt_m1 = 1.0 / (1.0/dt_adv + 1.0/dt_diff);
          if(debug_simulation && Mpi::Root()){
            std::cout << "DT(adv,diff): (" << dt_adv << ", " << dt_diff << ")" << std::endl;
            std::cout << "Max specific volume: " << 1.0 / diag.min_dens << std::endl;
            std::cout << "Effective viscosity: " << nu_eff << std::endl;
            std::cout << "Max wavespeed: " << max_char_speed << std::endl;
          }
          if(variable_dt){
            dt = cfl / dim * dt_m1;
          } else {
            cfl_rep = dim * dt / dt_m1;
          }
#else
          if(variable_dt){
            dt = cfl / dim * dt_adv;
          } else {
            cfl_rep = dt * dim  / dt_adv;
          }
#endif
        }

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
        if (visualize && (done || t >= next_save_t || ti % vis_steps == 0))
        {

#ifdef AXISYMMETRIC
        Vector U_cons(sol->Size());
        NS->RecoverStateFromWeighted(*sol, U_cons);
        ConservativeToPrimitive(U_cons, *rho_axi, *u, *v, *p);
#else
        const real_t *sol_state = sol->HostRead();
        for (int i = 0; i < num_dofs_scalar; i++)
        {       
          Prandtl::DofStateView dofState{sol_state, i};
          (*u)(i) = gasModel->velocity(dofState, 0);
          if (dim > 1)
            {
              (*v)(i) = gasModel->velocity(dofState, 1);
              if (dim > 2)
                {
                  (*w)(i) = gasModel->velocity(dofState, 2);
                }
            }
          (*p)(i) = gasModel->pressure(dofState);
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

        if (ti % print_interval == 0 || debug_simulation)
        {
          real_t ke0 = diag0.ke;
          if(ke0 == 0.0){ ke0 = 1.0; };
            if (Mpi::Root())
            {
              std::ostringstream Ostr;
              Ostr << "time step: " << ti << ", time: " << t;
              if(variable_dt){
                Ostr << ", dt: " << dt;
              } else {
                Ostr << ", cfl: " << cfl_rep;
              }
              Ostr << std::endl
                   << "rho(" << diag.min_dens << "," << diag.max_dens << "), "
                   << "p(" << diag.min_press << "," << diag.max_press << "), "
                   << "T(" << diag.min_temp << "," << diag.max_temp << ")" << std::endl
                   << "TotalChange: Mass: " << (diag.mass - diag0.mass) / diag0.mass
                   << ", Energy: " <<  (diag.en - diag0.en) / diag0.en
                   << ", K.E.: " << (diag.ke - diag0.ke) / ke0 << std::endl;
              std::cout << Ostr.str();
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
