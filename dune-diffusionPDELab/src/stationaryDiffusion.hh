//
// Created by justus on 28.11.15.
//

#ifndef DUNE_TEST_DIFFUSION_HH
#define DUNE_TEST_DIFFUSION_HH

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include<iostream>
#include<vector>
#include<map>
#include<memory>
#include<string>
#include<dune/common/parallel/mpihelper.hh>
#include<dune/common/exceptions.hh>
#include<dune/common/fvector.hh>
#include<dune/common/static_assert.hh>
#include<dune/common/timer.hh>
#include<dune/grid/yaspgrid.hh>
#include<dune/istl/bvector.hh>
#include<dune/istl/operators.hh>
#include<dune/istl/solvers.hh>
#include<dune/istl/preconditioners.hh>
#include<dune/istl/io.hh>
#include<dune/istl/paamg/amg.hh>
#include<dune/istl/superlu.hh>
#include<dune/grid/io/file/vtk/subsamplingvtkwriter.hh>

#include<dune/pdelab/finiteelementmap/monomfem.hh>
#include<dune/pdelab/finiteelementmap/opbfem.hh>
#include<dune/pdelab/finiteelementmap/qkdg.hh>
#include<dune/pdelab/finiteelementmap/qkfem.hh>
#include<dune/pdelab/finiteelementmap/pkfem.hh>
#include<dune/pdelab/constraints/conforming.hh>
#include<dune/pdelab/constraints/common/constraints.hh>
#include<dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#include<dune/pdelab/gridfunctionspace/gridfunctionspaceutilities.hh>
#include<dune/pdelab/gridfunctionspace/interpolate.hh>
#include<dune/pdelab/common/function.hh>
#include<dune/pdelab/common/functionutilities.hh>
#include<dune/pdelab/common/vtkexport.hh>
#include<dune/pdelab/backend/istlvectorbackend.hh>
#include<dune/pdelab/backend/istl/bcrsmatrixbackend.hh>
#include<dune/pdelab/backend/istlmatrixbackend.hh>
#include<dune/pdelab/backend/istlsolverbackend.hh>
#include<dune/pdelab/localoperator/convectiondiffusionparameter.hh>
#include<dune/pdelab/localoperator/convectiondiffusiondg.hh>
#include<dune/pdelab/localoperator/convectiondiffusionfem.hh>
#include<dune/pdelab/stationary/linearproblem.hh>
#include<dune/pdelab/gridoperator/gridoperator.hh>

#include<dune/pdelab/gridfunctionspace/vtk.hh>


namespace stationaryDiffusion {
    const bool graphics = true;

    template<typename GV, typename RF>
    class U0Initial
            : public Dune::PDELab::GridFunctionBase<Dune::PDELab::
            GridFunctionTraits<GV, RF, 1, Dune::FieldVector<RF, 1> >, U0Initial<GV, RF> > {
        const GV &gv;

    public:
        typedef Dune::PDELab::GridFunctionTraits<GV, RF, 1, Dune::FieldVector<RF, 1> > Traits;

        //! construct from grid view
        U0Initial(const GV &gv_)
                : gv(gv_) { }

        //! evaluate extended function on element
        inline virtual void evaluate(const typename Traits::ElementType &e,
                                     const typename Traits::DomainType &x,
                                     typename Traits::RangeType &y) const {
            y=10;

            return;
        }

        //! get a reference to the grid view
        inline const GV &getGridView() { return gv; }
    };

    template<class GV>
    class SinkSourceDefault {
    public:
        using Traits = typename Dune::PDELab::ConvectionDiffusionParameterTraits<GV, double>;

        SinkSourceDefault() { }

        virtual ~SinkSourceDefault() { }

        virtual typename Traits::RangeFieldType evaluate(const typename Traits::ElementType &e,
                                                         const typename Traits::DomainType &x) const{
            return 0.0;
        }
    };

    template<typename GV, typename RF>
    class Parameter {
    public:
        typedef Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type BCType;
        typedef Dune::PDELab::ConvectionDiffusionParameterTraits<GV, RF> Traits;

    private:
        SinkSourceDefault<GV> sink_term;
        SinkSourceDefault<GV> source_term;

    public:


        Parameter(SinkSourceDefault<GV> sink_term, SinkSourceDefault<GV> source_term)
                : sink_term(sink_term), source_term(source_term) {

        }

//! tensor diffusion coefficient
        typename Traits::PermTensorType
        A(const typename Traits::ElementType &e, const typename Traits::DomainType &x) const {
            typename Traits::PermTensorType I;
            for (std::size_t i = 0; i < Traits::dimDomain; i++) {
                for (std::size_t j = 0; j < Traits::dimDomain; j++) {
                    if (i == j) {
                        I[i][j] = 1;
                    }
                    I[i][j] = (i == j) ? 1 : 0;
                }
            }
            return I;
        }

        //! velocity field
        typename Traits::RangeType
        b(const typename Traits::ElementType &e, const typename Traits::DomainType &x) const {
            typename Traits::RangeType v(0.0);
            return v;
        }


        //! sink term
        typename Traits::RangeFieldType
        c(const typename Traits::ElementType &e, const typename Traits::DomainType &x) const {
            return sink_term.evaluate(e, x);
        }

        //! source term
        typename Traits::RangeFieldType
        f(const typename Traits::ElementType &e, const typename Traits::DomainType &x) const {

            return source_term.evaluate(e, x);
        }

        //! boundary condition type function
        BCType
        bctype(const typename Traits::IntersectionType &is, const typename Traits::IntersectionDomainType &x) const {
            return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
        }

        //! Dirichlet boundary condition value
        typename Traits::RangeFieldType
        g(const typename Traits::ElementType &e, const typename Traits::DomainType &x) const {
            typename Traits::DomainType xglobal = e.geometry().global(x);
            typename Traits::RangeFieldType norm = xglobal.two_norm2();
            return exp(-norm);
        }

        //! Neumann boundary condition
        typename Traits::RangeFieldType
        j(const typename Traits::IntersectionType &is, const typename Traits::IntersectionDomainType &x) const {
            return 0.0;
        }

        //! outflow boundary condition
        typename Traits::RangeFieldType
        o(const typename Traits::IntersectionType &is, const typename Traits::IntersectionDomainType &x) const {
            return 0.0;
        }
    };

/*! \brief Adapter returning ||f1(x)-f2(x)||^2 for two given grid functions

  \tparam T1  a grid function type
  \tparam T2  a grid function type
*/
    template<typename T1, typename T2>
    class DifferenceSquaredAdapter
            : public Dune::PDELab::GridFunctionBase<
                    Dune::PDELab::GridFunctionTraits<typename T1::Traits::GridViewType,
                            typename T1::Traits::RangeFieldType,
                            1, Dune::FieldVector<typename T1::Traits::RangeFieldType, 1> >, DifferenceSquaredAdapter<T1, T2> > {
    public:
        typedef Dune::PDELab::GridFunctionTraits<typename T1::Traits::GridViewType,
                typename T1::Traits::RangeFieldType,
                1, Dune::FieldVector<typename T1::Traits::RangeFieldType, 1> > Traits;

        //! constructor
        DifferenceSquaredAdapter(const T1 &t1_, const T2 &t2_) : t1(t1_), t2(t2_) { }

        //! \copydoc GridFunctionBase::evaluate()
        inline void evaluate(const typename Traits::ElementType &e,
                             const typename Traits::DomainType &x,
                             typename Traits::RangeType &y) const {
            typename T1::Traits::RangeType y1;
            t1.evaluate(e, x, y1);
            typename T2::Traits::RangeType y2;
            t2.evaluate(e, x, y2);
            y1 -= y2;
            y = y1.two_norm2();
        }

        inline const typename Traits::GridViewType &getGridView() const {
            return t1.getGridView();
        }

    private:
        const T1 &t1;
        const T2 &t2;
    };

//! solve problem with DG method
    template<class GV, class FEM, class PROBLEM, int degree, int blocksize>
    void runDG(const GV &gv,
               const FEM &fem,
               PROBLEM &problem,
               std::string basename,
               int level,
               std::string method,
               std::string weights,
               double alpha,
               U0Initial<GV, double> &u0)
    {
        // coordinate and result type
        typedef double Real;
        const int dim = GV::Grid::dimension;

        std::stringstream fullname;
        fullname << basename << "_" << method << "_w" << weights << "_k" << degree << "_dim" << dim << "_level" <<
        level;

        // make grid function space
        typedef Dune::PDELab::NoConstraints CON;
        typedef Dune::PDELab::ISTLVectorBackend<Dune::PDELab::ISTLParameters::static_blocking, blocksize> VBE;
        typedef Dune::PDELab::GridFunctionSpace<GV, FEM, CON, VBE> GFS;
        GFS gfs(gv, fem);

        // make local operator
        Dune::PDELab::ConvectionDiffusionDGMethod::Type m;
        if (method == "SIPG")
            m = Dune::PDELab::ConvectionDiffusionDGMethod::SIPG;
        else if (method == "NIPG")
            m = Dune::PDELab::ConvectionDiffusionDGMethod::NIPG;

        Dune::PDELab::ConvectionDiffusionDGWeights::Type w;
        if (weights == "ON")
            w = Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn;
        else if (weights == "OFF")
            w = Dune::PDELab::ConvectionDiffusionDGWeights::weightsOff;

        typedef Dune::PDELab::ConvectionDiffusionDG<PROBLEM, FEM> LOP;
        LOP lop(problem, m, w, alpha);
        typedef Dune::PDELab::istl::BCRSMatrixBackend<> MBE;
        MBE mbe(5); // Maximal number of nonzeroes per row can be cross-checked by patternStatistics().
        typedef typename GFS::template ConstraintsContainer<Real>::Type CC;
        CC cc;
        Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<PROBLEM> bctype(gv, problem);

        typedef Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, Real, Real, Real, CC, CC> GO;
        GO go(gfs, cc, gfs, cc, lop, mbe);

        // make a vector of degree of freedom vectors and initialize it with Dirichlet extension
        typedef typename GO::Traits::Domain U;
        U u(gfs, 0.0);
        typedef Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter<PROBLEM> G;
        G g(gv, problem);

        //TODO: fix initial values
        //Dune::PDELab::interpolate(u0, gfs, u);

        //initialize constraints container
        Dune::PDELab::constraints(bctype, gfs, cc);
        Dune::PDELab::set_nonconstrained_dofs(cc, 0.0, u);


        // make linear solver and solve problem
        if (method == "SIPG") {
            typedef Dune::PDELab::ISTLBackend_SEQ_CG_ILU0 LS;
            LS ls(10000, 1);
            typedef Dune::PDELab::StationaryLinearProblemSolver<GO, LS, U> SLP;
            SLP slp(go, ls, u, 1e-12);
            slp.apply();
        }
        else {
            typedef Dune::PDELab::ISTLBackend_SEQ_BCGS_ILU0 LS;
            LS ls(10000, 1);
            typedef Dune::PDELab::StationaryLinearProblemSolver<GO, LS, U> SLP;
            SLP slp(go, ls, u, 1e-12);
            slp.apply();
        }

        // compute L2 error
        typedef Dune::PDELab::DiscreteGridFunction<GFS, U> UDGF;
        UDGF udgf(gfs, u);
        typedef DifferenceSquaredAdapter<G, UDGF> DifferenceSquared;
        DifferenceSquared differencesquared(g, udgf);
        typename DifferenceSquared::Traits::RangeType l2errorsquared(0.0);
        Dune::PDELab::integrateGridFunction(differencesquared, l2errorsquared, 12);
        std::cout << fullname.str()
        << " N=" << std::setw(11) << gfs.globalSize()
        << " L2ERROR=" << std::setw(11) << std::setprecision(3) << std::scientific << std::uppercase <<
        sqrt(l2errorsquared[0]) << std::endl;

        // write vtk file
        if (graphics) {
            Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, degree - 1);

            vtkwriter.addVertexData(std::shared_ptr<Dune::PDELab::VTKGridFunctionAdapter<UDGF>>(new Dune::PDELab::VTKGridFunctionAdapter<UDGF>(udgf,"u_h")));
            vtkwriter.addVertexData(std::shared_ptr<Dune::PDELab::VTKGridFunctionAdapter<G>>(new Dune::PDELab::VTKGridFunctionAdapter<G>(g,"u")));

            vtkwriter.write(fullname.str(), Dune::VTK::appendedraw);
        }
    }


//! solve problem with DG method
    template<class GV, class FEM, class PROBLEM, int degree>
    void runFEM(const GV &gv, const FEM &fem, PROBLEM &problem, std::string basename, int level,
                U0Initial<GV, double> &u0) {
        // coordinate and result type
        typedef double Real;
        const int dim = GV::Grid::dimension;
        std::stringstream fullname;
        fullname << basename << "_FEM" << "_k" << degree << "_dim" << dim << "_level" << level;

        // make grid function space
        typedef Dune::PDELab::ISTLVectorBackend<> VBE;
        typedef Dune::PDELab::ConformingDirichletConstraints CON;
        typedef Dune::PDELab::GridFunctionSpace<GV, FEM, CON, VBE> GFS;
        GFS gfs(gv, fem);

        // make constraints container
        typedef typename GFS::template ConstraintsContainer<Real>::Type CC;
        CC cc;
        Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<PROBLEM> bctype(gv, problem);

        // make local operator
        typedef Dune::PDELab::ConvectionDiffusionFEM<PROBLEM, FEM> LOP;
        LOP lop(problem);
        typedef Dune::PDELab::istl::BCRSMatrixBackend<> MBE;
        MBE mbe(5); // Maximal number of nonzeroes per row can be cross-checked by patternStatistics().
        typedef Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, Real, Real, Real, CC, CC> GO;
        GO go(gfs, cc, gfs, cc, lop, mbe);

        // make a vector of degree of freedom vectors and initialize it with Dirichlet extension
        typedef typename GO::Traits::Domain U;
        U u(gfs, 0.0);
        typedef Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter<PROBLEM> G;
        G g(gv, problem);

        Dune::PDELab::interpolate(g, gfs, u);

        //initialize constraints container
        Dune::PDELab::constraints(bctype, gfs, cc);
        Dune::PDELab::set_nonconstrained_dofs(cc, 0.0, u);

        // make linear solver and solve problem
        typedef Dune::PDELab::ISTLBackend_SEQ_CG_ILU0 LS;
        LS ls(10000, 1);
        typedef Dune::PDELab::StationaryLinearProblemSolver<GO, LS, U> SLP;
        SLP slp(go, ls, u, 1e-12);
        slp.apply();

        // compute L2 error
        typedef Dune::PDELab::DiscreteGridFunction<GFS, U> UDGF;
        UDGF udgf(gfs, u);
        typedef DifferenceSquaredAdapter<G, UDGF> DifferenceSquared;
        DifferenceSquared differencesquared(g, udgf);
        typename DifferenceSquared::Traits::RangeType l2errorsquared(0.0);
        Dune::PDELab::integrateGridFunction(differencesquared, l2errorsquared, 12);
        std::cout << fullname.str()
        << " N=" << std::setw(11) << gfs.globalSize()
        << " L2ERROR=" << std::setw(11) << std::setprecision(3) << std::scientific << std::uppercase <<
        sqrt(l2errorsquared[0]) << std::endl;

        // write vtk file
        if (graphics) {
            Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, degree - 1);

            vtkwriter.addVertexData(std::shared_ptr<Dune::PDELab::VTKGridFunctionAdapter<UDGF>>(new Dune::PDELab::VTKGridFunctionAdapter<UDGF>(udgf,"u_h")));
            vtkwriter.addVertexData(std::shared_ptr<Dune::PDELab::VTKGridFunctionAdapter<G>>(new Dune::PDELab::VTKGridFunctionAdapter<G>(g,"u")));

            vtkwriter.write(fullname.str(), Dune::VTK::appendedraw);


        }
    }



//======================================================================================
//======================================================================================
// Solve the convection-diffusion equation
//--------------------------------------------------------------------------------------
//      \nabla\cdot(-A(x) \nabla u + b(x) u) + c(x)u = f \mbox{ in } \Omega, \\
//      u = g \mbox{ on } \partial\Omega_D \\
//      (b(x,u) - A(x)\nabla u) \cdot n = j \mbox{ on } \partial\Omega_N \\
//      -(A(x)\nabla u) \cdot n = j \mbox{ on } \partial\Omega_O
//======================================================================================
//======================================================================================

    template<const int dim, const int degree>
    int solveDiffusionPDE(int maxlevel, std::string &method, Dune::YaspGrid<dim> &grid,
                          U0Initial<typename Dune::YaspGrid<dim>::LeafGridView, double> &u0,
                          SinkSourceDefault<typename Dune::YaspGrid<dim>::LeafGridView> &sink_term,
                          SinkSourceDefault<typename Dune::YaspGrid<dim>::LeafGridView> &source_term) {

        try {
            using Grid = Dune::YaspGrid<dim>;
            using GV = typename Grid::LeafGridView;
            using PROBLEM = Parameter<GV, double>;

            const GV &gv = grid.leafGridView();
            PROBLEM problem(sink_term, source_term);


            if (method == "SIPG") {

                using FEMDG = Dune::PDELab::QkDGLocalFiniteElementMap<typename Grid::ctype, double, degree, dim>;

                for (int i = 0; i <= maxlevel; ++i) {

                    FEMDG femdg;
                    const int blocksize = Dune::QkStuff::QkSize<degree, dim>::value;
                    runDG<GV, FEMDG, PROBLEM, degree, blocksize>(gv, femdg, problem, "CUBE", i, "SIPG", "ON", 2.0, u0);

                    // refine grid
                    if (i < maxlevel) grid.globalRefine(1);
                }
            }
            else if (method == "FEM") {

                using FEMCG = Dune::PDELab::QkLocalFiniteElementMap<GV, typename Grid::ctype, double, degree>;

                for (int i = 0; i <= maxlevel; ++i) {

                    FEMCG femcg(gv);

                    runFEM<GV, FEMCG, PROBLEM, degree>(gv, femcg, problem, "CUBE", i, u0);

                    // refine grid
                    if (i < maxlevel) grid.globalRefine(1);
                }
            }
            return 0;
        }
        catch (Dune::Exception &e) {
            std::cerr << "Dune reported error: " << e << std::endl;
            return 1;
        }
        catch (std::exception &e) {
            std::cerr << "STL reported error: " << e.what() << std::endl;
            return 1;
        }
        catch (std::string &e) {
            std::cerr << "An error has been detected: " << e << std::endl;
            return 1;
        }
        catch (...) {
            std::cerr << "Unknown exception thrown!" << std::endl;
            return 1;
        }
    }
}

#endif //DUNE_TEST_DIFFUSION_HH