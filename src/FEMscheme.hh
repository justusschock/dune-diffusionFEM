#ifndef DUNE_DIFFUSIONFEM_FEMSCHEME_HH
#define DUNE_DIFFUSIONFEM_FEMSCHEME_HH

// iostream includes
#include <iostream>

// include discrete function space
#include <dune/fem/space/discontinuousgalerkin.hh>

// adaptation ...
#include <dune/fem/function/adaptivefunction.hh>
#include <dune/fem/space/common/adaptmanager.hh>

// include discrete function
#include <dune/fem/function/blockvectorfunction.hh>

// include linear operators
#include <dune/fem/operator/linear/spoperator.hh>
#include <dune/fem/solver/diagonalpreconditioner.hh>

#include <dune/fem/operator/linear/istloperator.hh>
#include <dune/fem/solver/istlsolver.hh>
#include <dune/fem/solver/cginverseoperator.hh>

/*********************************************************/

// include norms
#include <dune/fem/misc/l2norm.hh>
#include <dune/fem/misc/h1norm.hh>

// include parameter handling
#include <dune/fem/io/parameter.hh>

// local includes
#include "problemInterface.hh"

#include "nonlinearModel.hh"

#include "rhs.hh"

#include "ellipticOperator.hh"

// ISTL is only working of LinearOperators with matrix representation
#if HAVE_DUNE_ISTL && WANT_ISTL
#define USE_ISTL 1
#endif

// DataOutputParameters
// --------------------

struct DataOutputParameters
        : public Dune::Fem::LocalParameter< Dune::Fem::DataOutputParameters, DataOutputParameters >
{
    DataOutputParameters ( const int step )
            : step_( step )
    {}

    DataOutputParameters ( const DataOutputParameters &other )
            : step_( other.step_ )
    {}

    std::string prefix () const
    {
        std::stringstream s;
        s << "poisson-" << step_ << "-";
        return s.str();
    }

    int outputformat() const{
        return 1;
    }

    double savestep() const{
        return 1.0;
    }

    int savecount() const{
        return 1;
    }

private:
    int step_;
};

// FemScheme
//----------

/*******************************************************************************
 * template arguments are:
 * - GridPart: the part of the grid used to tesselate the
 *             computational domain
 * - Model: description of the data functions and methods required for the
 *          elliptic operator (massFlux, diffusionFlux)
 *     Model::ProblemType boundary data, exact solution,
 *                        and the type of the function space
 *******************************************************************************/
template < class Model >
class FemScheme
{
public:
    //! type of the mathematical model
    typedef Model ModelType ;

    //! grid view (e.g. leaf grid view) provided in the template argument list
    typedef typename ModelType::GridPartType GridPartType;

    //! type of underlying hierarchical grid needed for data output
    typedef typename GridPartType::GridType GridType;

    //! type of function space (scalar functions, \f$ f: \Omega -> R \f$)
    typedef typename ModelType :: FunctionSpaceType   FunctionSpaceType;

    //! choose type of discrete function space
    typedef Dune::Fem::DiscontinuousGalerkinSpace< FunctionSpaceType, GridPartType, POLORDER > DiscreteFunctionSpaceType;

    // choose type of discrete function, Matrix implementation and solver implementation
#if HAVE_DUNE_ISTL && WANT_ISTL
    typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< DiscreteFunctionSpaceType > DiscreteFunctionType;
  typedef Dune::Fem::ISTLLinearOperator< DiscreteFunctionType, DiscreteFunctionType > LinearOperatorType;
  typedef Dune::Fem::ISTLCGOp< DiscreteFunctionType, LinearOperatorType > LinearInverseOperatorType;
#else
    typedef Dune::Fem::AdaptiveDiscreteFunction< DiscreteFunctionSpaceType > DiscreteFunctionType;
    typedef Dune::Fem::SparseRowLinearOperator< DiscreteFunctionType, DiscreteFunctionType > LinearOperatorType;
    typedef Dune::Fem::CGInverseOperator< DiscreteFunctionType > LinearInverseOperatorType;
#endif

    /*********************************************************/

    //! define Laplace operator
    typedef DifferentiableDGEllipticOperator< LinearOperatorType, ModelType > EllipticOperatorType;

    FemScheme( GridPartType &gridPart,
               const ModelType& implicitModel )
            : implicitModel_( implicitModel ),
              gridPart_( gridPart ),
              discreteSpace_( gridPart_ ),
              solution_( "solution", discreteSpace_ ),
              rhs_( "rhs", discreteSpace_ ),
            // the elliptic operator (implicit)
              implicitOperator_( implicitModel_, discreteSpace_ ),
            // create linear operator (domainSpace,rangeSpace)
              linearOperator_( "assembled elliptic operator", discreteSpace_, discreteSpace_ ),
            // exact solution
              solverEps_(1e-12)
    {
        // set all DoF to zero
        solution_.clear();
    }

    const DiscreteFunctionType &solution() const
    {
        return solution_;
    }

    //! setup the right hand side
    void prepare()
    {
        // assemble rhs
        assembleDGRHS ( implicitModel_, implicitModel_.rightHandSide(), implicitModel_.neumanBoundary(), implicitModel_.dirichletBoundary(), rhs_ );
    }

    //! solve the system - bool parameter
    //! false: only assemble if grid has changed
    //! true:  assemble in any case
    void solve ( bool assemble )
    {
        if( assemble )
        {
            // assemble linear operator (i.e. setup matrix)
            implicitOperator_.jacobian( solution_ , linearOperator_ );
        }

        // inverse operator using linear operator
        LinearInverseOperatorType invOp( linearOperator_, solverEps_, solverEps_ );
        // solve system
        invOp( rhs_, solution_ );
    }

protected:
    const ModelType& implicitModel_;   // the mathematical model

    GridPartType  &gridPart_;         // grid part(view), e.g. here the leaf grid the discrete space is build with

    DiscreteFunctionSpaceType discreteSpace_; // discrete function space
    DiscreteFunctionType solution_;   // the unknown
    DiscreteFunctionType rhs_;        // the right hand side

    EllipticOperatorType implicitOperator_; // the implicit operator

    LinearOperatorType linearOperator_;  // the linear operator (i.e. jacobian of the implicit)

    const double solverEps_ ; // eps for linear solver
};


#endif //DUNE_DIFFUSIONFEM_FEMSCHEME_HH
