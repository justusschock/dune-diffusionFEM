
#ifndef DUNE_TEST_PDETESTSUITE_H
#define DUNE_TEST_PDETESTSUITE_H


#include "stationaryDiffusion.hh"
#include "poisson.hh"
#include <cpptest.h>

template <int dim>
class PDETestSuite : public Test::Suite{

    using GV = typename Dune::YaspGrid<dim>::LeafGridView;
    using Traits = typename Dune::PDELab::ConvectionDiffusionParameterTraits<GV,double>;

public:
    PDETestSuite(stationaryDiffusion::SinkSourceDefault<typename Dune::YaspGrid<dim>::LeafGridView> &sink_term,
                 stationaryDiffusion::SinkSourceDefault<typename Dune::YaspGrid<dim>::LeafGridView> &source_term)
    :sink_term(sink_term), source_term(source_term)
    {
        //initialize grid
        setup();

        //add all tests
        TEST_ADD(PDETestSuite::test_DiffusionParameterClass);
    };

    virtual ~PDETestSuite() {
        grid.reset();
    }

protected:

    //setup ressources
    void setup() override {
        Dune::FieldVector<double, dim> L(1.0);
        Dune::array<int, dim> N (Dune::fill_array<int, dim>(1));
        std::bitset<dim> B(false);
        grid.reset(new Dune::YaspGrid<dim>(L,N,B,false));
    };

    //remove ressources
    void tear_down() override {
        grid.reset();
    }

    //Sink & Source term references
    stationaryDiffusion::SinkSourceDefault<GV> &sink_term, &source_term;

    //Grid pointer
    std::shared_ptr<Dune::YaspGrid<dim>> grid;

    void test_DiffusionParameterClass(){


        typename Traits::ElementType e;
        typename Traits::DomainType x;
        typename Traits::PermTensorType passTensor, failTensor;

        //Setup some example PermTensors for variable dim
        for (int i=0; i<dim; i++) {
            for (int j = 0; j < dim; j++) {
                if(i == j) {
                    passTensor[i][j] = 1;
                    failTensor[i][j] = 1;
                }
                else {
                    passTensor[i][j] = 0;
                    failTensor[i][j] = 1;
                }
            }
        }

        //Setup parameter class
        stationaryDiffusion::Parameter<GV, double> parameter(sink_term,source_term);

        //Test parameter.A()

        TEST_ASSERT_EQUALS_OBJ(passTensor, parameter.A(e, x));
        TEST_ASSERT(failTensor != parameter.A(e, x));

        //Setup some example FieldVectors and test parameter.b()
        typename Traits::RangeType passVelocityField(0), failVelocityField(1);
        TEST_ASSERT_EQUALS_OBJ(passVelocityField, parameter.b(e, x));
        TEST_ASSERT(failVelocityField != parameter.b(e, x));

        //Setup some example FieldTypes and test parameter.c()
        typename Traits::RangeFieldType failSink = sink_term.evaluate(e,x) + 42;
        TEST_ASSERT_EQUALS_OBJ(sink_term.evaluate(e,x), parameter.c(e, x));
        TEST_ASSERT(failSink != parameter.c(e, x));


        //Setup some example Fieldtypes and test parameter.f()

        typename Traits::RangeFieldType failSource = source_term.evaluate(e,x) + 42;
        TEST_ASSERT_EQUALS_OBJ(source_term.evaluate(e, x), parameter.f(e,x));
        TEST_ASSERT(failSource != parameter.f(e,x));


    }
};

#endif //DUNE_TEST_PDETESTSUITE_H
