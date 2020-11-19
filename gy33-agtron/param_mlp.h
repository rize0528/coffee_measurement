
#ifndef GY33_PARAM_H
#define MODEL_NAME "MLP"
#define MLP 1
#include <MatrixMath.h>
mtx_type X0[3][6] = {
{-0.277296824697875, -0.13368612874248798, -0.43386672598303266, 0.7218123323828692, 0.49831728017106847, 0.5619889526059939},
{0.21249483472978065, -0.1868387971360139, -0.4389240920925303, 0.6527012354375464, 0.43829934121748937, -0.12923110173487062},
{0.027486541485671496, 0.2951201687234561, 0.6835808423076193, 0.09011214751921189, 0.6365209853297996, 0.19905261182009168},
};
mtx_type X1[6][5] = {
{-0.5564822752664857, -0.2367376186519924, -0.2636641826592862, -0.35281927702211163, -0.26729935934991156},
{0.09304973181878165, 0.10553615445817396, -0.39458229052694, -0.5303259454244378, 0.2846848549990321},
{0.5635783938609681, -0.7569902291689379, -0.6751219522743213, -0.4263080904558724, 0.7243069576396319},
{-0.26384205181879017, 0.8759259447184883, 0.832620470146984, -0.6513829852945017, -0.5630172690075957},
{-0.6499243965582503, 0.683351175031905, -0.429366732589459, -0.5510693465794098, 0.35887918609381714},
{0.3205914184763398, -0.27398265350375406, 0.6772693877936962, -0.035358702000122276, -0.30848864299990053},
};
mtx_type X2[5][5] = {
{-0.36436525753875604, 0.8168539135479194, -0.41522759098739315, 0.2891105488589341, 0.15056379439857379},
{-0.5299076138070604, -0.7348403908560489, -0.29184236037890765, 0.0014126045079940956, -0.23299403478590788},
{0.5840092336831718, -0.1359675996322338, -0.07347119523387625, -0.46202103973254594, 0.42833130687047916},
{-0.14609340811011734, 0.5757512738157881, -0.2968854283946637, -0.28916518864896523, 0.19143824799469877},
{0.034410926512607715, 0.2317398651330018, 0.6571770469225494, -0.06993384170528634, 0.16664771440201587},
};
mtx_type X3[5][1] = {
{0.11386228360166004},
{0.9075946867511847},
{-0.007660970827600817},
{-1.066401524507356},
{-0.61053214605748},
};
mtx_type W0[6] = {-0.7672675504927644, -0.8925274162201805, 0.15441187356202246, -0.4927648580071387, -0.6484576535001657, 0.00420368123841698};
mtx_type W1[5] = {-0.20012834557952963, -0.10860785705395445, 0.15352376027334774, 0.42180545089004085, -0.3100188466048086};
mtx_type W2[5] = {-0.3403358018818524, -0.40675640217083003, 0.21601653706151694, -0.5436827404825457, -0.6046623013597268};
mtx_type W3[1] = {-0.5304927723595428};
#define MAX_DIM 6
#endif	// GY33_PARAM_H