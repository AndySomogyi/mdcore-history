
// include some standard headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#ifdef CELL
    #include <libspe2.h>
    #include <ppu_intrinsics.h>
#else
    #include "cycle.h"
#endif

// include local headers
#include "part.h"
#include "potential.h"
#include "cell.h"
#include "space.h"
#include "engine.h"

int main ( int argc , char *argv[] ) {

    // hard-wired OO-potential
    #ifdef CELL
        double alpha_OO[3] = { -1.6684940013428644e+01 , 9.2609640080571850e+01 , -4.5924700067143213e+01 };
        double c_OO[180] = { -7.2977751193029405e-02, 6.9463287090496051e-01, -5.4960498047096280e+00, 3.5532487535859175e+01, -1.6350888550254928e+02, 3.9918084963688023e+02, -2.6629239449110491e-02, 2.6523528071435643e-01, -2.1984497046546965e+00, 1.4832268620621987e+01, -7.0917151641196213e+01, 1.7809032249855605e+02, -1.0221877407049464e-02, 1.0612347811784649e-01, -9.1718920764359047e-01, 6.4240155640541543e+00, -3.1690483198605953e+01, 8.0897562488763768e+01, -4.1049189994002743e-03, 4.4252037536472673e-02, -3.9693047514283902e-01, 2.8700928085696327e+00, -1.4488829120077801e+01, 3.7005597886953026e+01, -1.7162447247102617e-03, 1.9137647645535230e-02, -1.7729494447388375e-01, 1.3151992352325905e+00, -6.7255264855161379e+00, 1.6791203661246236e+01, -7.4384166972019106e-04, 8.5456327114045456e-03, -8.1338816757221544e-02, 6.1446470657880570e-01, -3.1403139803572850e+00, 7.3755958913351938e+00, -3.3288811108623487e-04, 3.9234377831132558e-03, -3.8140329981638341e-02, 2.9072827300582188e-01, -1.4564462938029346e+00, 2.9866383691565832e+00, -1.5325925799004682e-04, 1.8443448016094539e-03, -1.8182598060772024e-02, 1.3815215599837066e-01, -6.5782732083865614e-01, 9.7021088909246089e-01, -7.2327348031154501e-05, 8.8385036251768790e-04, -8.7587592672105190e-03, 6.5187299862145220e-02, -2.7878688430682325e-01, 8.0355090588294958e-02, -3.4860454943199827e-05, 4.2970892894215909e-04, -4.2302140624908210e-03, 3.0004054688947110e-02, -1.0105694765010979e-01, -2.7694751589454014e-01, -1.7092329365564734e-05, 2.1072147422363375e-04, -2.0261296795919480e-03, 1.3035949366292043e-02, -2.0438119964469342e-02, -3.8756378725813723e-01, -8.4864435643599515e-06, 1.0343875389676174e-04, -9.4582803237919417e-04, 4.9452461645268370e-03, 1.3505322339395007e-02, -3.8929613544602054e-01, -4.2425850740703280e-06, 5.0270855944664855e-05, -4.1658484515842399e-04, 1.2017929132940638e-03, 2.5395099596065888e-02, -3.4797423426755081e-01, -2.1189803531405432e-06, 2.3755689927360355e-05, -1.6021302430290860e-04, -4.1983162165469117e-04, 2.7273645626135525e-02, -2.9424050877927188e-01, -1.0448688924680403e-06, 1.0541419538878477e-05, -3.9548419294950433e-05, -1.0215019154165826e-03, 2.4987849207522110e-02, -2.4156705467430734e-01, -4.9833746092992571e-07, 4.0265462812701882e-06, 1.3824262094062182e-05, -1.1499054924504701e-03, 2.1315969746066126e-02, -1.9515558463398189e-01, -2.2035273499330437e-07, 9.0090536936427586e-07, 3.4285592327153195e-05, -1.0746625660481175e-03, 1.7517787348367848e-02, -1.5634655366081307e-01, -8.0299922858146992e-08, -5.1373087029704674e-07, 3.9154209945140589e-05, -9.2635541459566467e-04, 1.4109348587884178e-02, -1.2479317109015940e-01, -1.1423530608373696e-08, -1.0746514478744554e-06, 3.7115210386386033e-05, -7.6580175615751451e-04, 1.1251467105797041e-02, -9.9516170437460694e-02, 2.0769865476743194e-08, -1.2212256229780135e-06, 3.2461083728916188e-05, -6.1909409366660523e-04, 8.9439494121932635e-03, -7.9397729367336023e-02, 3.4249989297934335e-08, -1.1781570707630953e-06, 2.7248554887561766e-05, -4.9539009805675636e-04, 7.1215126573981116e-03, -6.3396249703435464e-02, 3.8429133426782069e-08, -1.0593267870959308e-06, 2.2404360022718161e-05, -3.9593360867176648e-04, 5.6998873092766736e-03, -5.0624699462396325e-02, 3.8256318541302365e-08, -9.2147261655713028e-07, 1.8295422259397387e-05, -3.1859410218693757e-04, 4.5968038714201522e-03, -4.0364709174620789e-02, 3.6357691839946204e-08, -7.9231592309693474e-07, 1.5024749844844373e-05, -2.6011891878514014e-04, 3.7405061125469958e-03, -3.2052608744634274e-02, 3.4191983886990486e-08, -6.8575276892348269e-07, 1.2588681765972492e-05, -2.1723995661767311e-04, 3.0722825561228582e-03, -2.5255121619999515e-02, 3.2748641951576823e-08, -6.1110246618708333e-07, 1.0971338969764472e-05, -1.8726166570648224e-04, 2.5463924480631566e-03, -1.9642872595665852e-02, 3.3204648599677779e-08, -5.8202884724353916e-07, 1.0236416137597937e-05, -1.6857955414664497e-04, 2.1293344429974370e-03, -1.4964583181386127e-02, 3.8540683537432984e-08, -6.3817473128471267e-07, 1.0744730677255889e-05, -1.6191119876222309e-04, 1.8005055018835436e-03, -1.1020302553858536e-02, 6.4134897112686876e-08, -9.7341992350597008e-07, 1.4308262080426582e-05, -1.7646918704331970e-04, 1.5623850759884177e-03, -7.6173630741749243e-03, 1.2412338148577834e-06, -1.0589886796958070e-05, 7.4608496831993259e-05, -4.2422862381832861e-04, 1.7328171215021315e-03, -3.9745628454115238e-03 };
        double mi_OO[30] = { 2.0679200929746125e-01, 2.2049416343790734e-01, 2.3443897617589066e-01, 2.4863981976667110e-01, 2.6311134218337034e-01, 2.7786964436664735e-01, 2.9293249046795367e-01, 3.0831955892194363e-01, 3.2405274448378563e-01, 3.4015652448392036e-01, 3.5665840683146965e-01, 3.7358948325505492e-01, 3.9098511969495375e-01, 4.0888582788090549e-01, 4.2733837989906492e-01, 4.4639725415604148e-01, 4.6612654192968073e-01, 4.8660250789173071e-01, 5.0791710213479169e-01, 5.3018289615278191e-01, 5.5354022104976053e-01, 5.7816784685619926e-01, 6.0429962927214409e-01, 6.3225181035686817e-01, 6.6247078230650747e-01, 6.9562411664362855e-01, 7.3279610935589168e-01, 7.7599223009146412e-01, 8.2995497466958357e-01, 9.3024114549054515e-01 };
        double hi_OO[30] = { 1.4723183614807280e+02, 1.4471476687137709e+02, 1.4215312150143987e+02, 1.3954444357101212e+02, 1.3688604226500453e+02, 1.3417495985993571e+02, 1.3140793310249532e+02, 1.2858134708784729e+02, 1.2569117977573400e+02, 1.2273293470992076e+02, 1.1970155872050135e+02, 1.1659134029423396e+02, 1.1339578275031633e+02, 1.1010744413259937e+02, 1.0671773246496973e+02, 1.0321664012949377e+02, 9.9592393635322665e+01, 9.5830983253753985e+01, 9.1915517863092518e+01, 8.7825318214960205e+01, 8.3534605653902119e+01, 7.9010540339795938e+01, 7.4210163261107297e+01, 6.9075381130953886e+01, 6.3524192238455328e+01, 5.7433964495823275e+01, 5.0605514030988566e+01, 4.2670438475607348e+01, 3.2757547037407470e+01, 1.4335097774067718e+01 };
        double alpha_HH[3] = { -9.9834718957418245e-01 , 2.5457026928928741e+01 , -1.2458679739354558e+01 };
        double c_HH[72] = { -1.8550217536746450e-02, 5.3250931545148136e-02, -1.1585768963916021e-01, 3.4011726816945437e-01, -1.0079109454304416e+00, 2.3493872519129653e+00, -8.0145660469800400e-04, 3.7773960513719845e-03, -1.6229782806672876e-02, 7.7627170714717877e-02, -3.5777708479392212e-01, 1.1442237080498368e+00, -1.1346016097610125e-04, 7.3048395961533672e-04, -4.5099959031438723e-03, 2.9618306636017413e-02, -1.7618364670354425e-01, 6.3966405724733655e-01, -2.7937469295711619e-05, 2.2490589507870708e-04, -1.7902917156006660e-03, 1.4661346605617869e-02, -9.9723396664734898e-02, 3.7277786672497720e-01, -9.5859273624258889e-06, 9.1996123930441150e-05, -8.9288059050951149e-04, 8.4258753043362083e-03, -5.9915859609205140e-02, 2.1678629582069878e-01, -4.1494048083050988e-06, 4.6455619378162848e-05, -5.2452487557593051e-04, 5.2679844117444041e-03, -3.6481761894573431e-02, 1.2216096491580812e-01, -2.1820671015723167e-06, 2.8177219448088270e-05, -3.4678548291673688e-04, 3.4180279388162905e-03, -2.1743537790471661e-02, 6.4926258273218670e-02, -1.4107467148495416e-06, 2.0236959631769154e-05, -2.4726377366689591e-04, 2.2114585011531497e-03, -1.2272656437893267e-02, 3.1526931671933149e-02, -1.1720439108947836e-06, 1.6804294466049970e-05, -1.8223859543129993e-04, 1.3718117255411236e-03, -6.3027000694554419e-03, 1.3362569445638243e-02, -1.2939024808480116e-06, 1.5649148696367348e-05, -1.3343807241044794e-04, 7.7830960195393757e-04, -2.7697756808719733e-03, 4.5666083504342057e-03, -2.0028300399545082e-06, 1.6584762109672617e-05, -9.4459660044431354e-05, 3.7502844630067167e-04, -9.1801523225088866e-04, 1.0499685145756220e-03, -1.6952808564374106e-05, 5.0868418969332218e-05, -8.4239045653007326e-05, 1.1700796284634816e-04, -1.1037634834609118e-04, 4.7659416271538858e-05 };
        double mi_HH[12] = { 6.0885618668288052e-02, 1.0362929603650858e-01, 1.4846782984653101e-01, 1.9574421196038261e-01, 2.4590708707741782e-01, 2.9956309428822370e-01, 3.5756842279411472e-01, 4.2120325381622115e-01, 4.9254634381538664e-01, 5.7544073073061297e-01, 6.7896542601280596e-01, 8.6876014138259516e-01 };
        double hi_HH[12] = { 4.7879836162974819e+01, 4.5749717014122972e+01, 4.3515201283237829e+01, 4.1159196067270557e+01, 3.8659343283716950e+01, 3.5985411243989255e+01, 3.3094731999865481e+01, 2.9923508079976006e+01, 2.6368145240313822e+01, 2.2237126765245986e+01, 1.7078002670705548e+01, 7.6196363706489203e+00 };
        double alpha_OH[3] = { -1.1647383878365460e+00 , 2.9699864750416864e+01 , -1.4535126362580318e+01 };
        double c_OH[84] = { 2.1717867606800660e-02, -6.9530533630346739e-02, 1.7928131743860506e-01, -5.8392558355674140e-01, 1.9088866446261132e+00, -5.0068066616092244e+00, 1.2296903581451143e-03, -6.2166956545699125e-03, 2.9004273763616060e-02, -1.4841503295443589e-01, 7.3801831845383237e-01, -2.6274929379401941e+00, 1.9161318060393777e-04, -1.3052177286675991e-03, 8.5476022062428800e-03, -5.9211593137971245e-02, 3.8093422985988140e-01, -1.5636821278602393e+00, 4.9063240329639112e-05, -4.1549478437156800e-04, 3.4652441998812727e-03, -2.9940047713608257e-02, 2.2492476471146605e-01, -9.7573087883049314e-01, 1.6997780511434001e-05, -1.7067390560443062e-04, 1.7230341471068122e-03, -1.7466853565584958e-02, 1.4208850891162189e-01, -6.1618128086636648e-01, 7.2535591180174815e-06, -8.4044482520110073e-05, 9.9269405672370897e-04, -1.1149230551848398e-02, 9.2471844740010592e-02, -3.8528520010809492e-01, 3.6321957718733555e-06, -4.7956051326491776e-05, 6.3998470807796423e-04, -7.5085551480451335e-03, 6.0414755717482706e-02, -2.3443179896590696e-01, 2.0947520702110228e-06, -3.1266124514509565e-05, 4.4903157074285827e-04, -5.1849097300335325e-03, 3.8794052343500147e-02, -1.3646858302308479e-01, 1.4023495904948714e-06, -2.3097360457033399e-05, 3.3381227605014271e-04, -3.5806356885690368e-03, 2.3984569926052576e-02, -7.4516062610600314e-02, 1.1200880616376346e-06, -1.9069828461712240e-05, 2.5598189906343119e-04, -2.4136337335821504e-03, 1.3941849671127972e-02, -3.7168355887636277e-02, 1.0952938486049251e-06, -1.7242077860828701e-05, 1.9734921689899891e-04, -1.5457955920355757e-03, 7.3763995904084145e-03, -1.6264226020374674e-02, 1.3287722141252490e-06, -1.6849010758587357e-05, 1.4959083608734604e-04, -9.0774343614453349e-04, 3.3689186846175066e-03, -5.8079888492929624e-03, 2.1551518618222974e-06, -1.8572097939440654e-05, 1.1075830827603763e-04, -4.6127818177079153e-04, 1.1876882970428073e-03, -1.4322909663862058e-03, 1.9366202110736622e-05, -6.1646052760039527e-05, 1.1257383088430141e-04, -1.6937480035455655e-04, 1.6986211593910072e-04, -7.8716486867035914e-05 };
        double mi_OH[14] = { 5.7845491216141368e-02, 9.4236715358401835e-02, 1.3211795466729082e-01, 1.7168915204351087e-01, 2.1319953735017039e-01, 2.5696658069179196e-01, 3.0340553155298022e-01, 3.5307785293192517e-01, 4.0677593614317875e-01, 4.6568449302577863e-01, 5.3172706862486963e-01, 6.0845917626256529e-01, 7.0427758754129699e-01, 8.7923513678195431e-01 };
        double hi_OH[14] = { 5.6036563403505141e+01, 5.3920759237918190e+01, 5.1718324837414826e+01, 4.9417635420686928e+01, 4.7004200637520199e+01, 4.4459562615572182e+01, 4.1759578539920035e+01, 3.8871604668057408e+01, 3.5749570968441702e+01, 3.2324597695938351e+01, 2.8484848969366151e+01, 2.4023605447042065e+01, 1.8452674583849820e+01, 8.2805542386485449e+00 };
    #endif

    const double origin[3] = { 0.0 , 0.0 , 0.0 };
    // const double dim[3] = { 3.166 , 3.166 , 3.166 };
    // const int nr_mols = 1000, nx = 10;
    const double dim[3] = { 6.332 , 6.332 , 6.332 };
    const int nr_mols = 8000, nx = 20;
    // const double dim[3] = { 4.0 , 4.0 , 4.0 };
    // const int nr_mols = 2016, nx = 13;
    // const double dim[3] = { 8.0 , 8.0 , 8.0 };
    // const int nr_mols = 16128, nx = 26;
    
    double x[3], vtot[3] = { 0.0 , 0.0 , 0.0 };
    double epot, ekin, v2, temp, ee, eff;
    struct engine e;
    struct part p_O, p_H;
    struct potential *pot_OO, *pot_OH, *pot_HH;
    struct cellpair cp;
    int i, j, k, cid, pid, nr_runners = 1, nr_steps = 1000;
    double old_O[3], old_H1[3], old_H2[3], new_O[3], new_H1[3], new_H2[3];
    double v_OH1[3], v_OH2[3], v_HH[3];
    double d_OH1, d_OH2, d_HH, lambda;
    double vcom[3], vcom_tot[3], w;
    #ifdef CELL
        unsigned long long tic, toc;
    #else
        ticks tic, toc;
    #endif
    
    #ifdef CELL
        tic = __mftb();
    #else
        tic = getticks();
    #endif

    // initialize the engine
    printf("main: initializing the engine... "); fflush(stdout);
    if ( engine_init( &e , origin , dim , 1.0 , space_periodic_full , 2 ) != 0 ) {
        printf("main: engine_init failed with engine_err=%i.\n",engine_err);
        return 1;
        }
    printf("done.\n"); fflush(stdout);
        
    // mix-up the pair list just for kicks
    // printf("main: shuffling the interaction pairs... "); fflush(stdout);
    // srand(6178);
    // for ( i = 0 ; i < e.s.nr_pairs ; i++ ) {
    //     j = rand() % e.s.nr_pairs;
    //     if ( i != j ) {
    //         cp = e.s.pairs[i];
    //         e.s.pairs[i] = e.s.pairs[j];
    //         e.s.pairs[j] = cp;
    //         }
    //     }
    // printf("done.\n"); fflush(stdout);
        

    // initialize the O-H potential
    #ifdef CELL
        pot_OH = (struct potential *)malloc( sizeof(struct potential) );
        pot_OH->a = 0.04;
        pot_OH->b = 1.0;
        pot_OH->n = 14;
        pot_OH->alpha[0] = alpha_OH[0]; pot_OH->alpha[1] = alpha_OH[1]; pot_OH->alpha[2] = alpha_OH[2];
        pot_OH->c = c_OH;
        pot_OH->mi = mi_OH;
        pot_OH->hi = hi_OH;
    #else
        if ( ( pot_OH = potential_create_Ewald( 0.04 , 1.0 , -0.35921288 , 3.0 , 1.0e-3 ) ) == NULL ) {
            printf("main: potential_createLJ126_Ewald failed with potential_err=%i.\n",potential_err);
            return 1;
            }
        // printf("double alpha_OH[3] = { %.16e , %.16e , %.16e };\n",pot_OH->alpha[0],pot_OH->alpha[1],pot_OH->alpha[2]);
        // printf("double c_OH[%i] = { %.16e",pot_OH->n*6,pot_OH->c[0]);
        // for ( i = 1 ; i < pot_OH->n*6 ; i++ )
        //     printf(", %.16e",pot_OH->c[i]);
        // printf(" };\n");
        // printf("double mi_OH[%i] = { %.16e",pot_OH->n,pot_OH->mi[0]);
        // for ( i = 1 ; i < pot_OH->n ; i++ )
        //     printf(", %.16e",pot_OH->mi[i]);
        // printf(" };\n");
        // printf("double hi_OH[%i] = { %.16e",pot_OH->n,pot_OH->hi[0]);
        // for ( i = 1 ; i < pot_OH->n ; i++ )
        //     printf(", %.16e",pot_OH->hi[i]);
        // printf(" };\n");
    #endif
    printf("main: constructed OH-potential with %i intervals.\n",pot_OH->n); fflush(stdout);
    #ifdef EXPLICIT_POTENTIALS
        pot_OH->flags = potential_flag_Ewald;
        pot_OH->alpha[0] = 0.0;
        pot_OH->alpha[1] = 0.0;
        pot_OH->alpha[2] = -0.35921288;
    #endif

    // initialize the H-H potential
    #ifdef CELL
        pot_HH = (struct potential *)malloc( sizeof(struct potential) );
        pot_HH->a = 0.04;
        pot_HH->b = 1.0;
        pot_HH->n = 12;
        pot_HH->alpha[0] = alpha_HH[0]; pot_HH->alpha[1] = alpha_HH[1]; pot_HH->alpha[2] = alpha_HH[2];
        pot_HH->c = c_HH;
        pot_HH->mi = mi_HH;
        pot_HH->hi = hi_HH;
    #else
        if ( ( pot_HH = potential_create_Ewald( 0.04 , 1.0 , 1.7960644e-1 , 3.0 , 1.0e-3 ) ) == NULL ) {
            printf("main: potential_createLJ126_Ewald failed with potential_err=%i.\n",potential_err);
            return 1;
            }
        // printf("double alpha_HH[3] = { %.16e , %.16e , %.16e };\n",pot_HH->alpha[0],pot_HH->alpha[1],pot_HH->alpha[2]);
        // printf("double c_HH[%i] = { %.16e",pot_HH->n*6,pot_HH->c[0]);
        // for ( i = 1 ; i < pot_HH->n*6 ; i++ )
        //     printf(", %.16e",pot_HH->c[i]);
        // printf(" };\n");
        // printf("double mi_HH[%i] = { %.16e",pot_HH->n,pot_HH->mi[0]);
        // for ( i = 1 ; i < pot_HH->n ; i++ )
        //     printf(", %.16e",pot_HH->mi[i]);
        // printf(" };\n");
        // printf("double hi_HH[%i] = { %.16e",pot_HH->n,pot_HH->hi[0]);
        // for ( i = 1 ; i < pot_HH->n ; i++ )
        //     printf(", %.16e",pot_HH->hi[i]);
        // printf(" };\n");
    #endif
    printf("main: constructed HH-potential with %i intervals.\n",pot_HH->n); fflush(stdout);
    #ifdef EXPLICIT_POTENTIALS
        pot_HH->flags = potential_flag_Ewald;
        pot_HH->alpha[0] = 0.0;
        pot_HH->alpha[1] = 0.0;
        pot_HH->alpha[2] = 1.7960644e-1;
    #endif

    // initialize the O-O potential
    #ifdef CELL
        pot_OO = (struct potential *)malloc( sizeof(struct potential) );
        pot_OO->a = 0.2;
        pot_OO->b = 1.0;
        pot_OO->n = 30;
        pot_OO->alpha[0] = alpha_OO[0]; pot_OO->alpha[1] = alpha_OO[1]; pot_OO->alpha[2] = alpha_OO[2];
        pot_OO->c = c_OO;
        pot_OO->mi = mi_OO;
        pot_OO->hi = hi_OO;
    #else
        if ( ( pot_OO = potential_createLJ126_Ewald( 0.2 , 1.0 , 2.637775819766153e-06 , 2.619222661792581e-03 , 7.1842576e-01 , 3.0 , 1.0e-3 ) ) == NULL ) {
            printf("main: potential_createLJ126_Ewald failed with potential_err=%i.\n",potential_err);
            return 1;
            }
        // printf("double alpha_OO[3] = { %.16e , %.16e , %.16e };\n",pot_OO->alpha[0],pot_OO->alpha[1],pot_OO->alpha[2]);
        // printf("double c_OO[%i] = { %.16e",pot_OO->n*6,pot_OO->c[0]);
        // for ( i = 1 ; i < pot_OO->n*6 ; i++ )
        //     printf(", %.16e",pot_OO->c[i]);
        // printf(" };\n");
        // printf("double mi_OO[%i] = { %.16e",pot_OO->n,pot_OO->mi[0]);
        // for ( i = 1 ; i < pot_OO->n ; i++ )
        //     printf(", %.16e",pot_OO->mi[i]);
        // printf(" };\n");
        // printf("double hi_OO[%i] = { %.16e",pot_OO->n,pot_OO->hi[0]);
        // for ( i = 1 ; i < pot_OO->n ; i++ )
        //     printf(", %.16e",pot_OO->hi[i]);
        // printf(" };\n");
    #endif
    printf("main: constructed OO-potential with %i intervals.\n",pot_OO->n); fflush(stdout);
    #ifdef EXPLICIT_POTENTIALS
        pot_OO->flags = potential_flag_LJ126 + potential_flag_Ewald;
        pot_OO->alpha[0] = 2.637775819766153e-06;
        pot_OO->alpha[1] = 2.619222661792581e-03;
        pot_OO->alpha[2] = 7.1842576e-01;
    #endif
    // for ( i = 0 ; i < 1000 ; i++ ) {
    //     temp = 0.3 + (double)i/1000 * 0.7;
    //     potential_eval( &pot , temp*temp , &ee , &eff );
    //     printf("%e %e %e %e\n", temp , ee , eff , dfdr( temp ) );
    //     }
    // return 0;
        
    
    // register this particle type
    e.p[0] = pot_OO;
    e.p[3] = pot_HH;
    e.p[1] = pot_OH;
    e.p[2] = pot_OH;
    e.types[0].mass = 15.9994;
    e.types[0].imass = 1.0 / 15.9994;
    e.types[0].charge = -0.8476;
    e.types[1].mass = 1.00794;
    e.types[1].imass = 1.0 / 1.00794;
    e.types[1].charge = 0.4238;
        
    // set fields for all particles
    srand(6178);
    p_O.type = 0;
    p_H.type = 1;
    p_O.flags = part_flag_none;
    p_H.flags = part_flag_none;
    for ( k = 0 ; k < 3 ; k++ ) {
        p_O.v[k] = 0.0; p_H.v[k] = 0.0;
        p_O.f[k] = 0.0; p_H.f[k] = 0.0;
        }
    #ifdef VECTORIZE
        p_O.v[3] = 0.0; p_O.f[3] = 0.0; p_O.x[3] = 0.0;
        p_H.v[3] = 0.0; p_H.f[3] = 0.0; p_H.x[3] = 0.0;
    #endif
    
    // create and add the particles
    printf("main: initializing particles... "); fflush(stdout);
    for ( i = 0 ; i < nx ; i++ ) {
        x[0] = 0.1 + i * 0.31;
        for ( j = 0 ; j < nx ; j++ ) {
            x[1] = 0.1 + j * 0.31;
            for ( k = 0 ; k < nx && k + nx * ( j + nx * i ) < nr_mols ; k++ ) {
                p_O.id = 3 * (k + nx * ( j + nx * i ));
                x[2] = 0.1 + k * 0.31;
                p_O.v[0] = ((double)rand()) / RAND_MAX - 0.5;
                p_O.v[1] = ((double)rand()) / RAND_MAX - 0.5;
                p_O.v[2] = ((double)rand()) / RAND_MAX - 0.5;
                temp = 0.7 / sqrt( p_O.v[0]*p_O.v[0] + p_O.v[1]*p_O.v[1] + p_O.v[2]*p_O.v[2] );
                p_O.v[0] *= temp; p_O.v[1] *= temp; p_O.v[2] *= temp;
                vtot[0] += p_O.v[0]; vtot[1] += p_O.v[1]; vtot[2] += p_O.v[2];
                if ( space_addpart( &(e.s) , &p_O , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    return 1;
                    }
                x[0] += 0.1;
                p_H.id = 3 * (k + nx * ( j + nx * i )) + 1;
                p_H.v[0] = p_O.v[0]; p_H.v[1] = p_O.v[1]; p_H.v[2] = p_O.v[2];
                if ( space_addpart( &(e.s) , &p_H , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    return 1;
                    }
                x[0] -= 0.13333;
                x[1] += 0.09428;
                p_H.id = 3 * (k + nx * ( j + nx * i )) + 2;
                if ( space_addpart( &(e.s) , &p_H , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    return 1;
                    }
                x[0] += 0.03333;
                x[1] -= 0.09428;
                }
            }
        }
    for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
        for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ )
            for ( v2 = 0.0 , k = 0 ; k < 3 ; k++ )
                e.s.cells[cid].parts[pid].v[k] -= vtot[k] / 8000;
    printf("done.\n"); fflush(stdout);
        
    // set the time and time-step by hand
    e.time = 0;
    e.dt = 0.002;
    
    #ifdef CELL
        toc = __mftb();
        printf("main: setup took %.3f ms.\n",(double)(toc-tic) / 25000);
    #else
        toc = getticks();
        printf("main: setup took %.3f ms.\n",elapsed(toc,tic) / 2300000);
    #endif
    
    // did the user specify a number of runners?
    if ( argc > 1 )
        nr_runners = atoi( argv[1] );
    #ifdef CELL
        else
            nr_runners = spe_cpu_info_get( SPE_COUNT_USABLE_SPES , -1 );
    #endif
        
    // start the engine
    #ifdef CELL
        if ( engine_start( &e , nr_runners ) != 0 ) {
            printf("main: engine_start failed with engine_err=%i.\n",engine_err);
            return 1;
            }
    #else
        if ( engine_start( &e , nr_runners ) != 0 ) {
            printf("main: engine_start failed with engine_err=%i.\n",engine_err);
            return 1;
            }
    #endif
    
    // did the user specify a number of steps?
    if ( argc > 2 )
        nr_steps = atoi( argv[2] );
        
    // do a few steps
    for ( i = 0 ; i < nr_steps ; i++ ) {
    
        // take a step
        #ifdef CELL
            tic = __mftb();
        #else
            tic = getticks();
        #endif
        if ( engine_step( &e ) != 0 ) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            fflush(stdout);
            return 1;
            }
        #ifdef CELL
            toc = __mftb();
        #else
            toc = getticks();
        #endif
        
        // shake the water molecules
        for ( j = 0 ; j < nr_mols ; j++ ) {
        
            // unpack the data
            for ( k = 0 ; k < 3 ; k++ ) {
                new_O[k] = e.s.partlist[j*3]->x[k] + e.s.celllist[j*3]->origin[k];
                old_O[k] = new_O[k] - e.dt * e.s.partlist[j*3]->v[k];
                new_H1[k] = e.s.partlist[j*3+1]->x[k] + e.s.celllist[j*3+1]->origin[k];
                if ( new_H1[k] - new_O[k] > e.s.dim[k] * 0.5 )
                    new_H1[k] -= e.s.dim[k];
                else if ( new_H1[k] - new_O[k] < -e.s.dim[k] * 0.5 )
                    new_H1[k] += e.s.dim[k];
                old_H1[k] = new_H1[k] - e.dt * e.s.partlist[j*3+1]->v[k];
                new_H2[k] = e.s.partlist[j*3+2]->x[k] + e.s.celllist[j*3+2]->origin[k];
                if ( new_H2[k] - new_O[k] > e.s.dim[k] * 0.5 )
                    new_H2[k] -= e.s.dim[k];
                else if ( new_H2[k] - new_O[k] < -e.s.dim[k] * 0.5 )
                    new_H2[k] += e.s.dim[k];
                old_H2[k] = new_H2[k] - e.dt * e.s.partlist[j*3+2]->v[k];
                v_OH1[k] = old_O[k] - old_H1[k];
                v_OH2[k] = old_O[k] - old_H2[k];
                v_HH[k] = old_H1[k] - old_H2[k];
                }
                
            // main loop
            while ( 1 ) {
            
                // correct for the OH1 constraint
                for ( d_OH1 = 0.0 , k = 0 ; k < 3 ; k++ )
                    d_OH1 += (new_O[k] - new_H1[k]) * (new_O[k] - new_H1[k]);
                lambda = 0.5 * ( 0.1*0.1 - d_OH1 ) /
                    ( (new_O[0] - new_H1[0]) * v_OH1[0] + (new_O[1] - new_H1[1]) * v_OH1[1] + (new_O[2] - new_H1[2]) * v_OH1[2] );
                for ( k = 0 ; k < 3 ; k++ ) {
                    new_O[k] += lambda * 1.00794 / ( 1.00794 + 15.9994 ) * v_OH1[k];
                    new_H1[k] -= lambda * 15.9994 / ( 1.00794 + 15.9994 ) * v_OH1[k];
                    }
                    
                // correct for the OH2 constraint
                for ( d_OH2 = 0.0 , k = 0 ; k < 3 ; k++ )
                    d_OH2 += (new_O[k] - new_H2[k]) * (new_O[k] - new_H2[k]);
                lambda = 0.5 * ( 0.1*0.1 - d_OH2 ) /
                    ( (new_O[0] - new_H2[0]) * v_OH2[0] + (new_O[1] - new_H2[1]) * v_OH2[1] + (new_O[2] - new_H2[2]) * v_OH2[2] );
                for ( k = 0 ; k < 3 ; k++ ) {
                    new_O[k] += lambda * 1.00794 / ( 1.00794 + 15.9994 ) * v_OH2[k];
                    new_H2[k] -= lambda * 15.9994 / ( 1.00794 + 15.9994 ) * v_OH2[k];
                    }
                    
                // correct for the HH constraint
                for ( d_HH = 0.0 , k = 0 ; k < 3 ; k++ )
                    d_HH += (new_H1[k] - new_H2[k]) * (new_H1[k] - new_H2[k]);
                lambda = 0.5 * ( 0.1633*0.1633 - d_HH ) /
                    ( (new_H1[0] - new_H2[0]) * v_HH[0] + (new_H1[1] - new_H2[1]) * v_HH[1] + (new_H1[2] - new_H2[2]) * v_HH[2] );
                for ( k = 0 ; k < 3 ; k++ ) {
                    new_H1[k] += lambda * 0.5 * v_HH[k];
                    new_H2[k] -= lambda * 0.5 * v_HH[k];
                    }
                    
                // check the tolerances
                if ( fabs( d_OH1 - 0.1*0.1 ) < 1e-8 &&
                    fabs( d_OH2 - 0.1*0.1 ) < 1e-8 &&  
                    fabs( d_HH - 0.1633*0.1633 ) < 1e-8 )
                    break;
                    
                // printf("main: mol %i: d_OH1=%e, d_OH2=%e, d_HH=%e.\n",j,sqrt(d_OH1),sqrt(d_OH2),sqrt(d_HH));
                // getchar();
                    
                }
                
            // write the positions back
            for ( k = 0 ; k < 3 ; k++ ) {
            
                // write O
                e.s.partlist[j*3]->x[k] = new_O[k] - e.s.celllist[j*3]->origin[k];
                e.s.partlist[j*3]->v[k] = (new_O[k] - old_O[k]) / e.dt;
                
                // write H1
                if ( new_H1[k] - e.s.celllist[j*3+1]->origin[k] > e.s.dim[k] * 0.5 )
                    e.s.partlist[j*3+1]->x[k] = new_H1[k] - e.s.celllist[j*3+1]->origin[k] - e.s.dim[k];
                else if ( new_H1[k] - e.s.celllist[j*3+1]->origin[k] < -e.s.dim[k] * 0.5 )
                    e.s.partlist[j*3+1]->x[k] = new_H1[k] - e.s.celllist[j*3+1]->origin[k] + e.s.dim[k];
                else
                    e.s.partlist[j*3+1]->x[k] = new_H1[k] - e.s.celllist[j*3+1]->origin[k];
                e.s.partlist[j*3+1]->v[k] = (new_H1[k] - old_H1[k]) / e.dt;
                
                // write H2
                if ( new_H2[k] - e.s.celllist[j*3+2]->origin[k] > e.s.dim[k] * 0.5 )
                    e.s.partlist[j*3+2]->x[k] = new_H2[k] - e.s.celllist[j*3+2]->origin[k] - e.s.dim[k];
                else if ( new_H2[k] - e.s.celllist[j*3+2]->origin[k] < -e.s.dim[k] * 0.5 )
                    e.s.partlist[j*3+2]->x[k] = new_H2[k] - e.s.celllist[j*3+2]->origin[k] + e.s.dim[k];
                else
                    e.s.partlist[j*3+2]->x[k] = new_H2[k] - e.s.celllist[j*3+2]->origin[k];
                e.s.partlist[j*3+2]->v[k] = (new_H2[k] - old_H2[k]) / e.dt;
                }
                
            } // shake molecules
            
        // re-shuffle the space just to be sure...
        if ( space_shuffle( &e.s ) < 0 ) {
            printf("main: space_shuffle failed with space_err=%i.\n",space_err);
            return 1;
            }
            
            
        // get the total COM-velocities and ekin
        vcom_tot[0] = 0.0; vcom_tot[1] = 0.0; vcom_tot[2] = 0.0;
        ekin = 0.0;
        for ( j = 0 ; j < nr_mols ; j++ ) {
            for ( k = 0 ; k < 3 ; k++ ) {
                vcom[k] = ( e.s.partlist[j*3]->v[k] * 15.9994 +
                    e.s.partlist[j*3+1]->v[k] * 1.00794 +
                    e.s.partlist[j*3+2]->v[k] * 1.00794 ) / 1.801528e+1;
                vcom_tot[k] += vcom[k];
                }
            ekin += 9.00764 * ( vcom[0]*vcom[0] + vcom[1]*vcom[1] + vcom[2]*vcom[2] );
            }
        vcom_tot[0] /= 1000 * 1.801528e+1;
        vcom_tot[1] /= 1000 * 1.801528e+1;
        vcom_tot[2] /= 1000 * 1.801528e+1;
                
        // compute the temperature and scaling
        temp = ekin / ( 1.5 * 6.022045E23 * 1.380662E-26 * nr_mols );
        w = sqrt( 1.0 + 0.1 * ( 300.0 / temp - 1.0 ) );

        // compute the molecular heat
        if ( i < 5000 ) {
        
            // scale the COM-velocities
            for ( j = 0 ; j < nr_mols ; j++ ) {
                for ( k = 0 ; k < 3 ; k++ ) {
                    vcom[k] = ( e.s.partlist[j*3]->v[k] * 15.9994 +
                        e.s.partlist[j*3+1]->v[k] * 1.00794 +
                        e.s.partlist[j*3+2]->v[k] * 1.00794 ) / 1.801528e+1;
                    vcom[k] -= vcom_tot[k];
                    vcom[k] *= ( w - 1.0 );
                    e.s.partlist[j*3]->v[k] += vcom[k];
                    e.s.partlist[j*3+1]->v[k] += vcom[k];
                    e.s.partlist[j*3+2]->v[k] += vcom[k];
                    }
                }
                
            } // apply molecular thermostat
            
        // tabulate the total potential and kinetic energy
        epot = 0.0; ekin = 0.0;
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ ) {
            epot += e.s.cells[cid].epot;
            for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ ) {
                for ( v2 = 0.0 , k = 0 ; k < 3 ; k++ )
                    v2 += e.s.cells[cid].parts[pid].v[k] * e.s.cells[cid].parts[pid].v[k];
                ekin += 0.5 * e.types[ e.s.cells[cid].parts[pid].type ].mass * v2;
                }
            }
            
            
        printf("%i %e %e %e %i %i %.3f ms\n",
        #ifdef CELL
            // e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(double)(toc-tic) / 25000); fflush(stdout);
            e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(double)(toc-tic) / 26664.184); fflush(stdout);
        #else
            // e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,elapsed(toc,tic) / 2300000); fflush(stdout);
            e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,elapsed(toc,tic) / 2199977); fflush(stdout);
        #endif
        
        // print some particle data
        // printf("main: part 13322 is at [ %e , %e , %e ].\n",
        //     e.s.partlist[13322]->x[0], e.s.partlist[13322]->x[1], e.s.partlist[13322]->x[2]);
            
        }
     
    // dump the particle positions, just for the heck of it
    // for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
    //     for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ ) {
    //         for ( k = 0 ; k < 3 ; k++ )
    //             x[k] = e.s.cells[cid].origin[k] + e.s.cells[cid].parts[pid].x[k];
    //         printf("%i %e %e %e\n",e.s.cells[cid].parts[pid].id,x[0],x[1],x[2]);
    //         }
        
    // clean break
    return 0;

    }
