//
//  main.cpp
//  vt-model
//
//  Created by William J Wagner on 3/6/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

// #define NDEBUG 1 //disable assertions in the code
// TODO: Create include file and add assert.h. needs to be in only one spot.

#include <iostream>
#include <string>
#include "Speaker.h"
#include "Artword.h"
#include "Control.h"
#include "ArtwordControl.h"
#include "RandomStim.h"
#include "BasePrimControl.h"
#include <gsl/gsl_matrix.h>

using namespace std;

Artword apa () {
    Artword apa(0.5);
    apa.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    apa.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
    apa.setTarget(kArt_muscle_LUNGS,0,0.2);
    apa.setTarget(kArt_muscle_LUNGS,0.1,0);
    apa.setTarget(kArt_muscle_MASSETER,0.25,0.7);
    apa.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.25,0.2);
    return apa;
}

Artword ipa101 () { // Same as apa
    Artword apa(0.5);
    apa.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    apa.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
    apa.setTarget(kArt_muscle_LUNGS,0,0.2);
    apa.setTarget(kArt_muscle_LUNGS,0.1,0);
    apa.setTarget(kArt_muscle_MASSETER,0.25,0.7);
    apa.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.25,0.2);
    return apa;
}

Artword ahh () {
    Artword ahh(0.5);
    ahh.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    ahh.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    ahh.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    ahh.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
    ahh.setTarget(kArt_muscle_LUNGS,0,0.2);
    ahh.setTarget(kArt_muscle_LUNGS,0.1,0);
    //apa.setTarget(kArt_muscle_MASSETER,0.25,0.7);
    //apa.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.25,0.2);
    return ahh;
}

Artword aaa () {
    Artword artw(0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    //artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    //artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
    artw.setTarget(kArt_muscle_LUNGS,0,0.2);
    artw.setTarget(kArt_muscle_LUNGS,0.1,0);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.7);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.7);
    //artw.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.25,0.2);
    return artw;
}

Artword aaatwo () {
    Artword artw(0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    artw.setTarget(kArt_muscle_CRICOTHYROID,0,1.0);
    artw.setTarget(kArt_muscle_CRICOTHYROID,0.5,1.0);
    artw.setTarget(kArt_muscle_LUNGS,0,0.2);
    artw.setTarget(kArt_muscle_LUNGS,0.1,0);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.7);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.7);
    //artw.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.25,0.2);
    return artw;
}

Artword ohh () {
    double length = 1;
    Artword artw(length);
    artw.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,length,0.5);
    artw.setTarget(kArt_muscle_CRICOTHYROID,0,1.0);
    artw.setTarget(kArt_muscle_CRICOTHYROID,length,1.0);
    artw.setTarget(kArt_muscle_LUNGS,0,0.2);
    artw.setTarget(kArt_muscle_LUNGS,0.1,0);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.9);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.9);
    artw.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.0,0.9);
    artw.setTarget(kArt_muscle_ORBICULARIS_ORIS,length,0.9);
    return artw;
}

Artword sss () {
    double length = 1;
    Artword artw(length);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0,0.5);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,0.5);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,1.0);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,1.0);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.9);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.9);
    artw.setTarget(kArt_muscle_LUNGS,0,0.4);
    artw.setTarget(kArt_muscle_LUNGS,0.3,0);
    return artw;
}

Artword khh () {
    double length = 1;
    Artword artw(length);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0,0.5);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,0.5);
    artw.setTarget(kArt_muscle_TRANSVERSE_TONGUE,0,1.0);
    artw.setTarget(kArt_muscle_TRANSVERSE_TONGUE,length,1.0);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.5);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.5);
    artw.setTarget(kArt_muscle_RISORIUS,0,0.3);
    artw.setTarget(kArt_muscle_RISORIUS,length,0.3);
    artw.setTarget(kArt_muscle_LUNGS,0,0.4);
    artw.setTarget(kArt_muscle_LUNGS,0.3,0);
    return artw;
}

// From page 129 of Boersma
// Doesn't work correctly
Artword apa2 () {
    Artword apa(0.5);
    apa.setTarget(kArt_muscle_LUNGS,0,0.1);
    apa.setTarget(kArt_muscle_LUNGS,0.1,0);
    apa.setTarget(kArt_muscle_MASSETER,0.0,-0.4);
    apa.setTarget(kArt_muscle_MASSETER,0.2,0.3);
    apa.setTarget(kArt_muscle_MASSETER,0.3,0.3);
    apa.setTarget(kArt_muscle_MASSETER,0.5,-0.4);
    apa.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.2,0.7);
    apa.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.3,0.7);
    apa.setTarget(kArt_muscle_HYOGLOSSUS, 0.0, 0.4);
    apa.setTarget(kArt_muscle_HYOGLOSSUS, 0.0, 0.4);
    return apa;
}

Artword sigh () {
    Artword articulation(0.5);
    articulation.setTarget(kArt_muscle_LUNGS, 0, 0.1 );
    articulation.setTarget(kArt_muscle_LUNGS, 0.1, 0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
    return articulation;
}

Artword unstable () {
    Artword articulation(1.0);
    articulation.setTarget(kArt_muscle_LUNGS, 0.0, 0.75309590858857711);
    articulation.setTarget(kArt_muscle_LUNGS, 0.27024999999998411, 0.32997949304765417);
    articulation.setTarget(kArt_muscle_LUNGS, 0.41074999999996864, 0.91598820268168613);
    articulation.setTarget(kArt_muscle_LUNGS, 0.68337500000002005, 0.13565701632217769);
    articulation.setTarget(kArt_muscle_LUNGS, 0.68350000000002009, 0.24923117402862585);
    articulation.setTarget(kArt_muscle_LUNGS, 0.84737500000007482, 0.59625697257499666);
    articulation.setTarget(kArt_muscle_LUNGS, 0.99987500000012575, 0.37804446385418006);
    articulation.setTarget(kArt_muscle_LUNGS, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.0, 0.17365220998817027);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.25437499999998586, 0.52140695658586278);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.46299999999996289, 0.6105491714501905);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.605374999999994, 0.15589583879992749);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.78600000000005432, 0.71565150619968299);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.88225000000008647, 0.9038577297367334);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.97650000000011794, 0.44978455269129175);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.99987500000012575, 0.7445054620956737);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.0, 0.0);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.000125, 0.91253099487046496);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.20962499999999076, 0.30351385810450587);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.40412499999996937, 0.75455631478337148);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.77612500000005102, 0.1822617596683829);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.94562500000010762, 0.67539781603278504);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.99987500000012575, 0.89368165812242473);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI,0,0.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.000125, 0.31904929332933152);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.2111249999999906, 0.056208407054591562);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.31549999999997913, 0.060845406774190439);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.52524999999996724, 0.19650379298816162);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.63837500000000502, 0.9939777251549704);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.671250000000016, 0.69876139224334066);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.75875000000004522, 0.20685668492483289);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.99987500000012575, 0.04734473867241823);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_MASSETER, 0.0, 0.0);
    articulation.setTarget(kArt_muscle_MASSETER, 0.000125, 0.79262812955798245);
    articulation.setTarget(kArt_muscle_MASSETER, 0.08325000000000006, 0.55991893947184423);
    articulation.setTarget(kArt_muscle_MASSETER, 0.09462500000000007, 0.50417251459111856);
    articulation.setTarget(kArt_muscle_MASSETER, 0.34649999999997572, 0.81162849613087562);
    articulation.setTarget(kArt_muscle_MASSETER, 0.41162499999996854, 0.39456923513414943);
    articulation.setTarget(kArt_muscle_MASSETER, 0.425624999999967, 0.46455199093588245);
    articulation.setTarget(kArt_muscle_MASSETER, 0.55037499999997563, 0.80183344684177049);
    articulation.setTarget(kArt_muscle_MASSETER, 0.7916250000000562, 0.47311957327652882);
    articulation.setTarget(kArt_muscle_MASSETER, 0.99987500000012575, 0.49231054893179638);
    articulation.setTarget(kArt_muscle_MASSETER, 1.0, 0.0);
    return articulation;
}

Artword unstable2() {
    Artword articulation(1.0);
    articulation.setTarget(kArt_muscle_LUNGS, 0.0, 0.94356413403645789);
    articulation.setTarget(kArt_muscle_LUNGS, 0.28349999999999997, 0.59638202157935238);
    articulation.setTarget(kArt_muscle_LUNGS, 0.73150000000000004, 0.4216740203526364);
    articulation.setTarget(kArt_muscle_LUNGS, 0.86450000000000004, 0.83202332606107865);
    articulation.setTarget(kArt_muscle_LUNGS, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_LUNGS, 1.1486252384186544, 0.31418252832907284);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.0, 0.38691723014282153);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.23275000000000001, 0.44596362383548888);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.326625, 0.7850592968614053);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.56412499999999999, 0.00084446985690646986);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.754, 0.55044039782704146);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 1.0471991097015776, 0.00096087265391242159);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.0, 0.0);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.000125, 0.50331576028852842);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.087374999999999994, 0.023368495304847862);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.229625, 0.85784625600983089);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.58237499999999998, 0.15606742774860799);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.82662500000000005, 0.048803767918438108);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.97750000000000003, 0.24254444818873866);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS,1.1908989766148974,0.99067956464389328);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.0, 0.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.000125, 0.82281003726176805);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.218, 0.82425397138648449);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.35325000000000001, 0.24596988972707653);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.35425000000000001, 0.04692264798176024);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.76324999999999998, 0.89643497397253002);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.90774999999999995, 0.080667759034903224);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 1.3443127287411367, 0.97247717311943593);
    articulation.setTarget(kArt_muscle_MASSETER, 0.0, 0.0);
    articulation.setTarget(kArt_muscle_MASSETER, 0.000125, 0.59886875483083235);
    articulation.setTarget(kArt_muscle_MASSETER, 0.121375, 0.68035917094972265);
    articulation.setTarget(kArt_muscle_MASSETER, 0.53212499999999996, 0.45806205408614242);
    articulation.setTarget(kArt_muscle_MASSETER, 0.62212500000000004, 0.63985934328622407);
    articulation.setTarget(kArt_muscle_MASSETER, 0.98050000000000003, 0.32019867329196977);
    articulation.setTarget(kArt_muscle_MASSETER, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_MASSETER, 1.040003424289184, 0.06688279008427446);
    return articulation;
}

Artword ejective () {
    Artword articulation(0.5);
    articulation.setTarget(kArt_muscle_LUNGS, 0, 0.1 );
    articulation.setTarget(kArt_muscle_LUNGS, 0.1, 0);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);

    articulation.setTarget(kArt_muscle_MASSETER,0.0,-0.3);
    articulation.setTarget(kArt_muscle_MASSETER,0.5,-0.3);
    articulation.setTarget(kArt_muscle_HYOGLOSSUS,0.0,0.5);
    articulation.setTarget(kArt_muscle_HYOGLOSSUS,0.5,0.5);

    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.0,0.0);
    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.1,0.0);
    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.15,1.0);

    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.0,0.5);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.17,0.5);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.2,1.0);

    articulation.setTarget(kArt_muscle_STYLOHYOID,0.0,0.0);
    articulation.setTarget(kArt_muscle_STYLOHYOID,0.22,0.0);
    articulation.setTarget(kArt_muscle_STYLOHYOID,0.27,1.0);

    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.29,1.0);
    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.32,0.0);

    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.35,1.0);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.38,0.5);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);

    articulation.setTarget(kArt_muscle_STYLOHYOID,0.35,1.0);
    articulation.setTarget(kArt_muscle_STYLOHYOID,0.38,0.0);
    articulation.setTarget(kArt_muscle_STYLOHYOID,0.5,0.0);
    return articulation;
}

// bilabial click (functional phonology pg 140)
Artword click () {
    Artword articulation(0.5);
    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.0,0.9);
    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.5,0.9);

    articulation.setTarget(kArt_muscle_MASSETER,0.0,0.25);
    articulation.setTarget(kArt_muscle_MASSETER,0.2,0.25);
    articulation.setTarget(kArt_muscle_MASSETER,0.3,-0.25); // TODO: Should these actually be negative or not?
    articulation.setTarget(kArt_muscle_MASSETER,0.5,-0.25);

    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.0,0.75);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.2,0.75);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.3,0.0);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.5,0.0);
    return articulation;
}

// Initialize artword with an articulation then set all articulators to 0.5
Artword half(Articulation art,double utterance_length) {
    Artword artw(utterance_length);
    for (int i=0; i<kArt_muscle_MAX; i++) {
        artw.setTarget(i, 0.0, art[i]);
        //artw.setTarget(i, 0.02, 0.5);
        //artw.setTarget(i, utterance_length, 0.5);
    }
    artw.setTarget(0, utterance_length,0.51893);
    artw.setTarget(1, utterance_length,0.52312);
    artw.setTarget(2, utterance_length,0.49115);
    artw.setTarget(3, utterance_length,0.54254);
    artw.setTarget(4, utterance_length,0.56644);
    artw.setTarget(5, utterance_length,0.47614);
    artw.setTarget(6, utterance_length,0.55359);
    artw.setTarget(7, utterance_length,0.51222);
    artw.setTarget(8, utterance_length,0.50521);
    artw.setTarget(9, utterance_length,0.46648);
    artw.setTarget(10, utterance_length,0.47237);
    artw.setTarget(10, utterance_length,0.44816);
    artw.setTarget(12, utterance_length, 0.5365);
    artw.setTarget(13, utterance_length,0.53544);
    artw.setTarget(14, utterance_length,0.51018);
    artw.setTarget(15, utterance_length,0.45486);
    artw.setTarget(16, utterance_length,0.48739);
    artw.setTarget(17, utterance_length,0.49523);
    artw.setTarget(18, utterance_length,0.49068);
    artw.setTarget(19, utterance_length, 0.5484);
    artw.setTarget(20, utterance_length,0.54635);
    artw.setTarget(21, utterance_length,0.49587);
    artw.setTarget(22, utterance_length,0.49096);
    artw.setTarget(23, utterance_length,0.50532);
    artw.setTarget(24, utterance_length,0.51403);
    artw.setTarget(25, utterance_length,0.48275);
    artw.setTarget(26, utterance_length,0.49424);
    artw.setTarget(27, utterance_length,0.51676);
    artw.setTarget(28, utterance_length,0.48322);
    return artw;
}

Artword ipa109 () {
    double length = 0.5;
    Artword artw(length);
    artw.setTarget(kArt_muscle_LUNGS,0,0.4);
    artw.setTarget(kArt_muscle_LUNGS,0.05,0.2);
    artw.setTarget(kArt_muscle_LUNGS,0.15,0.4);
    artw.setTarget(kArt_muscle_LUNGS,length,0.4);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,1);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.8);
    artw.setTarget(kArt_muscle_MASSETER,length,0.8);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0.0,1);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0.1,0.8);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,0.8);
    //artw.setTarget(kArt_muscle_TRANSVERSE_TONGUE,0,0.2);
    //artw.setTarget(kArt_muscle_TRANSVERSE_TONGUE,length,0.2);
    return artw;
}

Artword ipa140 () {
    double length = 0.5;
    Artword artw(length);
    artw.setTarget(kArt_muscle_LUNGS,0,0.3);
    artw.setTarget(kArt_muscle_LUNGS,0.05,0.2);
    artw.setTarget(kArt_muscle_LUNGS,length,0.2);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,1);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.3);
    artw.setTarget(kArt_muscle_MASSETER,length,0.3);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0.0,0.5);
    //artw.setTarget(kArt_muscle_STYLOGLOSSUS,0.1,0.3);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,0.5);
    //artw.setTarget(kArt_muscle_TRANSVERSE_TONGUE,0,0.2);
    //artw.setTarget(kArt_muscle_TRANSVERSE_TONGUE,length,0.2);
    return artw;
}

Artword ipa114 () {
    double length = 0.5;
    Artword artw(length);
    artw.setTarget(kArt_muscle_LUNGS,0,0.4);
    artw.setTarget(kArt_muscle_LUNGS,0.05,0.2);
    artw.setTarget(kArt_muscle_LUNGS,length,0.2);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,0);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,0);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.9);
    artw.setTarget(kArt_muscle_MASSETER,length,0.9);
    artw.setTarget(kArt_muscle_INTERARYTENOID,0.0,0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,length,0.5);
    //artw.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.0,0.5);
    //artw.setTarget(kArt_muscle_ORBICULARIS_ORIS,length,0.5);
    //artw.setTarget(kArt_muscle_STYLOGLOSSUS,0,1);
    //artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,1);
    //artw.setTarget(kArt_muscle_UPPER_TONGUE,0,1);
    //artw.setTarget(kArt_muscle_UPPER_TONGUE,length,1);
    artw.setTarget(kArt_muscle_LOWER_TONGUE,0,1);
    artw.setTarget(kArt_muscle_LOWER_TONGUE,length,1);
    return artw;
}

Artword ipa134 () {
    double length = 0.5;
    Artword artw(length);
    artw.setTarget(kArt_muscle_LUNGS,0,0.3);
    artw.setTarget(kArt_muscle_LUNGS,0.05,0.2);
    artw.setTarget(kArt_muscle_LUNGS,length,0.2);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,1);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.35);
    artw.setTarget(kArt_muscle_MASSETER,length,0.35);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0.0,0.6);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,0.6);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,0.2);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,0.2);
    return artw;
}

Artword ipa132 () {
    double length = 0.5;
    Artword artw(length);
    /*artw.setTarget(kArt_muscle_LUNGS,0,0.4);
    artw.setTarget(kArt_muscle_LUNGS,0.05,0.2);
    artw.setTarget(kArt_muscle_LUNGS,length,0.2);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,1);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.9);
    artw.setTarget(kArt_muscle_MASSETER,length,0.9);
    artw.setTarget(kArt_muscle_RISORIUS,0.0,1);
    artw.setTarget(kArt_muscle_RISORIUS,length,1);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,1);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,1);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0,0.1);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0.1,0.1);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,0.1);*/
    
    /*artw.setTarget(kArt_muscle_LUNGS,0,0.4);
    artw.setTarget(kArt_muscle_LUNGS,0.05,0.1);
    artw.setTarget(kArt_muscle_LUNGS,length,0.1);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,1);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.4);
    artw.setTarget(kArt_muscle_MASSETER,length,0.4);
    artw.setTarget(kArt_muscle_RISORIUS,0.0,1);
    artw.setTarget(kArt_muscle_RISORIUS,length,1);
    artw.setTarget(kArt_muscle_LOWER_TONGUE,0,0.1);
    artw.setTarget(kArt_muscle_LOWER_TONGUE,length,0.1);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0,0.5);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,0.5);*/
    
    artw.setTarget(kArt_muscle_LUNGS,0,0.4);
    artw.setTarget(kArt_muscle_LUNGS,0.05,.3);
    artw.setTarget(kArt_muscle_LUNGS,length,0.3);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,1);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.35);
    artw.setTarget(kArt_muscle_MASSETER,length,0.35);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,.8);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,.8);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0,1);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,1);
    return artw;
}

Artword ipa133 () {
    double length = 0.5;
    Artword artw(length);
    artw.setTarget(kArt_muscle_LUNGS,0,0.4);
    artw.setTarget(kArt_muscle_LUNGS,0.05,.3);
    artw.setTarget(kArt_muscle_LUNGS,length,0.3);
    artw.setTarget(kArt_muscle_INTERARYTENOID,0.0,0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,length,0.5);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,1);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.2);
    artw.setTarget(kArt_muscle_MASSETER,length,0.2);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,1);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,1);
    artw.setTarget(kArt_muscle_UPPER_TONGUE,0.0,.1);
    artw.setTarget(kArt_muscle_UPPER_TONGUE,length,.1);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0,1);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,1);
    return artw;
}

Artword ipa142 () {
    double length = 0.5;
    Artword artw(length);
    artw.setTarget(kArt_muscle_LUNGS,0,0.3);
    artw.setTarget(kArt_muscle_LUNGS,0.05,.25);
    artw.setTarget(kArt_muscle_LUNGS,length,0.22);
    //artw.setTarget(kArt_muscle_INTERARYTENOID,0.0,0.5);
    //artw.setTarget(kArt_muscle_INTERARYTENOID,length,0.5);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,1);
    artw.setTarget(kArt_muscle_MYLOHYOID,0.0,0.2);
    artw.setTarget(kArt_muscle_MYLOHYOID,length,0.2);
    //artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,1);
    //artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,1);
    //artw.setTarget(kArt_muscle_UPPER_TONGUE,0.0,.1);
    //artw.setTarget(kArt_muscle_UPPER_TONGUE,length,.1);
    artw.setTarget(kArt_muscle_SPHINCTER,0.0,.6);
    artw.setTarget(kArt_muscle_SPHINCTER,length,.6);
    //artw.setTarget(kArt_muscle_STYLOGLOSSUS,0,1);
    //artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,1);
    return artw;
}

Artword ipa301 () { // OK can't get tounge forward enough.
    double length = 0.5;
    Artword artw(length);
    artw.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,length,0.5);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,1.0);
    artw.setTarget(kArt_muscle_LUNGS,0,0.3);
    artw.setTarget(kArt_muscle_LUNGS,0.05,0.2);
    artw.setTarget(kArt_muscle_LUNGS,length,0.2);
    artw.setTarget(kArt_muscle_MASSETER,0.0,0.1);
    artw.setTarget(kArt_muscle_MASSETER,length,0.1);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0,0.95);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,0.95);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,1);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,1);
    return artw;
}

Artword ipa304 () { // Not Great
    double length = 0.8;
    Artword artw(length);
    /*artw.setTarget(kArt_muscle_INTERARYTENOID,0,0.4);
    artw.setTarget(kArt_muscle_INTERARYTENOID,length,0.4);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    artw.setTarget(kArt_muscle_LEVATOR_PALATINI,length,1.0);
    artw.setTarget(kArt_muscle_LUNGS,0,0.3);
    artw.setTarget(kArt_muscle_LUNGS,0.05,0.25);
    artw.setTarget(kArt_muscle_LUNGS,length,0.25);
    artw.setTarget(kArt_muscle_MYLOHYOID,0.0,1);
    artw.setTarget(kArt_muscle_MYLOHYOID,length,1);
    
    //artw.setTarget(kArt_muscle_HYOGLOSSUS,0,1);
    //artw.setTarget(kArt_muscle_HYOGLOSSUS,length,1);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,1);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,1);
    
    //artw.setTarget(kArt_muscle_STERNOHYOID,0,1);
    //artw.setTarget(kArt_muscle_STERNOHYOID,length,1);
    //artw.setTarget(kArt_muscle_VERTICAL_TONGUE,0,1);
    //artw.setTarget(kArt_muscle_VERTICAL_TONGUE,length,1); */
    
    artw.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,length,0.5);
    //artw.setTarget(kArt_muscle_LUNGS,0,0.2);
    //artw.setTarget(kArt_muscle_LUNGS,0.1,0);
    artw.setTarget(kArt_muscle_LUNGS,0,0.3);
    //artw.setTarget(kArt_muscle_LUNGS,0.05,0.28);
    artw.setTarget(kArt_muscle_LUNGS,length,0.22);
    artw.setTarget(kArt_muscle_MYLOHYOID,0.0,.1);
    artw.setTarget(kArt_muscle_MYLOHYOID,length,.1);
    artw.setTarget(kArt_muscle_SPHINCTER,0.0,.7);
    artw.setTarget(kArt_muscle_SPHINCTER,length,.7);
    //artw.setTarget(kArt_muscle_MASSETER,0.0,0.7);
    //artw.setTarget(kArt_muscle_MASSETER,length,0);
    artw.setTarget(kArt_muscle_HYOGLOSSUS,0,1);
    artw.setTarget(kArt_muscle_HYOGLOSSUS,length,1);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,1);
    artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,1);
    return artw;
}

Artword ipa305 () { // pretty good
    double length = 0.8;
    Artword artw(length);
    
    artw.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,length,0.5);
    artw.setTarget(kArt_muscle_LUNGS,0,0.3);
    artw.setTarget(kArt_muscle_LUNGS,0.05,0.25);
    artw.setTarget(kArt_muscle_LUNGS,length,0.26);
    artw.setTarget(kArt_muscle_MYLOHYOID,0.0,.1);
    artw.setTarget(kArt_muscle_MYLOHYOID,length,.1);
    artw.setTarget(kArt_muscle_SPHINCTER,0.0,.7);
    artw.setTarget(kArt_muscle_SPHINCTER,length,.7);
    artw.setTarget(kArt_muscle_HYOGLOSSUS,0,.3);
    artw.setTarget(kArt_muscle_HYOGLOSSUS,length,.3);
    //artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,1);
    //artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,1);
    return artw;
}

Artword ipa316 () { 
    double length = 0.8;
    Artword artw(length);
    
    artw.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    artw.setTarget(kArt_muscle_INTERARYTENOID,length,0.5);
    artw.setTarget(kArt_muscle_LUNGS,0,0.3);
    artw.setTarget(kArt_muscle_LUNGS,0.05,0.25);
    artw.setTarget(kArt_muscle_LUNGS,length,0.25);
    artw.setTarget(kArt_muscle_MYLOHYOID,0.0,.1);
    artw.setTarget(kArt_muscle_MYLOHYOID,length,.1);
    artw.setTarget(kArt_muscle_SPHINCTER,0.0,.3);
    artw.setTarget(kArt_muscle_SPHINCTER,length,.3);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,0,.3);
    artw.setTarget(kArt_muscle_STYLOGLOSSUS,length,.3);
    //artw.setTarget(kArt_muscle_GENIOGLOSSUS,0,.1);
    //artw.setTarget(kArt_muscle_GENIOGLOSSUS,length,.1);
    return artw;
}

void simulate(Speaker* speaker, Control* controller) {
    // pass the articulator positions into the speaker BEFORE initializing the simulation
    // otherwise, we just get a strong discontinuity after the first instant
    Articulation art;
    controller->InitialArt(art);
    
    // initialize the simulation and tell it how many seconds to buffer
    speaker->InitSim(controller->utterance_length, art);
    
    cout << "Simulating...\n";
    
    while (speaker->NotDone())
    {
        controller->doControl(speaker);
        // generate the next acoustic sample
        speaker->IterateSim();
    }
    cout << "Done!\n";
}

void sim_artword(Speaker* speaker, Artword* artword, std::string artword_name,double log_period, std::string prefix)
    {
    cout << "Artword Starting----------\n";
    ArtwordControl awcontrol(artword);
    // Initialize the data logger
    prefix = prefix + "artword_logs/";
    speaker->ConfigDataLogger(prefix + artword_name + to_string(1) + ".log",log_period);
    simulate(speaker, &awcontrol);
        speaker->SaveSound(prefix + artword_name + "_sound" + to_string(1) + ".log");
    for(int i =0; i< 10; i++)
    {
        //cout << speaker->result->z[100*i] << ", ";
    }
    //cout << endl;
    
    // simple interface for playing back the sound that was generated
    int input =  0;// set to zero to test the speed of simulation.
    while (true)
    {
        cout << "Press (1) to play the sound or any key to quit.\n";
        std::cin.clear();
        //cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin >> input;
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        if(input == 1) {
            speaker->Speak();
            input =0;
        } else {
            break;
        }
    }
    cout << "Artword Ending-------\n";
}

void random_stim_trials(Speaker* speaker,double utterance_length, double log_period, double hold_mean, int num_trials, bool end_interp, std::string prefix) {
    std::normal_distribution<double>::param_type hold_time_param(hold_mean,0.25);
    std::uniform_real_distribution<double>::param_type activation_param(0.0,1.0);
    RandomStim rs(utterance_length, speaker->fsamp, hold_time_param, activation_param,end_interp);
    for (int trial=1; trial <= num_trials; trial++)
    {
        // Generate a new random artword
        rs.NewArtword();
        // Initialize the data logger
        speaker->ConfigDataLogger(prefix + "logs/datalog" + to_string(trial)+ ".log",log_period);
        cout << "Trial " << trial << "\n";
        simulate(speaker, &rs);
        speaker->Speak();
        speaker->SaveSound(prefix + "logs/sound" + to_string(trial) + ".log");
    }
}

void prim_control(Speaker* speaker,double utterance_length, double log_period, std::string prefix, std::string config,int prim_enabled) {
    Artword artw = apa();
    Articulation art = {};
    artw.intoArt(art, 0.0);
    BasePrimControl prim(utterance_length,log_period,art,prefix+config+"/",prim_enabled);
    // Initialize the data logger
    speaker->ConfigDataLogger(prefix +config + "/prim_logs/primlog" + to_string(prim_enabled)+ ".log",log_period);
    simulate(speaker, &prim);
    speaker->Speak();
    speaker->SaveSound(prefix + config + "/prim_logs/sound" + to_string(prim_enabled) + ".log");
}

void AreaRefControl(Speaker* speaker, double log_freq, double log_period, std::string prefix) {
    Artword artw = apa();
    Articulation art = {};
    artw.intoArt(art, 0.0);
    double utterance_length = artw.totalTime;
    
    std::ifstream f_stream;
    FILE* f_stream_mat;
    std::string filename;
    filename = prefix + "Aref.alog";
    f_stream_mat = fopen(filename.c_str(), "r");
    gsl_vector * Aref = gsl_vector_alloc(MAX_NUMBER_OF_TUBES*(utterance_length*log_freq+1));
    gsl_vector_fscanf(f_stream_mat, Aref);
    fclose(f_stream_mat);
    // Make control longer than sample
    utterance_length+=2,0;
    BasePrimControl prim(utterance_length,log_period,art,prefix,0,Aref);
    // Initialize the data logger
    speaker->ConfigDataLogger(prefix + "prim_logs/Areflog" + to_string(1)+ ".log",log_period);
    simulate(speaker, &prim);
    speaker->Speak();
    speaker->SaveSound(prefix + "prim_logs/Arefsound" + to_string(1) + ".log");
}

int main()
{
    double sample_freq = 8000;
    int oversamp = 70;
    int number_of_glottal_masses = 2;
    Speaker female("Female",number_of_glottal_masses, sample_freq, oversamp);
    //std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/test____/");
    //int lognum = 1;
    //double utterance_length = 2;
    double desired_log_freq = 50;
    int log_period = floor(sample_freq/desired_log_freq);
    double log_freq = sample_freq/log_period;
    // 1.) Create Artword to track
    /*Artword artword = khh();
    std::string artword_name = "khh";
    //Artword artword = unstable2();
    //std::string artword_name = "unstable2_artword";
    sim_artword(&female, &artword,artword_name,log_period,prefix);*/
    //Artword artword = apa();
    //std::string artword_name = "apa";
    //Artword artword = ahh();
    //std::string artword_name = "ahh";
    //Artword artword = aaa();
    //std::string artword_name = "aaa";
    //Artword artword = aaatwo();
    //std::string artword_name = "aaatwo";
    //Artword artword = ohh();
    //std::string artword_name = "ohh";
    //Artword artword = sss();
    //std::string artword_name = "sss";
    //Artword artword = khh();
    //std::string artword_name = "khh";
    //std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testBatch1000/");
    //Artword artword = ipa109();
    //std::string artword_name = "ipa109_ex";
    //Artword artword = ipa114();
    //std::string artword_name = "ipa114_ex";
    //Artword artword = ipa132();
    //std::string artword_name = "ipa132_ex";
    //Artword artword = ipa133();
    //std::string artword_name = "ipa133_ex";
    //Artword artword = ipa301();
    //std::string artword_name = "ipa301_ex";
    //Artword artword = ipa304();
    //std::string artword_name = "ipa304_ex";
    //Artword artword = ipa305();
    //std::string artword_name = "ipa305_ex";
    //Artword artword = ipa316();
    //std::string artword_name = "ipa316_ex";
    //Artword artword = ipa140();
    //std::string artword_name = "ipa140_ex";
    //Artword artword = ipa134();
    //std::string artword_name = "ipa134_ex";
    //Artword artword = ipa142();
    //std::string artword_name = "ipa142_ex";
    Artword artword = ipa101();
    std::string artword_name = "ipa101_ex";
    std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testStim3Batch300/");
    sim_artword(&female, &artword,artword_name,log_period,prefix);
    
    // 2.) Generate Randomly Stimulated data trials
    
    /*int num_trials1 = 1000;
    double hold_mean1 = 0.1;
    bool end_interp1 = true;
    double utterance_length1 = 0.5;
    std::string prefix1 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testBatch1000/");
    random_stim_trials(&female,utterance_length1,log_period,hold_mean1,num_trials1,end_interp1,prefix1);*/
    
    /*int num_trials1 = 50;
    double hold_mean1 = 0.2;
    bool end_interp1 = false;
    double utterance_length1 = 0.5;
    std::string prefix1 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testStim1BatchNoRand50/");
    random_stim_trials(&female,utterance_length1,log_period,hold_mean1,num_trials1,end_interp1,prefix1);*/
    
    /*int num_trials1 = 300;
    double hold_mean1 = 0.2;
    bool end_interp1 = false;
    double utterance_length1 = 0.5;
    std::string prefix1 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testStim1BatchNoRand300/");
    random_stim_trials(&female,utterance_length1,log_period,hold_mean1,num_trials1,end_interp1,prefix1);*/
    
    /*int num_trials1 = 50;
    double hold_mean1 = 0.2;
    bool end_interp1 = false;
    double utterance_length1 = 0.5;
    std::string prefix1 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testStim1Batch50/");
    random_stim_trials(&female,utterance_length1,log_period,hold_mean1,num_trials1,end_interp1,prefix1);*/
    
    /*int num_trials1 = 300;
    double hold_mean1 = 0.2;
    bool end_interp1 = false;
    double utterance_length1 = 0.5;
    std::string prefix1 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testStim1Batch300/");
    random_stim_trials(&female,utterance_length1,log_period,hold_mean1,num_trials1,end_interp1,prefix1);
    
    int num_trials2 = 300;
    double hold_mean2 = 0.1;
    bool end_interp2 = false;
    double utterance_length2 = 0.5;
    std::string prefix2 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testStim2Batch300/");
    random_stim_trials(&female,utterance_length2,log_period,hold_mean2,num_trials2,end_interp2,prefix2);
    
    int num_trials3 = 300;
    double hold_mean3 = 0.1;
    bool end_interp3 = true;
    double utterance_length3 = 0.5;
    std::string prefix3 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testStim3Batch300/");
    random_stim_trials(&female,utterance_length3,log_period,hold_mean3,num_trials3,end_interp3,prefix3);*/
    
    /*int num_trials1 = 30;
    double hold_mean1 = 0.1;
    bool end_interp1 = true;
    double utterance_length1 = 2;
    std::string prefix1 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testRandArt1/");
    random_stim_trials(&female,utterance_length1,log_period,hold_mean1,num_trials1,end_interp1,prefix1);*/
    
    /*int num_trials1 = 200;
    double hold_mean1 = 0.2;
    bool end_interp1 = false;
    double utterance_length1 = 0.5;
    std::string prefix1 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testBatch1/");
    random_stim_trials(&female,utterance_length1,log_period,hold_mean1,num_trials1,end_interp1,prefix1);
    int num_trials2 = 200;
    double hold_mean2 = 0.1;
    bool end_interp2 = true;
    double utterance_length2 = 0.5;
    std::string prefix2 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testBatch2/");
    random_stim_trials(&female,utterance_length2,log_period,hold_mean2,num_trials2,end_interp2,prefix2);
    int num_trials3 = 100;
    double hold_mean3 = 0.2;
    bool end_interp3 = false;
    double utterance_length3 = 1;
    std::string prefix3 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testBatch3/");
    random_stim_trials(&female,utterance_length3,log_period,hold_mean3,num_trials3,end_interp3,prefix3);
    int num_trials4 = 100;
    double hold_mean4 = 0.1;
    bool end_interp4 = true;
    double utterance_length4 = 1;
    std::string prefix4 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testBatch4/");
    random_stim_trials(&female,utterance_length4,log_period,hold_mean4,num_trials4,end_interp4,prefix4);
    int num_trials5 = 200;
    double hold_mean5 = 0.2;
    bool end_interp5 = false;
    double utterance_length5 = 0.3;
    std::string prefix5 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testBatch5/");
    random_stim_trials(&female,utterance_length5,log_period,hold_mean5,num_trials5,end_interp5,prefix5);
    int num_trials6 = 200;
    double hold_mean6 = 0.1;
    bool end_interp6 = true;
    double utterance_length6 = 0.3;
    std::string prefix6 ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testBatch6/");
    random_stim_trials(&female,utterance_length6,log_period,hold_mean6,num_trials6,end_interp6,prefix6);*/

    //random_stim_trials(&female,utterance_length,log_period,hold_mean,num_trials,prefix);
    
    // 3.) Perform MATLAB DFA to find primitives and generate Aref of 1.)
    
    // 4.) Perform Primitive Control based on IC only
    //int lognum = 8;
    
    /*double utterance_length = 2;
    std::string config = "tubart-medium_original_scale8";
    std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testBatch1000/");
    // Move prim_logs inside of config directory
    for (int i=0; i<=8; i++) {
        prim_control(&female, utterance_length, log_period,prefix,config,i);
    }*/
    
    /*int lognum = 4;
    double utterance_length = 4;
    std::string config = "scale_fix_perm_no_smooth8";
    std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testBatch2/");
    // Move prim_logs inside of config directory
    prim_control(&female, utterance_length, log_period,prefix,config,lognum);

    /*int lognum = 0;
    double utterance_length = 2;
    std::string config = "short8";
    std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testRandArt1/");
    // Move prim_logs inside of config directory
    prim_control(&female, utterance_length, log_period,prefix,config,lognum);*/
    
    /*int lognum = 0;
    double utterance_length = 2;
    std::string config = "original8";
    std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/testRevised1/");
    // Move prim_logs inside of config directory
    prim_control(&female, utterance_length, log_period,prefix,config,lognum);*/
    
    // 5.) Perform Area Function Tracking of 1.)
    //AreaRefControl(&female, log_freq, log_period,prefix);
    
    // 6.) Testing idea that average of Xf is really making the prim control make sound
    /*Artword artword_init = apa();
     Articulation articulation_init;
     artword_init.intoArt(articulation_init, 0.0);
     Artword artword = half(articulation_init,utterance_length);
     std::string artword_name = "half";
     sim_artword(&female, &artword,artword_name,log_period,prefix);*/
    
    return 0;
}
