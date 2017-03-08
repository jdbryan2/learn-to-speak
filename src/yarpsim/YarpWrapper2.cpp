//
//  YarpWrapper.cpp
//  vt-model
//
//  Created by Jacob D. Bryan 9/23/16
//  Copyright © 2016 Team Jacob. All rights reserved.
//

// #define NDEBUG 1 //disable assertions in the code
//

#include <iostream>
#include "Speaker.h"
#include "Artword.h"
#include "Control.h"
#include "ArtwordControl.h"


#include <yarp/os/all.h>
#include <yarp/os/ConstString.h>
#include <yarp/math/Math.h>
#include <yarp/sig/Vector.h>
#include <yarp/sig/Sound.h>

#include <string>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#define TIMEOUT 5.0 
#define NUM_ART 29

//namespaces
using namespace std;
using namespace yarp;
using namespace yarp::os;
//using namespace yarp::sig; // namespace for yarp::sig::Sound will cause cause conflicts with vt definition of Sound
//using namespace yarp::dev;


//YARP_DECLARE_DEVICES(icubmod)



class vtPort : public BufferedPort<yarp::sig::Vector> {

protected:

	string name;
    int sample;
    int max_sample;

	BufferedPort<yarp::os::Bottle> *acousticOut;
	BufferedPort<yarp::os::Bottle>  *areaOut;

    Speaker * speaker;
    ArtwordControl * controller;
    Artword * apa;
    

public:

    // rate of this thread should be at least as fast as the driver
    // ideally the driver will run slightly slower so that the input 
    // doesn't get flooded
	vtPort(const char * bname, int msample) : sample(0), name(bname), max_sample(msample)
	{


		//name=rf.check("name",Value("vtSim")).asString().c_str();

		//open up ports
		acousticOut=new BufferedPort<yarp::os::Bottle>;
		string acousticName="/"+name+"/acoustic:o";
		acousticOut->open(acousticName.c_str());

		areaOut=new BufferedPort<yarp::os::Bottle>;
		string areaName="/"+name+"/area:o";
		areaOut->open(areaName.c_str());

		//actuationIn=new BufferedPort<yarp::sig::Vector>;
		//string actuationName="/"+name+"/actuator:i";
		//actuationIn->open(actuationName.c_str());


        cout << "Setting up simulator."<<endl;

        // set up vocal tract simulator
        ////////////////////////////////////////////////////
        double sample_freq = 8000;
        int oversamp = 70;
        int number_of_glottal_masses = 2;
        speaker  = new Speaker("Female",number_of_glottal_masses, sample_freq, oversamp);

        cout << "Init Sim." <<endl;
        Articulation art;
        speaker->InitSim(1.0, art);
        ////////////////////////////////////////////////////
        cout << "Done constructing."<<endl;

	}

	~vtPort()
	{

		acousticOut->interrupt();
		areaOut->interrupt();

		acousticOut->close();
		areaOut->close();
        
		delete acousticOut;
		delete areaOut;

        delete apa;
        delete controller;
        delete speaker;

	}

	virtual void onRead(yarp::sig::Vector& actuation)
	{

        // setup output variables 
        yarp::os::Bottle &areaFunction = areaOut->prepare();
        yarp::os::Bottle &acousticSignal = acousticOut->prepare();


        // this should run some number of times? maybe...
        {
            // run next step of control inputs
            //cout << actuation.data() << std::endl;
            for(int k = 0; k<kArt_muscle_MAX; k++){
                speaker->art[k] = actuation[k];
                //cout << (*actuation)[k] << std::endl; // debug
            }

            // iterate simulator
            speaker->IterateSim();

            // loop back to start if we hit the of the buffer
            speaker->LoopBack();
        }


        // resize acousticSignal and put in samples
        //acousticSignal.resize(1); // (samples, channels) # of samples should correspond to loop above
        //acousticSignal(0) = speaker->getLastSample();
        acousticSignal.addDouble(speaker->getLastSample());

        // load area function 
        double temp[89];
        speaker->getAreaFcn(temp);

        // and pass into output variable
        //areaFunction.resize(89);
        for(int k=0;  k<89; k++){
            //areaFunction(k) = temp[k];
            areaFunction.addDouble(temp[k]);
        }
        cout << areaFunction.toString() << endl;

        areaOut->writeStrict();
        areaFunction.clear();

        if (acousticSignal.size() > max_sample) {
            //send out, cleanup
            acousticOut->writeStrict();
            acousticSignal.clear();
        }

	}


};

class DummyModule: public RFModule
{
protected:

public:

	virtual bool configure(ResourceFinder &rf) { return true; }
	virtual bool close() { return true; }
	virtual double getPeriod()    { return 1.0;  }
	virtual bool   updateModule() { return true; }
};


int main(int argc, char *argv[])
{

//	YARP_REGISTER_DEVICES(icubmod)

	Network yarp;

	if (!yarp.checkNetwork())
		return -1;

	ResourceFinder rf;

	rf.configure("ICUB_ROOT",argc,argv);

	string bname = rf.check("name",Value("vtSim")).asString().c_str();
    string pname("/"+bname+"/actuator:i");
    vtPort VocalTract(bname.c_str(), 1000); // write sound in 1000 sample chunks
    VocalTract.open(pname.c_str());
    VocalTract.setStrict(); // process every message without skipping
    VocalTract.useCallback();




	DummyModule mod;
	return mod.runModule(rf);
}
