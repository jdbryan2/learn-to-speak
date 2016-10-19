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
#include <yarp/math/Math.h>
#include <yarp/sig/Vector.h>
#include <yarp/sig/Sound.h>

#include <string>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <deque> // unsure if necessary

#define TIMEOUT 5.0 
#define NUM_ART 29

//namespaces
using namespace std;
using namespace yarp;
using namespace yarp::os;
//using namespace yarp::sig; // namespace for yarp::sig::Sound will cause cause conflicts with vt definition of Sound
//using namespace yarp::dev;


//YARP_DECLARE_DEVICES(icubmod)

// unsure if necessary
class StatusChecker : public PortReader {

protected:

	int * active;

public:

	StatusChecker(int * active_) : active(active_) {}

	virtual bool read(ConnectionReader& connection) {

		Bottle in, out;
		bool ok = in.read(connection);
		if (!ok) {
			return false;
		}
		out.add(*active);
		ConnectionWriter *returnToSender = connection.getWriter();
		if (returnToSender!=NULL) {
			out.write(*returnToSender);
		}
		return true;

	}

};

class DataBuffer : public deque<double> {

private:

	Semaphore mutex;

public:

	void lock()   { mutex.wait(); }
	void unlock() { mutex.post(); }

};


class VocalTractThread : public RateThread
{
protected:

	ResourceFinder &rf;
	string name;


	BufferedPort<yarp::sig::Vector> *acousticOut;
	BufferedPort<yarp::sig::Vector>  *areaOut;
	BufferedPort<yarp::sig::Vector>  *actuationIn;

	int status;
	Port   * statPort;
	StatusChecker * checker;
	Port   * outPort;

    Speaker * speaker;
    ArtwordControl * controller;
    Artword * apa;
    

public:

    // rate of this thread should be at least as fast as the driver
    // ideally the driver will run slightly slower so that the input 
    // doesn't get flooded
	VocalTractThread(ResourceFinder &_rf) : RateThread(1), rf(_rf)
	{ }

	virtual bool threadInit()
	{


		name=rf.check("name",Value("vtSim")).asString().c_str();

		//open up ports
		acousticOut=new BufferedPort<yarp::sig::Vector>;
		string acousticName="/"+name+"/acoustic:o";
		acousticOut->open(acousticName.c_str());

		areaOut=new BufferedPort<yarp::sig::Vector>;
		string areaName="/"+name+"/area:o";
		areaOut->open(areaName.c_str());

		actuationIn=new BufferedPort<yarp::sig::Vector>;
		string actuationName="/"+name+"/actuator:i";
		actuationIn->open(actuationName.c_str());
        //actuationIn->useCallback(); // unnecessary since we don't define one



        // set up vocal tract simulator
        ////////////////////////////////////////////////////
        double sample_freq = 8000;
        int oversamp = 70;
        int number_of_glottal_masses = 2;
        speaker  = new Speaker("Female",number_of_glottal_masses, sample_freq, oversamp);

        apa = new Artword(0.7);
        apa->setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
        apa->setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
        apa->setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
        apa->setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
        apa->setTarget(kArt_muscle_LUNGS,0,0.2);
        apa->setTarget(kArt_muscle_LUNGS,0.1,0);
        apa->setTarget(kArt_muscle_MASSETER,0.25,0.7);
        apa->setTarget(kArt_muscle_ORBICULARIS_ORIS,0.25,0.2);
        apa->setTarget(kArt_muscle_LUNGS, 0.5, 0);
        apa->setTarget(kArt_muscle_LUNGS, 0.7, 0.2);


        controller = new ArtwordControl(apa);

        Articulation art;
        controller->InitialArt(art);
        speaker->InitSim(controller->utterance_length, art);


        ////////////////////////////////////////////////////
        

		//set up status checking port
		statPort = new Port;
		checker = new StatusChecker(&status);
		string statName = "/"+ name + "/status";
		statPort->open(statName.c_str());
		statPort->setReader(*checker);

		status = 0;
		return true;

	}

	virtual bool   updateModule()
    {

	}


	virtual void run()
	{

		// get both input images

        yarp::sig::Vector *actuation=actuationIn->read(false);


		//if we have the actuator
		if (actuation != NULL)
        //if (speaker->NotDone()) // this should actually loop some fixed number of blocks based on the update size
		{

            // setup output variables 
            yarp::sig::Vector &areaFunction = areaOut->prepare();
            yarp::sig::Vector &acousticSignal = acousticOut->prepare();


            // this should run some number of times? maybe...
            {
                // run next step of control inputs
                cout << actuation->data() << std::endl;
                for(int k = 0; k<kArt_muscle_MAX; k++){
                    speaker->art[k] = (*actuation)[k];
                    //cout << (*actuation)[k] << std::endl; // debug
                }

                // iterate simulator
                speaker->IterateSim();

                // loop back to start if we hit the of the buffer
                speaker->LoopBack();
            }


            // resize acousticSignal and put in samples
            acousticSignal.resize(1); // (samples, channels) # of samples should correspond to loop above
            acousticSignal(0) = speaker->getLastSample();

            // load area function 
            double temp[89];
            speaker->getAreaFcn(temp);

            // and pass into output variable
            areaFunction.resize(89);
            for(int k=0;  k<89; k++){
                areaFunction(k) = temp[k];
            }
            areaFunction.resize(95);

			//send out, cleanup
			areaOut->write();
			acousticOut->write();


		} else {
           // speaker->Speak();
        }
	}

	virtual void threadRelease()
	{

		acousticOut->interrupt();
		areaOut->interrupt();
		actuationIn->interrupt();

		acousticOut->close();
		areaOut->close();
		actuationIn->close();
        
		delete acousticOut;
		delete areaOut;
		delete actuationIn;

        delete apa;
        delete controller;
        delete speaker;

	}

};

class VocalTractModule: public RFModule
{
protected:
	VocalTractThread *thr;

public:
	 VocalTractModule() { }

	virtual bool configure(ResourceFinder &rf)
	{
		Time::turboBoost();

		thr=new VocalTractThread(rf);
		if (!thr->start())
		{
			delete thr;
			return false;
		}

		return true;
	}

	virtual bool close()
	{
		thr->stop();
		delete thr;

		return true;
	}

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

	VocalTractModule mod;

	return mod.runModule(rf);
}
