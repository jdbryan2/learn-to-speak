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

/*
class VADPort : public BufferedPort<yarp::sig::Sound> {

protected:

	//data
	DataBuffer &buffer1;		//buffer shared with main thread
	DataBuffer &buffer2;

	//params
	int N;					//decimation factor


public:

	VADPort(DataBuffer &buf1, DataBuffer &buf2, int decimate) : buffer1(buf1),buffer2(buf2), N(decimate) { }

	//callback for incoming position data
	virtual void onRead(yarp::sig::Sound& s) {

		int blockSize = s.getSamples();
		Stamp tStamp;	int status;

		//lock the data buffer for the whole transfer
		buffer1.lock();
		buffer2.lock();
		for (int i = 0; i < blockSize/N; i++) {
			buffer1.push_back((double)s.getSafe(i*N,1)/(double)INT_MAX);
			buffer2.push_back((double)s.getSafe(i*N,0)/(double)INT_MAX);
		}
		buffer1.unlock();
		buffer2.unlock();

	}

};
*/


class VocalTractThread : public RateThread
{
protected:

	ResourceFinder &rf;
	string name;

	BufferedPort<yarp::sig::Vector>  *actuationOut;

	int status;
	Port   * statPort;
	StatusChecker * checker;
	Port   * outPort;

    Speaker * speaker;
    ArtwordControl * controller;
    Artword * apa;

    double fsamp;
    long sample;
    

public:

	VocalTractThread(ResourceFinder &_rf) : RateThread(10), rf(_rf)
	{ }

	virtual bool threadInit()
	{


		name=rf.check("name",Value("vtDriver")).asString().c_str();

		//get robot name and trajectory times. use diff default traj times for icub and sim
        // TODO: These if/else statements should not really matter on current iCub
		//robot = rf.check("robot",Value("nobot")).asString().c_str();

		//open up ports
		actuationOut=new BufferedPort<yarp::sig::Vector>;
		string actuationName="/"+name+"/actuator/out";
		actuationOut->open(actuationName.c_str());

		//stopped = false;


        // set up vocal tract simulator
        ////////////////////////////////////////////////////

        sample = 0;
        fsamp = 8000;
        
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


        //yarp::sig::Vector *areaFunction;
        //Sound *acousticSignal

		//if we have both images
		//if (actuation)
		{
            // setup output variables 
            yarp::sig::Vector &actuator = actuationOut->prepare();
            actuator.resize(kArt_muscle_MAX);

            cout << double(sample)/fsamp << std::endl;
            for(int k = 0; k<kArt_muscle_MAX; k++){
                actuator(k) = apa->getTarget(k, double(sample)/fsamp);
            }
            sample++;
            if( (sample)/fsamp > 0.7) {
                sample= 0;
            }
			//send out, cleanup
			actuationOut->write();

        }
	}

	virtual void threadRelease()
	{

		actuationOut->interrupt();

		actuationOut->close();
        
		delete actuationOut;

        delete apa;

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
