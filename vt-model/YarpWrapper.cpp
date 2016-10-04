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


#include <yarp/os/all.h>

#include <yarp/sig/Vector.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/ImageFile.h>
#include <yarp/sig/ImageDraw.h>

#include <yarp/dev/Drivers.h>
#include <yarp/dev/ControlBoardInterfaces.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/PolyDriver.h>

#include <yarp/sig/Sound.h>

#include <yarp/math/Math.h>

#include <string>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <deque>

#define TIMEOUT 5.0 
#define NUM_ART 29

//namespaces
using namespace std;
using namespace cv;
using namespace yarp;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;


//YARP_DECLARE_DEVICES(icubmod)

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

class VADPort : public BufferedPort<Sound> {

protected:

	//data
	DataBuffer &buffer1;		//buffer shared with main thread
	DataBuffer &buffer2;

	//params
	int N;					//decimation factor


public:

	VADPort(DataBuffer &buf1, DataBuffer &buf2, int decimate) : buffer1(buf1),buffer2(buf2), N(decimate) { }

	//callback for incoming position data
	virtual void onRead(Sound& s) {

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


class VocalTractThread : public RateThread
{
protected:

	ResourceFinder &rf;
	string name;


	BufferedPort<Sound> *acousticOut;
	BufferedPort<yarp::sig::Vector>  *areaOut;
	BufferedPort<yarp::sig::Vector>  *actuation;

	int status;
	Port   * statPort;
	StatusChecker * checker;
	Port   * outPort;

public:

	VocalTractThread(ResourceFinder &_rf) : RateThread(50), rf(_rf)
	{ }

	virtual bool threadInit()
	{


		name=rf.check("name",Value("VocalTract")).asString().c_str();

		//get robot name and trajectory times. use diff default traj times for icub and sim
        // TODO: These if/else statements should not really matter on current iCub
		robot = rf.check("robot",Value("nobot")).asString().c_str();
		if (robot == "icubSim") {
			neckTT = rf.check("nt",Value(0.6)).asDouble();
			eyeTT = rf.check("et",Value(0.1)).asDouble();
		}
		else if (robot == "icub") {
			neckTT = rf.check("nt",Value(1.0)).asDouble();
			eyeTT = rf.check("et",Value(0.5)).asDouble();
		}
		else {
			printf("No robot name specified, using real iCub default trajectory times\n");
			neckTT = rf.check("nt",Value(1.0)).asDouble();
			eyeTT = rf.check("et",Value(0.5)).asDouble();
		}

		//open up ports
		acousticOut=new BufferedPort<Sound>;
		string acousticName="/"+name+"/acoustic";
		acousticOut->open(acousticName.c_str());

		areaOut=new BufferedPort<yarp::sig::Vector>;
		string areaName="/"+name+"/area";
		areaOut->open(areaName.c_str());

		actuationIn=new BufferedPort<yarp::sig::Vector>;
		string actuationName="/"+name+"/actuator/in";
		actuationIn->open(actuationName.c_str());
        actuationIn->useCallBack();

		stopped = false;


		//set up status checking port
		statPort = new Port;
		checker = new StatusChecker(&status);
		string statName = sendPort + "/status";
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

        //yarp::sig::Vector *areaFunction;
        //Sound *acousticSignal

		//if we have both images
		if (actuation)
		{
            // setup variables 
            yarp::sig:Vector &areaFunction = areaOut->prepare();
            Sound &acousticSignal = acousticOut->prepare();

            // call simulator
            // and get it's outputs


			//send out, cleanup
			areaFunction->write();
			acousticSignal->write();


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

	YARP_REGISTER_DEVICES(icubmod)

	Network yarp;

	if (!yarp.checkNetwork())
		return -1;

	ResourceFinder rf;

	rf.configure("ICUB_ROOT",argc,argv);

	VocalTractModule mod;

	return mod.runModule(rf);
}
