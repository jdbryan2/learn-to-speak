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


YARP_DECLARE_DEVICES(icubmod)

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

	virtual bool   updateModule() {

	}


	virtual void run()
	{

		// get both input images

		ImageOf<PixelRgb> *pImgL=portImgL->read(false);
		ImageOf<PixelRgb> *pImgR=portImgR->read(false);
		ImageOf<PixelRgb> *tImg;

		ImageOf<PixelFloat> *pImgBRL;
		ImageOf<PixelFloat> *pImgBRR;
		ImageOf<PixelFloat> *oImg;

		//if we have both images
		if (pImgL && pImgR)
		{

			//set up processing
			yarp::sig::Vector loc;
			pImgBRL = new ImageOf<PixelFloat>;
			pImgBRR = new ImageOf<PixelFloat>;
			pImgBRL->resize(*pImgL);
			pImgBRR->resize(*pImgR);
			Mat * T, * X;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			int biggestBlob;


			ImageOf<PixelRgb> &imgOut= portImgD->prepare();


			//pull out the individual color channels
			PixelRgb pxl;
			float lum, rn, bn, gn, val;
			for (int lr = 0; lr < 2; lr++) {

				if (!lr) {
					tImg = pImgL;
					oImg = pImgBRL;
				}
				else {
					tImg = pImgR;
					oImg = pImgBRR;
				}

				for (int x = 0; x < tImg->width(); x++) {
					for (int y = 0; y < tImg->height(); y++) {

						//normalize brightness (above a given threshold)
						pxl = tImg->pixel(x,y);
						lum = (float)(pxl.r+pxl.g+pxl.b);
						rn = 255.0F * pxl.r/lum;
						gn = 255.0F * pxl.g/lum;
						bn = 255.0F * pxl.b/lum;

						//get the selected color
						switch (color) {
						case 0:
							val = (rn - (gn+bn)/2);
							break;
						case 1:
							val = (gn - (rn+bn)/2);
							break;
						case 2:
							val = (bn - (rn+gn)/2);
							break;
						case 3:
							val = (rn+gn)/2.0 - bn;
							break;
						}
						if (val > 255.0) {
							val = 255.0;
						}
						if (val < 0.0) {
							val = 0.0;
						}
						oImg->pixel(x,y) = val;

					}
				}

				//threshold to find blue blobs
				T = new Mat(oImg->height(), oImg->width(), CV_32F, (void *)oImg->getRawImage());
				threshold(*T, *T, thresh, 255.0, CV_THRESH_BINARY);

				imgOut.copy(*oImg);

				X = new Mat(oImg->height(), oImg->width(), CV_8UC1);
				T->convertTo(*X,CV_8UC1);
				findContours(*X, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

				//find largest blob and its moment
				double maxSize = 0.0;
				biggestBlob = -1;
				double xloc, yloc;
;
				for (int i = 0; i < contours.size(); i++) {
					//do a heuristic check so that only 'compact' blobs are grabbed
					if (abs(contourArea(Mat(contours[i])))/arcLength(Mat(contours[i]), true) > maxSize &&
							abs(contourArea(Mat(contours[i])))/arcLength(Mat(contours[i]), true) > 2.0) {
						maxSize = abs(contourArea(Mat(contours[i])))/arcLength(Mat(contours[i]), true);
						biggestBlob = i;
					}
				}
				if (biggestBlob >= 0) {

					//if a valid object was found, add its location
					Moments m = moments(Mat(contours[biggestBlob]));
					xloc = m.m10/m.m00;
					yloc = m.m01/m.m00;
					loc.push_back(xloc);
					loc.push_back(yloc);

				}

				delete T;
				delete X;

			}

			// load all the audio
			//earPort->read();
			energy_left = 0.0;
			energy_right = 0.0;
			int inc = 0;
			if (buf_left.size() > 0 && buf_right.size()> 0) {
				buf_left.lock();
				buf_right.lock();

				while (buf_left.size() > 0 && buf_right.size() > 0 ) {
					//double temp;
					//temp = buf_left.front();
					energy_left += buf_left.front()*buf_left.front()*100000000;				
					energy_right += buf_right.front()*buf_right.front()*100000000;
					buf_left.pop_front();
					buf_right.pop_front();
					inc++;
				}
				while (buf_left.size() > 0) {buf_left.pop_front();}
				while (buf_right.size() > 0) {buf_right.pop_front();}

				buf_left.unlock();
				buf_right.unlock();
				//M->pushData(signal,inc);
		
				//energy_left = 1;
				printf("energy: %f,\t%f,\t%i\n", energy_left, energy_right, inc);
				
				if (energy_left > 20 || energy_right > 20) {
					yarp::sig::Vector &cPos = portPos->prepare();

					igaze->getFixationPoint(cPos);

					igaze->getAngles(cPos);

					//printf("fixed azelr: %f,\t%f,\t%f\n", cPos[0], cPos[1], cPos[2]);
					
					yarp::sig::Vector nPos(3);
					if(energy_left > energy_right) {
						nPos[0] = -5.0;
					} else {
						nPos[0] = +5.0;
					}
					nPos[1] = 0.0;
					nPos[2] = 0.0;
					if (energy_left < 40 || energy_right < 40) {
						printf("fixed azelr: %f,\t%f,\t%f\n", nPos[0], nPos[1], nPos[2]);
						igaze->lookAtRelAngles(nPos);
					}
					
				}

			}			
			


			//if a blob in both images was detected, go to it
			if (loc.size() == 4) {

				double du, dv;

				//check to see if within acceptable tolerance
				du = (loc[0] - 160 + loc[2] -160)/2.0;
				dv = (loc[1] - 120 + loc[3] -120)/2.0;
				printf("left/right average divergence: %f\n", sqrt(du*du+dv*dv));
				if (sqrt(du*du+dv*dv) < tol) {
				/////////////////////////////////////////////////////////////////////////////////
				//stop tracking command
				/////////////////////////////////////////////////////////////////////////////////
					if (!stopped) {
						
						igaze->stopControl();
						stopped = true;

						//generate a trigger signal indicating that the blob has been focused
						yarp::sig::Vector &cPos = portPos->prepare();
						if (mode) {
							igaze->getFixationPoint(cPos);
							if (verbose) {
								printf("fixated xyz: %f,\t%f,\t%f\n", cPos[0], cPos[1], cPos[2]);
							}
						} else {
							igaze->getAngles(cPos);
							if (verbose) {
								printf("fixed azelr: %f,\t%f,\t%f\n", cPos[0], cPos[1], cPos[2]);
							}
						}
						Bottle tStamp;
						tStamp.clear();
						tStamp.add(Time::now());
						portPos->setEnvelope(tStamp);
						portPos->write();

					}

					



				} else {

					if (!stopped || (sqrt(du*du+dv*dv) > hval*tol)) {
						//continue tracking the object
						yarp::sig::Vector pxl, pxr;
						pxl.push_back(loc[0]);
						pxl.push_back(loc[1]);
						pxr.push_back(loc[2]);
						pxr.push_back(loc[3]);
						igaze->lookAtStereoPixels(pxl,pxr);
						stopped = false;
					}

				}
				draw::addCrossHair(imgOut, PixelRgb(0, 255, 0), loc[0], loc[1], 10);
				draw::addCrossHair(imgOut, PixelRgb(0, 255, 0), loc[2], loc[3], 10);

			}

			//send out, cleanup
			portImgD->write();

			delete pImgBRL;
			delete pImgBRR;

		}
	}

	virtual void threadRelease()
	{

		clientGazeCtrl.close();

		portImgL->interrupt();
		portImgR->interrupt();
		portImgD->interrupt();
		portPos->interrupt();

		portImgL->close();
		portImgR->close();
		portImgD->close();
		portPos->close();

		earPort->close();

		delete portImgL;
		delete portImgR;
		delete portImgD;
		delete portPos;

	}

};

class strBallLocModule: public RFModule
{
protected:
	strBallLocThread *thr;

public:
	strBallLocModule() { }

	virtual bool configure(ResourceFinder &rf)
	{
		Time::turboBoost();

		thr=new strBallLocThread(rf);
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

	strBallLocModule mod;

	return mod.runModule(rf);
}
