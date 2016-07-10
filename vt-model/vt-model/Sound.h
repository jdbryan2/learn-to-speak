//
//  Sound.h
//  vt-model
//
//  Created by William J Wagner on 3/17/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#ifndef Sound_h
#define Sound_h

#include "portaudio.h"
#include <vector>

#define MAX_DURATION 5
#define MAX_BUFFER_LEN 256
#define SOUND_VOLUME 0.4

class Sound {
public:
    std::vector<float> z;
    long numberOfChannels;
    float duration;
    float samplingFrequency;
    int numberOfSamples;
    unsigned long framesPerBuffer;
    static float amplitude;
private:
    bool isInitialized = false;
public:
    Sound(long _numberOfChannels, float _duration, double _samplingFrequency);
    Sound();
    ~Sound();
    void Initialize(long _numberOfChannels, float _duration, double _samplingFrequency);
    void ResetArray(float _duration);
    bool IsInitialized() {return isInitialized;};
    int play();
    void scale();
    int save(std::string filepath);
    // TODO: Not sure this variable should be static or not.
    //       I think it makes sense since we are only playing 1 sound at a time.
    static int buffer_offset;
    static int paCallback( const void *inputBuffer, void *outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void *userData );
};

#endif /* Sound_h */
