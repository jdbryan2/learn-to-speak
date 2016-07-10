//
//  Sound.cpp
//  vt-model
//
//  Created by William J Wagner on 3/17/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#include "Sound.h"
#include <assert.h>
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>

int Sound::buffer_offset = 0;
float Sound::amplitude = 0;

Sound::Sound(long _numberOfChannels, float _duration, double _samplingFrequency) {
    Initialize( _numberOfChannels, _duration, _samplingFrequency);
}

Sound::Sound() {}

void Sound::Initialize(long _numberOfChannels, float _duration, double _samplingFrequency) {
    assert(!isInitialized);
    assert(_numberOfChannels>=1 && _numberOfChannels<=2);
    // TODO: Look up actual acceptable sampling frequency values
    //assert(_samplingFrequency>=0 && _samplingFrequency<= 1800)
    assert( _duration>0 && _duration<=MAX_DURATION);
    numberOfChannels = _numberOfChannels;
    duration = _duration;
    samplingFrequency = _samplingFrequency;
    numberOfSamples = round(duration*samplingFrequency);
    framesPerBuffer = MAX_BUFFER_LEN;
    z.resize(numberOfSamples);
    isInitialized = true;
}

void Sound::ResetArray(float _duration) {
    assert(isInitialized);
    std::vector<float>().swap(z);   // clear x reallocating
    duration = _duration;
    numberOfSamples = round(duration*samplingFrequency);
    framesPerBuffer = MAX_BUFFER_LEN;
    z.resize(numberOfSamples);
    isInitialized = true;
}

Sound::~Sound() {
}

int Sound::play() {
    PaStream *stream;
    PaError err;
    
    scale();
    printf("Praat Articulatory Synthesis: Playing speech sound.\n");
    /* Initialize library before making any other calls. */
    err = Pa_Initialize();
    if( err != paNoError ) goto error;
    
    /* Open an audio I/O stream. */
    err = Pa_OpenDefaultStream( &stream,
                               0,          /* no input channels */
                               1,          /* stereo output */
                               paFloat32,  /* 32 bit floating point output */
                               samplingFrequency,
                               framesPerBuffer,        /* frames per buffer */
                               paCallback,
                               this );
    if( err != paNoError ) goto error;
    
    err = Pa_StartStream( stream );
    if( err != paNoError ) goto error;
    
    /* Sleep for several seconds. */
    Pa_Sleep((duration+1)*1000);
    // Reset Buffer Offset
    buffer_offset = 0;
    
    err = Pa_StopStream( stream );
    if( err != paNoError ) goto error;
    err = Pa_CloseStream( stream );
    if( err != paNoError ) goto error;
    Pa_Terminate();
    return err;
error:
    Pa_Terminate();
    fprintf( stderr, "An error occured while using the portaudio stream\n" );
    fprintf( stderr, "Error number: %d\n", err );
    fprintf( stderr, "Error message: %s\n", Pa_GetErrorText( err ) );
    return err;
    
}

void Sound::scale() {
    for (int i=0; i<numberOfSamples; i++) {
        if (std::abs(z[i])>amplitude)
            amplitude = std::abs(z[i]);
    }
}

int Sound::save(std::string filepath) {
    std::ofstream out1(filepath);
    if(!out1)
    {
        exit(1);
    }
    out1 << "Sampling Frequency :" ;
    out1 << samplingFrequency;
    out1 << "\n";
    out1 << "Duration :" ;
    out1 << duration;
    out1 << "\n\n";
    
    copy(z.begin(),z.end(),std::ostream_iterator<float>(out1,"\n"));
    out1.close();
    return 0;
}

int Sound::paCallback( const void *inputBuffer, void *outputBuffer,
                      unsigned long framesPerBuffer,
                      const PaStreamCallbackTimeInfo* timeInfo,
                      PaStreamCallbackFlags statusFlags,
                      void *userData )
{
    /* Cast data passed through stream to our structure. */
    Sound *sound = (Sound*)userData;
    float *out = (float*)outputBuffer;
    unsigned int i;
    (void) inputBuffer; /* Prevent unused variable warning. */
    int index;
    
    for( i=0; i<framesPerBuffer; i++ ) {
        index = i + buffer_offset;
        if(index >= sound->numberOfSamples) {
            *out++ = 0;
        }
        else {
            // TODO: Probably need to figure out the proper way to scale sound. Maybe look through how Praat does it.
            *out++ = sound->z[index]/amplitude*SOUND_VOLUME;
        }
    }
    buffer_offset += sound->framesPerBuffer;
    return 0;
}
