#include <SoftwareSerial.h>

const int ppgPin = 0;
const int gsrPin = 1;

int ppg;	
int gsr;
int canStart;
SoftwareSerial softwareSerial(10, 11); // Bluetooth RX, TX

void setup() {
    Serial.begin(115200);	
    softwareSerial.begin(115200);    
}

void loop() {
    canStart = Serial.parseInt();

    if (canStart == 1) { 
        for (int i = 0; i < 8000; i++) { // We want 8000 samples for each signal per call.
            ppg = analogRead(ppgPin);	
            gsr = analogRead(gsrPin);

            Serial.print(ppg);	
            Serial.print(",");
            Serial.println(gsr);

            // Delays to spread out the sampling rate.
            delayMicroseconds(653); 
        }
    }

    canStart = 0;
}	
