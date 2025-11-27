#include <SPI.h>
#include <Ethernet2.h>
#include <EthernetUdp2.h>
#include <Servo.h>

// Ethernet settings
byte mac[] = {0xA8, 0x61, 0x0A, 0xAE, 0x64, 0xD4};
IPAddress ip(192, 168, 0, 200);
IPAddress gateway(192, 168, 0, 1);
IPAddress subnet(255, 255, 255, 0);
IPAddress DNS(192, 168, 0, 1);

// UDP communication settings
unsigned int localPort = 8888;
char packetBuffer[UDP_TX_PACKET_MAX_SIZE];
EthernetUDP Udp;

// Thruster settings
const byte stbdPin = 11;
const byte portPin = 9;
Servo stbd;
Servo port;

// Neutral PWM signal for thrusters
const int NEUTRAL_PWM = 1500;
const int MIN_PWM = 1200;
const int MAX_PWM = 1800;

void setup() {
    // Attach thrusters to their respective pins
    stbd.attach(stbdPin);
    port.attach(portPin);
    
    // Initialize thrusters in neutral position
    stbd.writeMicroseconds(NEUTRAL_PWM);
    port.writeMicroseconds(NEUTRAL_PWM);
    
    // Start Ethernet and UDP communication
    Ethernet.begin(mac, ip);
    Udp.begin(localPort);
    
    // Initialize Serial Monitor
    Serial.begin(9600);
}

void loop() {
    int packetSize = Udp.parsePacket();
    if (packetSize) {
        // Read received packet
        Udp.read(packetBuffer, UDP_TX_PACKET_MAX_SIZE);
        
        // Validate packet header
        if (packetBuffer[0] == '$' && packetBuffer[1] == 'C' && packetBuffer[2] == 'O') {
            int stbdVal = NEUTRAL_PWM;
            int portVal = NEUTRAL_PWM;
            
            // Extract command type
            char commandType = packetBuffer[3];
            
            if (commandType == '1' || commandType == '2') {
                Serial.print("$CO");
                Serial.print(commandType);
                Serial.print(" : ");
                
                // Extract values from packet
                int valList[2] = {0, 0};
                parsePacketValues(packetBuffer, packetSize, valList);
                
                if (commandType == '1') {
                    // Convert input range (-100 to 100) to PWM range (1200 to 1800)
                    stbdVal = map(valList[0], -100, 100, MIN_PWM, MAX_PWM);
                    portVal = map(valList[1], -100, 100, MAX_PWM, MIN_PWM);
                } else {
                    // Directly use received PWM values
                    stbdVal = valList[0];
                    portVal = valList[1];
                }
                
                // Ensure PWM values stay within valid range
                stbdVal = constrain(stbdVal, MIN_PWM, MAX_PWM);
                portVal = constrain(portVal, MIN_PWM, MAX_PWM);
                
                Serial.print("STBD = ");
                Serial.print(stbdVal);
                Serial.print(", PORT = ");
                Serial.println(portVal);
                
                // Send command to thrusters
                stbd.writeMicroseconds(stbdVal);
                port.writeMicroseconds(portVal);
            }
        }
    }
    
    delay(10); // Short delay to avoid excessive CPU usage
}

void parsePacketValues(char *buffer, int size, int *values) {
    char tmpBuf[10] = {0};
    int tmpBufIdx = 0;
    int valListIdx = 0;
    
    for (int i = 5; i < size && valListIdx < 2; i++) {
        if (buffer[i] == ',' || buffer[i] == '\0') {
            values[valListIdx] = atoi(tmpBuf);
            memset(tmpBuf, 0, sizeof(tmpBuf));
            valListIdx++;
            tmpBufIdx = 0;
        } else {
            tmpBuf[tmpBufIdx++] = buffer[i];
        }
    }
}
