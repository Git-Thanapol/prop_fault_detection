#include <SPI.h>
#include <Ethernet2.h>
#include <EthernetUdp2.h>
#include <Servo.h>

byte mac[] = {0xA8, 0x61, 0x0A, 0xAE, 0x64, 0xD4};

// Udp Comm
IPAddress ip(192, 168, 0, 200);
IPAddress gateway(192, 168, 0, 1);
IPAddress subnet(255, 255, 255, 0);
IPAddress DNS(192, 168, 0, 1);
unsigned int localPort = 8888;
char packetBuffer[UDP_TX_PACKET_MAX_SIZE];
EthernetUDP Udp;

// Thruster
byte stbdPin = 11;
byte portPin = 9;
Servo stbd;
Servo port;

void setup() {
  stbd.attach(stbdPin);
  port.attach(portPin);
  stbd.writeMicroseconds(1500);
  port.writeMicroseconds(1500);
  
  Ethernet.begin(mac, ip);
  Udp.begin(localPort);

  Serial.begin(9600);
}

void loop() {
  int packetSize = Udp.parsePacket();
  if (packetSize)
  {
    Udp.read(packetBuffer, UDP_TX_PACKET_MAX_SIZE);

    // Header == "$CO"
    if (packetBuffer[0] == '$' && packetBuffer[1] == 'C' && packetBuffer[2] == 'O') {
      char tmpBuf[10];
      int tmpBufIdx = 0;

      int valList[] = {0, 0};
      int valListIdx = 0;

      int stbdVal = 1500;
      int portVal = 1500;

      // Case : "$CO1"
      if (packetBuffer[3] == '1') {
        Serial.print("$CO1 : ");

        // Convert to int (Delimiter ',')
        for (int i = 5; i < packetSize; i++) {
          if (packetBuffer[i] == ',') {
            valList[valListIdx] = atoi(tmpBuf);
            memset(tmpBuf, '\0', 10);
            valListIdx++;

            if (valListIdx > 1) {
              break;
            }

            tmpBufIdx = 0;
          }
          else {
            tmpBuf[tmpBufIdx] = packetBuffer[i];
            tmpBufIdx++;
          }
        }

        // ±100 -> PWM signal
        stbdVal = map(valList[0], -100, 100, 1200, 1800);
        portVal = map(valList[1], -100, 100, 1800, 1200);
      }

      // Case : "$CO2"
      if (packetBuffer[3] == '2') {
        Serial.print("$CO2 : ");

        // Convert to int (Delimiter ',')
        for (int i = 5; i < packetSize; i++) {
          if (packetBuffer[i] == ',') {
            valList[valListIdx] = atoi(tmpBuf);
            memset(tmpBuf, '\0', 10);
            valListIdx++;

            if (valListIdx > 1) {
              break;
            }

            tmpBufIdx = 0;
          }
          else {
            tmpBuf[tmpBufIdx] = packetBuffer[i];
            tmpBufIdx++;
          }
        }

        stbdVal = valList[0];
        portVal = valList[1];
      }

      // if commands are out of range, stop
      if (stbdVal > 1800 || stbdVal < 1200) {
        stbdVal = 1500;
      }
      if (portVal > 1800 || portVal < 1200) {
        portVal = 1500;
      }

      Serial.print("STBD = ");
      Serial.print(stbdVal);
      Serial.print(", PORT = ");
      Serial.println(portVal);

      // send command to thruster 
      stbd.writeMicroseconds(stbdVal);
      port.writeMicroseconds(portVal);
    }
  }

  delay(10);
}


