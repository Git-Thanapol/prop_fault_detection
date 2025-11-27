#include <Servo.h>

// --- CONFIGURATION ---
const byte stbdPin = 11;
const byte portPin = 9;

// Propeller Direction Configuration
// Change these to "CW" or "CCW" based on your physical blades
String stbd_blade = "CCW"; 
String port_blade = "CCW"; 

// --- GLOBALS ---
Servo stbd;
Servo port;

bool running = false;
int targetSpeed = 1500; 

// Activation Flags (Default both ON)
bool stbd_active = true;
bool port_active = true;

void setup() {
  // Initialize Serial USB
  Serial.begin(9600);
  while (!Serial) { ; } // Wait for connection

  // Attach Motors
  stbd.attach(stbdPin);
  port.attach(portPin);

  // Safety Init
  stopMotors();

  Serial.println("READY: Enter command (start, stop, set <val>, activate <11>)");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    processCommand(command);
  }

  // We only write to motors if system is "running". 
  // Otherwise, we force 1500 (Neutral) for safety.
  if (running) {
    driveMotors(targetSpeed);
  } else {
    // Redundant safety write
    stbd.writeMicroseconds(1500);
    port.writeMicroseconds(1500);
  }
}

void driveMotors(int pwmVal) {
  int diff = pwmVal - 1500;
  int inverse_pwm = 1500 - diff;

  // --- STARBOARD LOGIC ---
  if (!stbd_active) {
    stbd.writeMicroseconds(1500);
  } else {
    if (stbd_blade == "CW") stbd.writeMicroseconds(inverse_pwm);
    else stbd.writeMicroseconds(pwmVal);
  }

  // --- PORT LOGIC ---
  if (!port_active) {
    port.writeMicroseconds(1500);
  } else {
    if (port_blade == "CW") port.writeMicroseconds(inverse_pwm);
    else port.writeMicroseconds(pwmVal);
  }
}

void stopMotors() {
  targetSpeed = 1500;
  stbd.writeMicroseconds(1500);
  port.writeMicroseconds(1500);
}

void processCommand(String cmd) {
  cmd.trim(); // Remove whitespace

  if (cmd.equalsIgnoreCase("start")) {
    running = true;
    Serial.println("ACK: Started");
  } 
  else if (cmd.equalsIgnoreCase("stop")) {
    running = false;
    stopMotors();
    Serial.println("ACK: Stopped");
  } 
  else if (cmd.startsWith("set ")) {
    int val = cmd.substring(4).toInt();
    // Constrain safety limits
    if (val >= 1100 && val <= 1900) {
      targetSpeed = val;
      Serial.print("ACK: Speed ");
      Serial.println(targetSpeed);
    }
  } 
  else if (cmd.startsWith("activate ")) {
    String mode = cmd.substring(9);
    if (mode == "11") { stbd_active = true; port_active = true; }
    else if (mode == "10") { stbd_active = true; port_active = false; }
    else if (mode == "01") { stbd_active = false; port_active = true; }
    else if (mode == "00") { stbd_active = false; port_active = false; }
    Serial.print("ACK: Active Mode ");
    Serial.println(mode);
  }
}